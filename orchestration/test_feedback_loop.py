#!/usr/bin/env python3
"""
Feedback Loop Testing Script for Orchestration ReAct Loop.

This script tests the ReAct loop behavior in the orchestration graph.
It mocks the four agents and simulates different scenarios to verify that
the loop behaves as expected.

Scenarios:
1. All agents succeed on first iteration -> loop should not repeat.
2. Some agents have gaps (no output) -> loop should repeat up to max iterations.
3. Some agents have errors -> loop should repeat.
4. Max iterations reached -> loop should stop even with gaps.

The script patches the agent imports in the orchestration.nodes module
to return mocked outputs or raise exceptions.

Usage:
    python orchestration/test_feedback_loop.py

Output:
    Prints a test report with pass/fail status for each scenario.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List, Callable

# Ensure the repo root is on sys.path
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from orchestration.graph import build_graph, run, _compiled_graph
from orchestration.state import OrchestrationState


def mock_agent_output(agent_name: str, ticker: str = "AAPL") -> Dict[str, Any]:
    """Generate a minimal valid output for a given agent."""
    if agent_name == "business_analyst":
        return {
            "ticker": ticker,
            "qualitative_summary": f"Mock BA analysis for {ticker}.",
            "sentiment_verdict": {"label": "Bullish"},
            "citations": []
        }
    elif agent_name == "quant_fundamental":
        return {
            "ticker": ticker,
            "quantitative_summary": f"Mock QF analysis for {ticker}.",
            "value_factors": {"pe_trailing": 25.0},
            "quality_factors": {"piotroski_f_score": 8}
        }
    elif agent_name == "financial_modelling":
        return {
            "ticker": ticker,
            "quantitative_summary": f"Mock FM analysis for {ticker}.",
            "valuation": {"dcf": {"intrinsic_value_base": 150.0}},
            "technicals": {"rsi_14": 55.0}
        }
    elif agent_name == "web_search":
        return {
            "ticker": ticker,
            "sentiment_signal": "Neutral",
            "breaking_news": [],
            "unknown_risk_flags": []
        }
    else:
        return {"ticker": ticker, "summary": "Mock output"}


class AgentMockManager:
    """Context manager to mock agent imports and planner."""
    
    def __init__(self, success_agents: List[str], error_agents: List[str], 
                 gap_agents: List[str], max_iterations: int = 3):
        self.success_agents = success_agents
        self.error_agents = error_agents
        self.gap_agents = gap_agents
        self.max_iterations = max_iterations
        self.patches = []
        
    def _make_mock_for_agent(self, agent_name: str) -> Callable:
        """Create a mock function for the given agent."""
        if agent_name in self.error_agents or agent_name in self.gap_agents:
            def error_mock(*args, **kwargs):
                raise Exception(f"Mock gap/error for {agent_name}")
            return error_mock
        else:
            # success agent
            def success_mock(ticker=None, *args, **kwargs):
                # Return a single dict (not list) because agent functions return dict
                return mock_agent_output(agent_name, ticker or "AAPL")
            return success_mock
    
    def __enter__(self):
        # Store original env var for sequential mode
        self._original_seq = os.environ.get("ORCHESTRATION_SEQUENTIAL")
        # Set sequential mode to avoid threading issues
        os.environ["ORCHESTRATION_SEQUENTIAL"] = "1"
        
        # Clear the compiled graph cache so new patches take effect
        import orchestration.graph
        orchestration.graph._compiled_graph = None
        
        # Patch plan_query where it is called (nodes.py imports it as a local name)
        def mock_plan_query(query):
            return {
                "ticker": "AAPL",
                "tickers": ["AAPL"],
                "complexity": self.max_iterations,
                "agents": ["business_analyst", "quant_fundamental", "financial_modelling", "web_search"],
                "run_business_analyst": True,
                "run_quant_fundamental": True,
                "run_web_search": True,
                "run_financial_modelling": True,
            }
        # nodes.py does `from .llm import plan_query` so patch the local name there
        planner_patch = patch("orchestration.nodes.plan_query", side_effect=mock_plan_query)
        self.patches.append(planner_patch)
        
        # Patch router embedding to avoid loading sentence-transformers
        def mock_router_embed(text):
            return None  # causes semantic router to return None, skipping cache
        router_embed_patch = patch("orchestration.llm._router_embed", side_effect=mock_router_embed)
        self.patches.append(router_embed_patch)
        
        # Patch few-shot query fetch to avoid DB calls
        few_shot_patch = patch("orchestration.llm._fetch_top_successful_queries", return_value=[])
        self.patches.append(few_shot_patch)
        
        # Patch episodic memory to avoid DB calls (node_planner uses dynamic imports so
        # patching the module-level names is correct)
        episodic_lookup_patch = patch("orchestration.episodic_memory.lookup_similar_failures", return_value={})
        self.patches.append(episodic_lookup_patch)
        episodic_hints_patch = patch("orchestration.episodic_memory.build_preemptive_plan_hints", return_value={})
        self.patches.append(episodic_hints_patch)
        episodic_record_patch = patch("orchestration.episodic_memory.record_failure", return_value=None)
        self.patches.append(episodic_record_patch)
        
        # Patch summarise_results where it is called (nodes.py local import)
        def mock_summarise(*args, **kwargs):
            return "Mock final summary generated by test."
        summarizer_patch = patch("orchestration.nodes.summarise_results", side_effect=mock_summarise)
        self.patches.append(summarizer_patch)
        
        # Patch ticker extraction to return AAPL (node_planner uses dynamic import)
        ticker_patch = patch("agents.quant_fundamental.agent.extract_tickers_from_prompt", 
                             return_value=["AAPL"])
        self.patches.append(ticker_patch)
        
        # Patch data availability check to avoid external calls (dynamic import in node_planner)
        data_avail_patch = patch("orchestration.data_availability.check_all", 
                                 return_value={"postgres": True, "qdrant": True, "neo4j": True, "ollama": True})
        self.patches.append(data_avail_patch)
        # Patch ticker_data_profile to avoid errors
        profile_patch = patch("orchestration.data_availability.ticker_data_profile",
                              return_value={"postgres": True, "qdrant": True, "neo4j": True, "ollama": True})
        self.patches.append(profile_patch)

        
        # Patch node_parallel_agents directly.
        # node_parallel_agents inlines its own closures (_run_ba, _run_qf, etc.) and
        # dispatches them via ThreadPoolExecutor — patching the individual node_*
        # functions has no effect on it.  Instead we replace the whole dispatcher
        # with a mock that honours the success/gap/error lists while preserving the
        # "skip already-completed agents" semantics that the ReAct loop relies on.
        _success_agents = self.success_agents
        _error_agents   = self.error_agents
        _gap_agents     = self.gap_agents

        def mock_node_parallel_agents(state):
            errors  = dict(state.get("agent_errors") or {})
            steps   = list(state.get("react_steps") or [])

            # Carry over any outputs that already exist from previous passes
            ba_outputs = list(state.get("business_analyst_outputs") or [])
            qf_outputs = list(state.get("quant_fundamental_outputs") or [])
            ws_outputs = list(state.get("web_search_outputs") or [])
            fm_outputs = list(state.get("financial_modelling_outputs") or [])

            _agent_cfg = {
                "business_analyst":   (state.get("run_business_analyst",   False), ba_outputs),
                "quant_fundamental":  (state.get("run_quant_fundamental",  False), qf_outputs),
                "web_search":         (state.get("run_web_search",         False), ws_outputs),
                "financial_modelling":(state.get("run_financial_modelling",False), fm_outputs),
            }

            for agent_name, (enabled, existing_outputs) in _agent_cfg.items():
                if not enabled:
                    continue
                # Skip agents that already succeeded on a previous pass
                if existing_outputs:
                    continue

                if agent_name in _error_agents or agent_name in _gap_agents:
                    errors[agent_name] = f"Mock gap/error for {agent_name}"
                    steps.append({
                        "tool": agent_name,
                        "input": {"tickers": ["AAPL"]},
                        "observation": f"Error: Mock gap/error for {agent_name}",
                    })
                else:
                    outputs = [mock_agent_output(agent_name, ticker="AAPL")]
                    steps.append({
                        "tool": agent_name,
                        "input": {"tickers": ["AAPL"]},
                        "observation": "Success",
                    })
                    if agent_name == "business_analyst":
                        ba_outputs[:] = outputs
                    elif agent_name == "quant_fundamental":
                        qf_outputs[:] = outputs
                    elif agent_name == "web_search":
                        ws_outputs[:] = outputs
                    elif agent_name == "financial_modelling":
                        fm_outputs[:] = outputs

            return {
                **state,
                "business_analyst_outputs":    ba_outputs,
                "quant_fundamental_outputs":   qf_outputs,
                "web_search_outputs":           ws_outputs,
                "financial_modelling_outputs":  fm_outputs,
                "business_analyst_output":     ba_outputs[0]  if ba_outputs  else None,
                "quant_fundamental_output":    qf_outputs[0]  if qf_outputs  else None,
                "web_search_output":           ws_outputs[0]  if ws_outputs  else None,
                "financial_modelling_output":  fm_outputs[0]  if fm_outputs  else None,
                "react_steps": steps,
                "agent_errors": errors,
            }

        # Patch the name as used inside graph.py (imported via `from .nodes import ...`)
        # so that build_graph() captures the mock when add_node("parallel_agents", ...) runs.
        parallel_agents_patch = patch(
            "orchestration.graph.node_parallel_agents",
            side_effect=mock_node_parallel_agents,
        )
        self.patches.append(parallel_agents_patch)

        # Patch telemetry to avoid DB calls
        telemetry_patch = patch("orchestration.nodes._log_telemetry", return_value=None)
        self.patches.append(telemetry_patch)
        
        # Enter all patches
        for p in self.patches:
            p.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in reversed(self.patches):
            p.__exit__(exc_type, exc_val, exc_tb)
        # Restore original env var
        if self._original_seq is None:
            os.environ.pop("ORCHESTRATION_SEQUENTIAL", None)
        else:
            os.environ["ORCHESTRATION_SEQUENTIAL"] = self._original_seq


def run_scenario(name: str, success_agents=None, error_agents=None, gap_agents=None,
                 max_iterations=3, expected_loops=1):
    """Run a single test scenario and verify loop behavior."""
    if success_agents is None:
        success_agents = []
    if error_agents is None:
        error_agents = []
    if gap_agents is None:
        gap_agents = []
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"Success agents: {success_agents}")
    print(f"Error agents: {error_agents}")
    print(f"Gap agents: {gap_agents}")
    print(f"Max iterations: {max_iterations}")
    print(f"Expected loops: {expected_loops}")
    
    with AgentMockManager(success_agents, error_agents, gap_agents, max_iterations):
        try:
            print("  Starting orchestration run...")
            result = run("Test query about AAPL")
            actual_loops = result.get("react_iteration", 0)  # react_iteration is incremented once per pass by node_react_check
            gaps_present = False
            for agent in ["business_analyst", "quant_fundamental", "financial_modelling", "web_search"]:
                output_key = f"{agent}_outputs"
                if output_key in result and not result[output_key]:
                    gaps_present = True
                    break
            
            print(f"Result: loops={actual_loops}, final_summary_present={'final_summary' in result}")
            print(f"Agent outputs:")
            for agent in ["business_analyst", "quant_fundamental", "financial_modelling", "web_search"]:
                outputs = result.get(f"{agent}_outputs", [])
                print(f"  {agent}: {len(outputs)} output(s)")
            
            # Verification
            passed = True
            if actual_loops != expected_loops:
                print(f"FAIL: Expected {expected_loops} loops, got {actual_loops}")
                passed = False
            # Check that successful agents have output
            for agent in success_agents:
                if not result.get(f"{agent}_outputs"):
                    print(f"FAIL: Success agent {agent} has no output")
                    passed = False
            # Check that error agents are in agent_errors (if not retried successfully)
            # This is more complex; we'll skip for now.
            
            if passed:
                print("PASS")
                return True
            else:
                print("FAIL")
                return False
                
        except Exception as e:
            print(f"ERROR during execution: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all test scenarios and generate a report."""
    print("Feedback Loop Test Report")
    print("=" * 60)
    
    results = []
    
    # Scenario 1: All agents succeed on first iteration
    # Should run only one pass (no gaps, no errors)
    results.append((
        "All agents succeed",
        run_scenario(
            "All agents succeed",
            success_agents=["business_analyst", "quant_fundamental", "financial_modelling", "web_search"],
            error_agents=[],
            gap_agents=[],
            max_iterations=3,
            expected_loops=1
        )
    ))

    
    # Scenario 2: One agent has gap (no output), others succeed
    # Should loop up to max_iterations (3) because gap persists
    # But note: gap agent returns empty list each time, so loop continues until max iterations
    results.append((
        "One gap agent (persistent)",
        run_scenario(
            "One gap agent",
            success_agents=["quant_fundamental", "financial_modelling", "web_search"],
            error_agents=[],
            gap_agents=["business_analyst"],
            max_iterations=3,
            expected_loops=3  # will loop until max iterations
        )
    ))
    
    # Scenario 3: One agent error, others succeed
    # Should loop up to max_iterations because error persists (mock error each time)
    results.append((
        "One error agent",
        run_scenario(
            "One error agent",
            success_agents=["quant_fundamental", "financial_modelling", "web_search"],
            error_agents=["business_analyst"],
            gap_agents=[],
            max_iterations=3,
            expected_loops=3
        )
    ))
    
    # Scenario 4: Max iterations = 1, gap present
    # Should stop after 1 loop despite gap
    results.append((
        "Max iterations = 1 with gap",
        run_scenario(
            "Max iterations = 1 with gap",
            success_agents=["quant_fundamental", "financial_modelling", "web_search"],
            error_agents=[],
            gap_agents=["business_analyst"],
            max_iterations=1,
            expected_loops=1
        )
    ))
    
    # Scenario 5: Two gaps, max iterations = 2
    # Should loop twice then stop
    results.append((
        "Two gaps, max iterations = 2",
        run_scenario(
            "Two gaps",
            success_agents=["financial_modelling", "web_search"],
            error_agents=[],
            gap_agents=["business_analyst", "quant_fundamental"],
            max_iterations=2,
            expected_loops=2
        )
    ))
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = 0
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if passed == len(results):
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())