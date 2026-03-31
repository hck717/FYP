# Test Metrics Summary

## Overview

**Date:** March 29, 2026  
**Total Tests:** 141 (79 prompts + 62 integration)  
**Passed:** 141  
**Failed:** 0  
**Passing Rate:** 100%

### Test Suite Duration
| Suite | Tests | Duration |
|-------|-------|----------|
| Prompts | 79 | 0.18s |
| Integration | 62 | 361.88s (6 min 2 sec) |
| **Total** | **141** | **~6 minutes** |

---

## Prompts Tests (79 tests - 0.18s)

Prompts tests verify that AI agent prompts produce correct outputs using mock data. These tests run very quickly as they don't require external services.

### Test Categories

#### 1. Business Analyst Prompts (test_ba_prompts.py) - 12 tests
Tests the Business Analyst agent's prompt engineering for moat analysis and earnings grading.

| Test | Meaning | Duration |
|------|---------|----------|
| test_crag_grading_prompt | Verifies CRAG (Chunk Relevance Assessment Grading) returns correct grades (CORRECT/INCORRECT/AMBIGUOUS) based on chunk-query relevance | <1ms |
| test_moat_analysis_structured_output | Verifies moat analysis prompt returns properly structured JSON with required fields | <1ms |
| test_sentiment_analysis_output | Verifies sentiment analysis prompt extracts correct sentiment from text | <1ms |
| test_qualitative_summary_generation | Tests generation of qualitative summaries from data | <1ms |
| test_citations_contain_chunk_ids | Verifies citations in outputs reference valid chunk IDs | <1ms |
| test_confidence_threshold_grading | Tests grading behavior at different confidence thresholds | <1ms |
| test_empty_chunks_handling | Tests behavior when no relevant chunks are found | <1ms |
| test_missing_chunk_id_handling | Tests handling of citations with missing chunk IDs | <1ms |
| test_ba_output_integration | End-to-end integration test for BA output format | <1ms |

#### 2. Citation Accuracy (test_citation_accuracy.py) - 18 tests
Verifies that citations in agent outputs correctly reference source documents in Neo4j/PostgreSQL.

| Test | Meaning | Duration |
|------|---------|----------|
| test_broker_report_citation_format | Broker research citations have correct format | <1ms |
| test_broker_rating_extraction | Extracts broker ratings correctly from citations | <1ms |
| test_earnings_transcript_citation_format | Earnings call transcript citations format | <1ms |
| test_earnings_quote_attribution | Quotes from earnings calls are properly attributed | <1ms |
| test_macro_report_citation_format | Macro research report citation format | <1ms |
| test_macro_impact_attribution | Macro impact claims are properly sourced | <1ms |
| test_citation_matches_source_doc | Citation references match actual source documents | <1ms |
| test_transcript_citation_matches_audio | Transcript citations align with audio sources | <1ms |
| test_ba_output_has_citations | Business analyst outputs contain required citations | <1ms |
| test_sr_output_broker_ratings_cited | Stock research outputs cite broker ratings | <1ms |
| test_mixed_citation_sources | Handles outputs citing multiple source types | <1ms |
| test_citation_chain_claim_to_doc | Traceability from claims to source documents | <1ms |
| test_citations_in_neo4j | Citations correctly reference Neo4j data | <1ms |
| test_citations_in_postgres | Citations correctly reference PostgreSQL data | <1ms |
| test_citation_relevance_scoring | Citation relevance scoring is accurate | <1ms |
| test_source_type_registry | Source type registry works correctly | <1ms |
| test_web_citations_use_human_labels_and_filter_low_quality_domains | Web citations use human-readable labels and filter spam | <1ms |

#### 3. Hallucination Guard (test_hallucination_guard.py) - 11 tests
Ensures numeric outputs are computed (not hallucinated) and qualitative claims always have citations.

| Test | Meaning | Duration |
|------|---------|----------|
| test_dcf_numbers_are_python_computed | DCF (Discounted Cash Flow) values match direct Python calculation | <1ms |
| test_wacc_computed_from_risk_free_rate | WACC (Weighted Average Cost of Capital) computed correctly | <1ms |
| test_all_claims_have_citations | Every factual claim has a source citation | <1ms |
| test_claims_extracted_from_citations | Claims can be traced back to source citations | <1ms |
| test_financial_data_from_db | Financial data comes from verified database sources | <1ms |
| test_valuation_multiples_computed | Valuation multiples (PE, EV/EBITDA) are computed, not hallucinated | <1ms |
| test_summarizer_cites_sources | Summarizer outputs include source citations | <1ms |
| test_hallucinated_numbers_detected | System can detect fabricated numbers | <1ms |
| test_inputs_from_verified_sources | All inputs come from verified sources | <1ms |
| test_citation_chain_validation | Citation chains are valid and traceable | <1ms |
| test_graph_output_has_citations | Graph/output nodes include citations | <1ms |

#### 4. Planner Prompts (test_planner_prompts.py) - 21 tests
Tests the Planner agent's ability to route queries to appropriate agents.

| Test | Meaning | Duration |
|------|---------|----------|
| test_planner_output_structure | Planner returns correct output structure | <1ms |
| test_planner_complexity_mapping | Maps query complexity to appropriate complexity levels | <1ms |
| test_planner_agent_selection | Routes queries to correct agents (BA, FM, SR, etc.) | <1ms |
| test_unsupported_ticker | Handles unsupported ticker symbols gracefully | <1ms |
| test_ambiguous_query | Handles ambiguous queries appropriately | <1ms |
| test_non_english_input | Handles non-English input queries | <1ms |
| test_long_query_complexity | Correctly assesses complexity of long queries | <1ms |
| test_planner_state_integration | Planner integrates with orchestration state | <1ms |
| test_planner_fallback_to_ba | Falls back to Business Analyst when needed | <1ms |
| test_planner_model_config | Uses correct model configuration | <1ms |
| test_planner_multi_ticker | Handles multi-ticker queries correctly | <1ms |

#### 5. Summarizer Prompts (test_summarizer_prompts.py) - 15 tests
Tests the multi-stage summarization pipeline.

| Test | Meaning | Duration |
|------|---------|----------|
| test_summarizer_stage1_structure | Stage 1 output has correct structure | <1ms |
| test_summarizer_stage2_structure | Stage 2 output has correct structure | <1ms |
| test_summarizer_stage3_citations | Stage 3 adds citations correctly | <1ms |
| test_summarizer_stage4_translation | Stage 4 handles translation if needed | <1ms |
| test_summarizer_full_pipeline | Full summarization pipeline works end-to-end | <1ms |
| test_cites_chunk_ids | Summaries cite chunk IDs properly | <1ms |
| test_multi_ticker_summarization | Handles multi-ticker summarization | <1ms |
| test_summarizer_model_config | Uses correct model for summarization | <1ms |
| test_translation_model_config | Uses correct model for translation | <1ms |
| test_empty_outputs_handling | Handles empty agent outputs | <1ms |
| test_partial_outputs_handling | Handles partial agent outputs | <1ms |
| test_summarizer_state_integration | Integrates with orchestration state | <1ms |

#### 6. Other Prompt Tests
| File | Tests | Meaning |
|------|-------|---------|
| test_fm_prompts.py | 1 | Financial Modeling prompt output schema |
| test_macro_prompts.py | 1 | Macro analysis prompt output schema |
| test_insider_news_prompts.py | 1 | Insider news prompt output schema |
| test_qf_prompts.py | 1 | Quant Fundamental prompt output schema |
| test_sr_prompts.py | 1 | Stock Research prompt output schema |
| test_ws_prompts.py | 1 | Web Search prompt output schema |
| test_orchestration_prompt_telemetry.py | 1 | Prompt telemetry tracking |

---

## Integration Tests (62 tests - 361.88s)

Integration tests verify that agents can successfully retrieve data from actual PostgreSQL and Neo4j databases. These tests take longer as they connect to real services.

### Test Categories

#### 1. Agent-Database Integration (test_agent_db.py) - 14 tests
Verifies each agent can successfully retrieve data from PostgreSQL and Neo4j.

| Test | Meaning | Duration |
|------|---------|----------|
| test_ba_neo4j_retrieval | Business Analyst retrieves chunks from Neo4j for AAPL | 10.05s |
| test_ba_postgres_sentiment | BA retrieves sentiment data from PostgreSQL | 40ms |
| test_qf_postgres_factors | Quant Fundamental reads financial factors from PostgreSQL | 21ms |
| test_qf_piotroski_inputs | QF retrieves Piotroski F-Score inputs | 11ms |
| test_fm_dcf_inputs | Financial Modeling fetches DCF inputs from PostgreSQL | 36ms |
| test_fm_neo4j_peers | FM retrieves peer comparison data from Neo4j | 18ms |
| test_sr_chunk_search | Stock Research returns PDF chunks from Neo4j | 24.38s |
| test_sr_broker_data | SR retrieves broker research data | 321ms |
| test_macro_pg_neo4j_data | Macro agent loads macro + earnings chunks | 91.24s |
| test_insider_news_postgres_data | Insider News loads insider transaction rows | 197.61s |
| test_ws_perplexity_response | Web Search gets results from Perplexity API | 8.15s |
| test_ws_fallback | Web Search fallback when API unavailable | 1ms |
| test_ba_full_pipeline | BA complete data retrieval pipeline | 5.31s |
| test_fm_full_pipeline | FM complete data retrieval pipeline | 22ms |

#### 2. Graph Nodes (test_graph_nodes.py) - 16 tests
Verifies LangGraph topology and routing logic.

| Test | Meaning | Duration |
|------|---------|----------|
| test_planner_output_for_valuation_query | Planner fan-out for valuation queries | <1ms |
| test_planner_output_for_moat_query | Planner fan-out for moat analysis queries | <1ms |
| test_planner_output_for_news_query | Planner fan-out for news queries | <1ms |
| test_planner_routing_targets | Planner routes to correct target agents | <1ms |
| test_planner_fallback_to_ba | Planner fallback logic to BA | <1ms |
| test_react_retry_routes_back_on_empty_output | ReAct retry when no output | <1ms |
| test_react_retry_at_max_iterations | ReAct retry limit behavior | <1ms |
| test_react_no_retry_when_output_exists | No retry when output exists | <1ms |
| test_react_all_agents | ReAct handles all agent types | <1ms |
| test_react_skips_disabled_agents | ReAct skips disabled agents | <1ms |
| test_summarizer_receives_all_outputs | Fan-in: summarizer gets all agent outputs | <1ms |
| test_summarizer_state_structure | Summarizer state structure is correct | <1ms |
| test_build_graph_compiles | LangGraph compiles without errors | 44ms |
| test_graph_has_required_nodes | Graph contains all required nodes | 47ms |
| test_complexity_1_no_retry | Complexity 1 queries don't retry | <1ms |
| test_complexity_2_one_retry | Complexity 2 queries get one retry | <1ms |

#### 3. Infrastructure (test_infra.py) - 7 tests
Verifies database and external service connectivity.

| Test | Meaning | Duration |
|------|---------|----------|
| test_postgres_tables | PostgreSQL has all required tables | 28ms |
| test_postgres_feedback_tables | Feedback tables exist in PostgreSQL | 23ms |
| test_neo4j_connection | Neo4j database is accessible | 27ms |
| test_neo4j_vector_index | Neo4j vector index is configured | 58ms |
| test_ollama_embed | Ollama embedding service works | 332ms |
| test_ollama_models | Ollama has required models | 4ms |
| test_deepseek_api | DeepSeek API configuration check | <1ms |
| test_all_services_health | All services are healthy | 30ms |

#### 4. Multi-Ticker (test_multi_ticker.py) - 14 tests
Tests handling of multi-ticker queries.

| Test | Meaning | Duration |
|------|---------|----------|
| test_multi_ticker_state_has_list | State contains ticker list | <1ms |
| test_multi_ticker_outputs_are_lists | Outputs are lists for multi-ticker | <1ms |
| test_single_ticker_backward_compat | Single ticker queries still work | <1ms |
| test_ba_processes_multiple_tickers | BA handles multiple tickers | <1ms |
| test_fm_processes_multiple_tickers | FM handles multiple tickers | <1ms |
| test_multi_ticker_factors | Multi-ticker factor analysis | <1ms |
| test_summarizer_comparative_output | Summarizer creates comparative output | <1ms |
| test_summarizer_legacy_aliases | Legacy ticker aliases work | <1ms |
| test_comparison_query_structure | Comparison query structure | <1ms |
| test_three_way_comparison | Three-way ticker comparison | <1ms |
| test_empty_ticker_list | Handles empty ticker list | <1ms |
| test_ticker_with_no_data | Handles tickers with no data | <1ms |
| test_multi_ticker_full_flow | Full multi-ticker flow | <1ms |

#### 5. RLAIF Memory (test_rlaif_memory.py) - 11 tests
Tests Reinforcement Learning from AI Feedback memory system.

| Test | Meaning | Duration |
|------|---------|----------|
| test_rlaif_scores_persisted | RLAIF scores saved to database | 256ms |
| test_rlaif_scoring_function | RLAIF scoring function works | 178ms |
| test_episodic_hints_loaded | Episodic memory hints loaded | 3.91s |
| test_episodic_memory_query | Episodic memory queries work | 7.81s |
| test_planner_queries_episodic_memory | Planner queries episodic memory | <1ms |
| test_episodic_hints_influence_routing | Hints influence routing decisions | <1ms |
| test_feedback_tables_created | Feedback tables created | 23ms |
| test_user_feedback_persistence | User feedback persists | 37ms |
| test_post_processing_state_structure | Post-processing state correct | <1ms |
| test_track_multiple_agent_failures | Tracks multiple agent failures | 11.62s |
| test_prompt_version_tracking | Tracks prompt versions | 21ms |

---

## Infrastructure Metrics

### Database Connection Times
| Service | Connection Time | Latency |
|---------|-----------------|---------|
| PostgreSQL (tables) | 16.02ms | 16.02ms |
| PostgreSQL (feedback) | 7.79ms | 7.79ms |
| Neo4j (connection) | 0.09ms | 24.39ms |
| Neo4j (vector index) | 0.08ms | 45.61ms |
| Ollama (embeddings) | N/A | 330.50ms |
| Ollama (models) | N/A | 3.17ms |

### Data Amounts Retrieved
| Source | Data Type | Amount |
|--------|-----------|--------|
| PostgreSQL Tables | list[str] | 6 |
| PostgreSQL Feedback | list[str] | 3 |
| Neo4j Query | int | 1 row |
| Neo4j Vector Indexes | list[str] | 1 |
| Ollama Embeddings | embedding_matrix | 768 dims |
| Ollama Models | list[str] | 2 |
| DeepSeek API | bool | configured/not |

---

## Summary

- **Total Tests:** 141
- **Passed:** 141
- **Failed:** 0
- **Passing Rate:** 100%
- **Total Duration:** ~6 minutes

All tests passed successfully. The test suite covers:
1. **Prompt correctness** - AI prompts produce expected outputs
2. **Citation accuracy** - Sources are properly cited
3. **Hallucination prevention** - Numbers are computed, not hallucinated
4. **Agent routing** - Queries routed to correct agents
5. **Database integration** - Agents can retrieve real data
6. **Graph topology** - LangGraph wiring is correct
7. **Infrastructure** - All services are accessible
8. **Multi-ticker support** - Multi-ticker queries work
9. **RLAIF memory** - Feedback system functions correctly
