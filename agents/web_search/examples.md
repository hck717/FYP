# 📋 Web Search Agent — Query Examples & Source Quality Guide

This file provides concrete examples of how the Web Search Agent should behave across different
query types. It is read by the Supervisor and used to calibrate query construction and output
quality expectations.

---

## 📌 Source Tier Reference

The agent enforces a strict source hierarchy. The Supervisor/Synthesizer should weight
citations accordingly:

| Tier | Sources | Trust Level | When to use |
|---|---|---|---|
| **1 — Regulatory** | sec.gov, EDGAR, exchange filings (HKEX, LSE), CFTC, FTC, DOJ press releases | Highest | Insider trades, 10-K/10-Q, official investigations |
| **2 — Financial Wire** | Reuters, Bloomberg, FT, WSJ, CNBC, Barron's | High | Earnings, M&A, macro events, executive changes |
| **3 — Corporate IR** | Official company IR pages, verified press releases | High | Product launches, guidance updates, dividends |
| **4 — Financial Data** | S&P Global, Moody's, FactSet, Refinitiv | Medium-High | Analyst upgrades, ratings changes |
| **❌ Excluded** | Generic blogs, Reddit, X/Twitter, unverified aggregators | None | Do not cite |

> **Note on ainvest.com, Seeking Alpha, Motley Fool:** These are Tier 4 at best. Flag any claims
> from these as `"verified": false` and `"source_tier": 3`. The Critic Agent will filter them.

---

## 🔍 Example 1: Earnings Surprise Query

### Input
```json
{
  "query": "NVDA Q4 2026 earnings results",
  "ticker": "NVDA",
  "recency_filter": "week"
}
```

### Ideal Search Angles (Step-Back reasoning)
1. **Direct:** "NVIDIA Q4 FY2026 earnings revenue EPS February 2026"
2. **Reaction:** "NVIDIA earnings beat miss analyst reaction February 2026"
3. **Forward guidance:** "NVIDIA Q1 2027 guidance outlook datacenter 2026"

### Expected Output Pattern
```json
{
  "breaking_news": [
    {
      "title": "NVIDIA Q4 Revenue Beats at $39.3B, Datacenter Up 93% YoY",
      "url": "https://www.reuters.com/...",
      "published_date": "2026-02-26",
      "source_tier": 2,
      "relevance_score": 0.98,
      "verified": true
    }
  ],
  "sentiment_signal": "BULLISH",
  "supervisor_escalation": {
    "action": "CONFIRMATORY_SIGNAL",
    "rationale": "Earnings beat aligns with Quant agent bullish fundamental view.",
    "conflict_with_agent": null
  }
}
```

### Source Quality Notes
- ✅ Use: Reuters/Bloomberg earnings wrap, NVIDIA IR press release (ir.nvidia.com)
- ✅ Use: SEC 8-K filing on EDGAR (earnings release is filed as 8-K)
- ⚠️ Flag: Seeking Alpha earnings summaries — `verified: false`
- ❌ Reject: Any source without a specific revenue/EPS figure and date

---

## 🔍 Example 2: Regulatory / Legal Risk Query

### Input
```json
{
  "query": "AAPL antitrust regulatory investigation 2026",
  "ticker": "AAPL",
  "recency_filter": "month"
}
```

### Ideal Search Angles (Step-Back reasoning)
1. **Direct regulatory:** "Apple antitrust DOJ EU DMA investigation 2026"
2. **Peer precedent:** "Google Meta antitrust ruling 2026 Big Tech implications"
3. **Financial impact:** "Apple App Store fine revenue impact 2026"

### Expected Output Pattern
```json
{
  "unknown_risk_flags": [
    {
      "risk": "EU DMA non-compliance fine up to 10% global revenue — not yet reflected in consensus EPS estimates",
      "source_url": "https://ec.europa.eu/...",
      "severity": "HIGH"
    }
  ],
  "competitor_signals": [
    {
      "company": "Alphabet",
      "signal": "Google lost EU antitrust appeal Feb 2026 — sets precedent for Apple App Store case",
      "source_url": "https://ft.com/..."
    }
  ],
  "supervisor_escalation": {
    "action": "CONFLICT_SIGNAL",
    "rationale": "Financial Modelling agent DCF does not include regulatory fine scenario. Synthesizer should add bear case.",
    "conflict_with_agent": "financial_modelling"
  }
}
```

### Source Quality Notes
- ✅ Use: Official EU Commission press releases (ec.europa.eu), DOJ.gov announcements
- ✅ Use: FT and Reuters for legal/regulatory reporting
- ⚠️ Flag: Law firm blog posts — cite as `source_tier: 3`, `verified: false`
- ❌ Reject: Legal opinion pieces without case citation numbers

---

## 🔍 Example 3: CEO / Leadership Change Query

### Input
```json
{
  "query": "Boeing CEO change leadership update 2026",
  "ticker": "BA",
  "recency_filter": "month"
}
```

### Ideal Search Angles (Step-Back reasoning)
1. **Direct:** "Boeing CEO resignation appointment 2026"
2. **Board reaction:** "Boeing board of directors statement leadership transition 2026"
3. **Market reaction:** "Boeing stock analyst reaction CEO change 2026"

### Expected Output Pattern
```json
{
  "breaking_news": [
    {
      "title": "Boeing Names Kelly Ortberg as Permanent CEO After Turnaround Progress",
      "url": "https://wsj.com/...",
      "published_date": "2026-01-15",
      "source_tier": 2,
      "relevance_score": 0.91,
      "verified": true
    }
  ],
  "sentiment_signal": "MIXED",
  "sentiment_rationale": "New permanent CEO appointment seen as stabilising, but WSJ (2026-01-15) notes production targets remain unmet."
}
```

### Source Quality Notes
- ✅ Use: SEC Form 8-K (executive changes are mandatory disclosures)
- ✅ Use: WSJ / Reuters / Bloomberg for corporate governance reporting
- ✅ Use: Boeing official IR page (ir.boeing.com) for press releases
- ⚠️ Flag: Executive profiles from LinkedIn or generic news aggregators

---

## 🔍 Example 4: M&A / Deal Query

### Input
```json
{
  "query": "MSFT acquisition deal announcement 2026",
  "ticker": "MSFT",
  "recency_filter": "month"
}
```

### Ideal Search Angles (Step-Back reasoning)
1. **Direct deal:** "Microsoft acquisition merger announcement 2026"
2. **Regulatory hurdle:** "Microsoft acquisition FTC antitrust review 2026"
3. **Strategic context:** "Microsoft AI cloud strategy expansion deal 2026"

### Expected Output Pattern
```json
{
  "breaking_news": [
    {
      "title": "Microsoft Acquires AI Startup for $4.2B to Bolster Azure Reasoning Layer",
      "url": "https://bloomberg.com/...",
      "published_date": "2026-02-10",
      "source_tier": 2,
      "relevance_score": 0.96,
      "verified": true
    }
  ],
  "unknown_risk_flags": [
    {
      "risk": "FTC may challenge deal under existing Big Tech scrutiny framework — timeline risk to close",
      "source_url": "https://ftc.gov/...",
      "severity": "MEDIUM"
    }
  ],
  "supervisor_escalation": {
    "action": "CONFLICT_SIGNAL",
    "rationale": "Acquisition adds EPS dilution not modelled by Financial Modelling Agent. Synthesizer should flag valuation impact.",
    "conflict_with_agent": "financial_modelling"
  }
}
```

### Source Quality Notes
- ✅ Use: SEC Form 8-K (material deals must be disclosed within 4 business days)
- ✅ Use: Bloomberg / Reuters deal reporters for M&A coverage
- ✅ Use: FTC/DOJ press releases for regulatory review status
- ❌ Reject: Rumour-based sources without deal value or named parties

---

## 🔍 Example 5: Macro / Sector Shock (No Specific Ticker)

### Input
```json
{
  "query": "US semiconductor export controls impact 2026",
  "ticker": null,
  "recency_filter": "week"
}
```

### Ideal Search Angles (Step-Back reasoning)
1. **Policy:** "US BIS semiconductor export control rule update February 2026"
2. **Sector impact:** "NVDA AMD Intel China export restriction revenue impact 2026"
3. **Geopolitical:** "US China tech trade war chips 2026 latest"

### Expected Output Pattern
```json
{
  "ticker": null,
  "sentiment_signal": "BEARISH",
  "unknown_risk_flags": [
    {
      "risk": "New BIS rule expands Entity List to include 12 additional Chinese AI firms — impacts NVDA, AMD H20 chip sales",
      "source_url": "https://bis.doc.gov/...",
      "severity": "HIGH"
    }
  ],
  "competitor_signals": [
    {
      "company": "TSMC",
      "signal": "TSMC confirms 5% revenue exposure to newly restricted customers; guidance unchanged",
      "source_url": "https://reuters.com/..."
    }
  ],
  "supervisor_escalation": {
    "action": "CONFLICT_SIGNAL",
    "rationale": "Macro agent does not model BIS rule changes. Synthesizer must apply sector-wide risk discount.",
    "conflict_with_agent": "macro_economic"
  }
}
```

### Source Quality Notes
- ✅ Use: BIS.doc.gov (Bureau of Industry and Security) — Tier 1 for export control
- ✅ Use: Reuters / FT for policy reaction and company impact
- ✅ Use: Company 8-K filings if companies disclose material impact
- ❌ Reject: Political opinion pieces without specific rule citations

---

## 🔍 Example 6: Contrarian Signal Detection

### Input
```json
{
  "query": "TSLA insider buying sentiment despite negative news 2026",
  "ticker": "TSLA",
  "recency_filter": "week"
}
```

### Ideal Search Angles (Step-Back reasoning)
1. **Insider activity:** "Tesla insider SEC Form 4 purchase 2026"
2. **News sentiment:** "Tesla negative news controversy February 2026"
3. **Analyst divergence:** "Tesla analyst upgrade downgrade February 2026"

### Expected Output Pattern — CONFLICT scenario
```json
{
  "sentiment_signal": "MIXED",
  "sentiment_rationale": "Reuters (2026-02-20) reports Musk political controversy causing brand damage; simultaneously, SEC Form 4 shows CFO purchased $3M shares on 2026-02-18.",
  "unknown_risk_flags": [
    {
      "risk": "Brand damage from political association not yet quantified in consensus revenue estimates",
      "source_url": "https://reuters.com/...",
      "severity": "MEDIUM"
    }
  ],
  "supervisor_escalation": {
    "action": "CONFLICT_SIGNAL",
    "rationale": "Insider & Sentiment Agent bullish on CFO buying; this agent flags unquantified brand risk. Classic contrarian setup — Synthesizer must arbitrate with conviction level.",
    "conflict_with_agent": "insider_sentiment"
  }
}
```

### Source Quality Notes
- ✅ Use: SEC EDGAR Form 4 filings (insider trades — Tier 1)
- ✅ Use: Reuters / Bloomberg for brand/controversy reporting
- ⚠️ Flag: Social media sentiment data — `verified: false`, `source_tier: 3`
- ❌ Reject: Any source that cannot separate fact from opinion on insider intent

---

## ⚠️ Common Failure Modes to Avoid

| Failure | Description | Correct Behaviour |
|---|---|---|
| **Stale news** | Citing articles > 30 days old as breaking | Label `[HISTORICAL]`, lower `relevance_score` |
| **Tier 4 as Tier 2** | Citing ainvest.com or Seeking Alpha as Bloomberg | Set `source_tier: 3`, `verified: false` |
| **Single source confidence** | High `confidence` score from only one source | Flag `⚠️ UNCONFIRMED — SINGLE SOURCE`, cap confidence at 0.6 |
| **Inferred figures** | Stating revenue growth % not in any source | Forbidden — output `INSUFFICIENT_DATA` instead |
| **Empty escalation** | Setting `action: STANDALONE` when a conflict exists | Always cross-check against other agent domains before choosing STANDALONE |
| **Generic query** | Searching "NVDA news" without year/quarter | Always include year and quarter for recency precision |

---

## 🏆 Ideal Output Checklist (Supervisor validation)

Before accepting Web Search Agent output, Supervisor should verify:

- [ ] `error` is `null`
- [ ] `fallback_triggered` is `false`
- [ ] `confidence` ≥ 0.5
- [ ] At least 1 item in `breaking_news` OR explicit `INSUFFICIENT_DATA` rationale
- [ ] All `breaking_news` items have `url` and `published_date` (not `"unknown"`)
- [ ] `sentiment_rationale` contains a URL and date
- [ ] `supervisor_escalation.action` is set (not null)
- [ ] `unknown_risk_flags` checked against Quant + Financial Modelling assumptions

---

*Last updated: 2026-02-24 | Author: hck717*
