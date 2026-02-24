# Web Search Agent — Examples

This agent is a **pure information gatherer**. It finds recent facts and fills gaps
for other agents. It does NOT judge news as good or bad. No BULLISH/BEARISH labels.

---

## Source Tier Quick Reference

| Tier | Sources | Use when |
|---|---|---|
| 1 | SEC/EDGAR, FTC, DOJ, BIS, EU Commission | Filings, investigations, policy |
| 2 | Reuters, Bloomberg, FT, WSJ, CNBC | Earnings, M&A, leadership, macro |
| 3 | Company IR pages, official press releases | Guidance, product launches |
| ❌ Excluded | Blogs, Reddit, StockTwits, KuCoin, Seeking Alpha | Never cite |

---

## Example 1: Earnings Query

**Input**
```json
{"query": "NVDA Q4 2026 earnings results", "ticker": "NVDA", "recency_filter": "week"}
```

**Search angles to cover**
1. `NVIDIA Q4 FY2026 earnings revenue EPS February 2026`
2. `NVIDIA Q1 2027 guidance outlook February 2026`
3. `NVIDIA datacenter segment growth Q4 2026`

**Expected output**
```json
{
  "breaking_news": [
    {
      "title": "NVIDIA Q4 Revenue $39.3B, Datacenter Up 93% YoY",
      "url": "https://ir.nvidia.com/...",
      "published_date": "2026-02-26",
      "source_tier": 1,
      "relevance_score": 0.98,
      "verified": true
    }
  ],
  "sentiment_signal": "NEUTRAL",
  "missing_context": [
    {
      "gap": "Q1 2027 guidance not yet available — earnings call transcript pending",
      "source_url": null,
      "severity": "HIGH"
    }
  ]
}
```

**Notes**
- ✅ Use: NVIDIA IR page, SEC 8-K on EDGAR, Reuters earnings wrap
- ❌ Reject: Seeking Alpha summaries, StockTwits recaps

---

## Example 2: Regulatory Risk Query

**Input**
```json
{"query": "AAPL antitrust EU investigation 2026", "ticker": "AAPL", "recency_filter": "month"}
```

**Search angles to cover**
1. `Apple EU DMA antitrust fine 2026`
2. `Apple DOJ investigation App Store 2026`
3. `Big Tech antitrust ruling 2026 precedent`

**Expected output**
```json
{
  "breaking_news": [
    {
      "title": "EU Opens Formal DMA Non-Compliance Probe into Apple App Store",
      "url": "https://ec.europa.eu/...",
      "published_date": "2026-01-20",
      "source_tier": 1,
      "relevance_score": 0.95,
      "verified": true
    }
  ],
  "sentiment_signal": "NEUTRAL",
  "missing_context": [
    {
      "gap": "Potential fine amount not yet disclosed by EU Commission",
      "source_url": "https://ec.europa.eu/...",
      "severity": "HIGH"
    }
  ]
}
```

**Notes**
- ✅ Best source: ec.europa.eu official press release (Tier 1)
- ✅ Use FT/Reuters for reaction and timeline context
- ⚠️ Law firm blog posts: set `verified: false`, `source_tier: 3`

---

## Example 3: Leadership Change

**Input**
```json
{"query": "Boeing CEO change 2026", "ticker": "BA", "recency_filter": "month"}
```

**Search angles to cover**
1. `Boeing CEO appointment resignation 2026`
2. `Boeing SEC 8-K executive change 2026`
3. `Boeing board of directors statement 2026`

**Expected output**
```json
{
  "breaking_news": [
    {
      "title": "Boeing Names Kelly Ortberg as Permanent CEO",
      "url": "https://ir.boeing.com/...",
      "published_date": "2026-01-15",
      "source_tier": 3,
      "relevance_score": 0.91,
      "verified": true
    }
  ],
  "sentiment_signal": "NEUTRAL",
  "missing_context": [
    {
      "gap": "New CEO's production targets and turnaround plan not yet published",
      "source_url": null,
      "severity": "MEDIUM"
    }
  ]
}
```

**Notes**
- ✅ Best source: SEC Form 8-K (executive changes are mandatory disclosures — Tier 1)
- ✅ Boeing IR page for official announcement
- ❌ Reject: LinkedIn profiles, generic news aggregators

---

## Example 4: M&A Announcement

**Input**
```json
{"query": "MSFT acquisition deal 2026", "ticker": "MSFT", "recency_filter": "month"}
```

**Search angles to cover**
1. `Microsoft acquisition announcement February 2026`
2. `Microsoft acquisition FTC antitrust review 2026`
3. `Microsoft Azure AI expansion deal 2026`

**Expected output**
```json
{
  "breaking_news": [
    {
      "title": "Microsoft Acquires AI Infrastructure Startup for $4.2B",
      "url": "https://bloomberg.com/...",
      "published_date": "2026-02-10",
      "source_tier": 2,
      "relevance_score": 0.96,
      "verified": true
    }
  ],
  "sentiment_signal": "NEUTRAL",
  "missing_context": [
    {
      "gap": "FTC review timeline and deal close date not yet confirmed",
      "source_url": "https://ftc.gov/...",
      "severity": "MEDIUM"
    }
  ]
}
```

**Notes**
- ✅ Best source: SEC 8-K (material deals must be disclosed within 4 business days)
- ✅ Bloomberg/Reuters deal desks for M&A reporting
- ❌ Reject: Rumour sources without deal value or named acquiree

---

## Example 5: Macro / Sector Shock

**Input**
```json
{"query": "US semiconductor export controls 2026", "ticker": null, "recency_filter": "week"}
```

**Search angles to cover**
1. `US BIS semiconductor export control rule February 2026`
2. `NVDA AMD Intel China export restriction 2026`
3. `US China tech trade chips policy 2026`

**Expected output**
```json
{
  "ticker": null,
  "breaking_news": [
    {
      "title": "BIS Expands Entity List: 12 Additional Chinese AI Firms Restricted",
      "url": "https://bis.doc.gov/...",
      "published_date": "2026-02-18",
      "source_tier": 1,
      "relevance_score": 0.97,
      "verified": true
    }
  ],
  "sentiment_signal": "NEUTRAL",
  "missing_context": [
    {
      "gap": "Revenue impact per company not yet quantified — no official company guidance on new restrictions",
      "source_url": null,
      "severity": "HIGH"
    }
  ],
  "competitor_signals": [
    {
      "company": "TSMC",
      "signal": "TSMC confirmed 5% revenue exposure to newly restricted customers in Feb 2026 earnings call",
      "source_url": "https://reuters.com/..."
    }
  ]
}
```

---

## Common Mistakes to Avoid

| ❌ Wrong | ✅ Correct |
|---|---|
| `"sentiment_signal": "BEARISH"` | `"sentiment_signal": "NEUTRAL"` always |
| Citing KuCoin, StockTwits, ainvest.com | Use Reuters, Bloomberg, SEC only |
| `"source_tier": 2` for a blog | Check: is it actually Reuters/Bloomberg/FT/WSJ/CNBC? |
| Inventing a revenue figure | Write `INSUFFICIENT_DATA` instead |
| High confidence from 1 source | Cap `confidence` at 0.6 for single-source findings |
| `published_date` left as `"unknown"` | Try harder to find the date; if truly unknown, flag it |

---

*Last updated: 2026-02-24 | Author: hck717*
