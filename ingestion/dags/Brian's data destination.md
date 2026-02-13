Agent	Data Type	FMP	EODHD	Primary	Destination	Reason
Business Analyst (CRAG)	SEC Filings - 10-K	✓	✗	FMP	Qdrant + Neo4j	Embeddings for semantic search + proposition graph
Business Analyst (CRAG)	SEC Filings - 10-Q	✓	✗	FMP	Qdrant + Neo4j	Embeddings for semantic search + proposition graph
Business Analyst (CRAG)	SEC Filings - 8-K	✓	✗	FMP	Qdrant + Neo4j	Embeddings for semantic search + proposition graph
Business Analyst (CRAG)	Earnings Call Transcripts	✓	✗	FMP	Qdrant + Neo4j	Embeddings for Q&A + proposition graph
Business Analyst (CRAG)	Company Profiles	✓	✓	Both	Neo4j	Entity nodes with company attributes
Business Analyst (CRAG)	Risk Factors	✓	✗	FMP	Neo4j	Risk nodes linked to company/strategy nodes
Business Analyst (CRAG)	Business Strategy Narratives	✓	✗	FMP	Neo4j	Strategy nodes with relationships to technology/risks
Business Analyst (CRAG)	Management Discussion & Analysis	✓	✗	FMP	Qdrant + Neo4j	Embeddings for CRAG + narrative graph
Quantitative (Dual-Path)	Financial Statements	✓	✓	Both	PostgreSQL	Structured time-series for dual-path verification
Quantitative (Dual-Path)	Financial Ratios	✓	✓	Both	PostgreSQL	Calculated metrics for verification
Quantitative (Dual-Path)	Key Metrics (TTM)	✓	✓	Both	PostgreSQL	Real-time metrics for anomaly detection
Quantitative (Dual-Path)	Historical Financials (30+ years)	✓	✓	Both	PostgreSQL	Time-series analysis, trend detection
Quantitative (Dual-Path)	Growth Metrics	✓	✗	FMP	PostgreSQL	Growth rate calculations
Quantitative (Dual-Path)	Enterprise Values	✓	✗	FMP	PostgreSQL	Valuation calculations
Quantitative (Dual-Path)	Market Cap (Historical)	✓	✓	Both	PostgreSQL	Time-series for market cap trends
Quantitative (Dual-Path)	Share Float Data	✓	✗	FMP	PostgreSQL	Float analysis for liquidity metrics
Quantitative (Dual-Path)	Financial Scores	✓	✗	FMP	PostgreSQL	Piotroski F-Score, Beneish M-Score tracking
Quantitative (Dual-Path)	Historical Stock Prices	✓	✓	Both	PostgreSQL	Dual-path price verification, EOD + intraday
Quantitative (Dual-Path)	Beta & Volatility	✓	✓	Both	PostgreSQL	Risk metrics calculation
Financial Modeling (DCF)	Financial Statements (Q & A)	✓	✓	Both	PostgreSQL	Model inputs, DCF calculations
Financial Modeling (DCF)	As-Reported Financials	✓	✗	FMP	PostgreSQL	GAAP/IFRS compliance checking
Financial Modeling (DCF)	DCF Valuation Models	✓	✗	FMP	PostgreSQL	Valuation model inputs and outputs
Financial Modeling (DCF)	Revenue Segmentation	✓	✗	FMP	PostgreSQL	Segment analysis, growth projections
Financial Modeling (DCF)	Analyst Estimates	✓	✓	Both	PostgreSQL	Consensus vs actual tracking
Financial Modeling (DCF)	Historical Stock Prices	✓	✓	Both	PostgreSQL	Returns calculation, valuation
Financial Modeling (DCF)	Dividend History	✓	✓	Both	PostgreSQL	Dividend discount model, yield analysis
Financial Modeling (DCF)	Treasury Rates	✓	✗	FMP	PostgreSQL	Risk-free rate for WACC/DCF
Financial Modeling (DCF)	Market Risk Premium	✓	✗	FMP	PostgreSQL	Cost of equity calculation
Financial Modeling (DCF)	Economic Indicators	✓	✓	Both	PostgreSQL	Macro factors for scenario analysis
Financial Modeling (DCF)	Peer Company Data	✓	✓	Both	PostgreSQL	Comparable company analysis
Financial Modeling (DCF)	Industry Benchmarks	✓	✗	FMP	PostgreSQL	Sector comparison metrics
