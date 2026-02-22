# Chrome Agentic AI System with Gemini Integration

## Project Name
**Chrome Agentic AI System with Gemini Integration**

---

## Overview
ML-based browser automation framework that performs web content analysis, security threat detection, and autonomous action generation. The system integrates transformer models for content understanding, ensemble classifiers for phishing detection, and automated performance benchmarking with visualization outputs.

---

## Goals & Purposes

**Goals:**
- Implement browser content understanding using NLP embeddings
- Detect security threats through trained ML classifiers
- Generate autonomous browser actions based on content analysis
- Benchmark system performance across multiple metrics
- Integrate workspace service APIs for action execution

**Purposes:**
- Automate web content analysis and threat detection
- Provide real-time security risk assessment for URLs and web pages
- Enable autonomous browser actions with priority-based optimization
- Generate performance analytics and model evaluation reports

---

## Technical Tools and Stacks

**Languages:**
Python 3.8+

**ML Frameworks:**
scikit-learn, transformers, torch, sentence-transformers

**Data Processing:**
pandas, numpy

**Visualization:**
matplotlib, seaborn, plotly

**Web Automation:**
selenium, webdriver-manager, beautifulsoup4, lxml

**Testing:**
pytest, pytest-cov, coverage

**NLP Models:**
SentenceTransformer (all-MiniLM-L6-v2)

**ML Algorithms:**
Random Forest Classifier, Gradient Boosting Classifier, TF-IDF Vectorizer

**APIs:**
Google Workspace (Gmail, Calendar, Drive, Sheets)

**Data Sources:**
UCI Machine Learning Repository (Phishing Dataset - ARFF format), GitHub datasets (Malware URLs - CSV)

**Datasets:**
- UCI Phishing Website Dataset (5000 samples, 30+ features)
- Malware URL Dataset (CSV format)
- Web Traffic Analytics (365-day temporal data)
- Security Event Logs (10000 synthetic events)

---

## Features & Functionality

**Content Understanding Module:**
- HTML parsing and DOM structure extraction
- Text extraction with whitespace normalization
- 384-dimensional transformer embeddings generation
- Link analysis (internal/external classification)
- Form detection (inputs, buttons, login forms)
- Script counting for XSS risk assessment
- Meta tag and heading hierarchy extraction

**Security Detection System:**
- URL feature extraction (11 numerical features)
- Risk scoring algorithm with threshold-based classification
- Malicious pattern detection using regex
- SSL/TLS verification
- Privacy compliance checking
- Threat categorization (safe, suspicious, dangerous, critical)

**Agentic Capabilities:**
- Autonomous action generation from content analysis
- Priority-based action sorting (critical, high, medium, low)
- Action batching for sequential execution
- User intent prediction using Dirichlet sampling
- Action quality measurement and tracking

**ML Model Training:**
- Iterative warm-start training with 20 epochs
- Random Forest and Gradient Boosting implementations
- Train-test splitting (80/20)
- 5-fold cross-validation
- Feature importance extraction

**Performance Benchmarking:**
- Processing latency measurement across 100+ iterations
- Throughput calculation (ops/sec)
- Real-time CPU and memory tracking
- Baseline comparison analysis
- Statistical metric computation (mean, median, std, percentiles)

**Automated Testing:**
- Unit tests for 5 core components
- Integration tests for end-to-end workflows
- Test coverage calculation per component
- Performance under load testing (100 sequential requests)

**Visualization Generation:**
- Training history plots (accuracy, loss over iterations)
- Confusion matrices with heatmaps
- ROC curves with AUC scores
- Precision-recall curves
- Feature importance bar charts
- Performance trade-off scatter plots
- Statistical distribution histograms
- Interactive Plotly dashboards

**Workspace Integration:**
- Gmail API mapping (send, schedule, draft)
- Calendar API (events, availability, invites)
- Drive API (file operations, sharing)
- Sheets API (data export, charts)
- Third-party CRM connectors

**Troubleshooting System:**
- Component health diagnostics
- Bottleneck detection (processing time, memory, CPU, response time)
- Error logging and resolution tracking
- System status reporting

---

## Comprehensive Description

This system performs automated browser analysis through three primary pipelines: content understanding, security detection, and performance monitoring.

The content understanding pipeline parses HTML documents using BeautifulSoup, extracts text content, and generates semantic embeddings via a pre-trained SentenceTransformer model. The system analyzes DOM structure by counting links, forms, scripts, and other elements to build a feature vector for downstream processing.

The security detection pipeline trains two ensemble classifiers (Random Forest with 150 estimators, Gradient Boosting with 150 estimators) on the UCI phishing dataset containing 4000 labeled samples. URL features are extracted including length metrics, special character counts, and protocol information. Risk scores are calculated using a weighted scoring function with thresholds defining four risk levels. Content safety checks apply regex pattern matching to detect malicious JavaScript patterns.

The agentic capabilities module generates actions based on content analysis results. A priority-based sorting algorithm orders actions by urgency and estimated execution time. Actions are batched into groups of three for efficient processing. User intent prediction applies Dirichlet distribution sampling across five categories using browsing context features.

Model training uses warm-start incremental learning across 20 iterations, increasing tree count by 5 per iteration. Training history tracks accuracy and loss metrics per iteration for both training and validation sets. Cross-validation performs 5-fold splits to assess generalization. Feature importance values are extracted from trained models for interpretability analysis.

Performance benchmarking executes 100+ iterations measuring processing time, memory usage, and throughput. Statistical analysis computes mean, median, standard deviation, and percentiles. Baseline comparison calculates percentage improvements across metrics. Real-time monitoring tracks CPU, memory, request rate, and response time at 0.5-second intervals over 10 seconds.

Visualization outputs include 6 PNG dashboards at 300 DPI resolution showing training curves, confusion matrices, ROC curves, precision-recall curves, feature importance, and trade-off analyses. An interactive Plotly dashboard exports to HTML with linked plots across 6 subplots. Text reports generate statistical summaries, trade-off analyses, and recommendations in ASCII-formatted documents.

Testing infrastructure runs unit tests on 5 components and integration tests on 3 workflows. Coverage calculation estimates percentage of code exercised per component. Performance tests validate latency thresholds (mean < 100ms, max < 500ms) under simulated load.

Workspace integration maps actions to Google service APIs with availability flags. The system returns service names, action types, and integration status for each requested operation.

---

## Target Audience and Operation Overview

**Target Audience:**
- Browser security teams implementing threat detection
- ML engineers building automated content analysis systems
- DevOps teams requiring performance monitoring frameworks
- Research teams studying phishing detection algorithms
- Chrome extension developers needing content understanding modules

**Operation Overview:**

System accepts HTML content and URLs as input. Content understanding module processes HTML to extract features and generate embeddings. Security module analyzes URLs and content for threats, outputting risk scores and classifications. Agentic module generates prioritized actions based on analysis results.

Training mode accepts labeled datasets in ARFF or CSV format. Models train iteratively with warm-start optimization, storing predictions and metrics per iteration. Evaluation generates confusion matrices, ROC curves, and statistical reports.

Benchmarking mode executes repeated operations measuring latency and throughput. Results compare against baseline metrics, calculating percentage improvements. Real-time monitoring collects system metrics at fixed intervals.

Testing mode runs automated test suites, recording pass/fail status and coverage percentages. Diagnostic mode checks component health and detects performance bottlenecks.

Visualization mode generates PNG images and HTML dashboards from collected metrics. Report mode writes statistical analyses and recommendations to text files.

All operations execute locally without external API calls except for dataset downloads. Models persist in memory during session. Output artifacts save to local filesystem.

---
