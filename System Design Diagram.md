## System Design Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  HTML Content  │  URLs  │  Training Data  │  User Context           │
└────────┬────────────────┬─────────────────┬─────────────────────────┘
         │                │                 │
         ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION MODULE                           │
├─────────────────────────────────────────────────────────────────────┤
│  DataCollector                                                      │
│  ├─ fetch_phishing_dataset() → UCI ARFF Parser                      │
│  ├─ fetch_malware_urls() → GitHub CSV Parser                        │
│  ├─ fetch_web_traffic_data() → Temporal Data Generator              │
│  └─ fetch_security_logs() → Event Log Generator                     │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   CORE PROCESSING LAYER                             │
├──────────────────────┬──────────────────────┬───────────────────────┤
│  CONTENT             │  SECURITY            │  AGENTIC              │
│  UNDERSTANDING       │  FRAMEWORK           │  CAPABILITIES         │
├──────────────────────┼──────────────────────┼───────────────────────┤
│ GeminiChrome         │ ChromeSecurity       │ AgenticCapabilities   │
│ Integration          │ Framework            │ Enhancer              │
│                      │                      │                       │
│ ┌──────────────┐     │ ┌──────────────┐     │ ┌──────────────┐      │
│ │BeautifulSoup │     │ │RandomForest  │     │ │Priority      │      │
│ │HTML Parser   │     │ │Classifier    │     │ │Sorter        │      │
│ └──────┬───────┘     │ │n_est=150     │     │ └──────┬───────┘      │
│        │             │ └──────┬───────┘     │        │              │
│ ┌──────▼───────┐     │ ┌──────▼───────┐     │ ┌──────▼───────┐      │
│ │Sentence      │     │ │Gradient      │     │ │Action        │      │
│ │Transformer   │     │ │Boosting      │     │ │Batcher       │      │
│ │Embedder      │     │ │Classifier    │     │ │              │      │
│ │dim=384       │     │ │n_est=150     │     │ │batch_size=3  │      │
│ └──────┬───────┘     │ └──────┬───────┘     │ └──────┬───────┘      │
│        │             │        │             │        │              │
│ ┌──────▼───────┐     │ ┌──────▼───────┐     │ ┌──────▼───────┐      │
│ │Feature       │     │ │Risk Score    │     │ │Intent        │      │
│ │Extractor     │     │ │Calculator    │     │ │Predictor     │      │
│ │15+ features  │     │ │4 risk levels │     │ │5 categories  │      │
│ └──────────────┘     │ └──────────────┘     │ └──────────────┘      │
└──────────┬───────────┴──────────┬───────────┴──────────┬────────────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  EnhancedModelTrainer                                               │
│  ├─ train_with_validation()                                         │
│  │  ├─ Train-Test Split (80/20)                                     │
│  │  ├─ Warm-Start Iteration (20 epochs)                             │
│  │  ├─ Accuracy/Loss Tracking                                       │
│  │  └─ Prediction Storage                                           │
│  ├─ Cross-Validation (5-fold)                                       │
│  └─ Feature Importance Extraction                                   │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  EVALUATION & TESTING LAYER                         │
├──────────────────────┬──────────────────────┬───────────────────────┤
│  AUTOMATED TESTING   │  PERFORMANCE         │  TROUBLESHOOTING      │
├──────────────────────┼──────────────────────┼───────────────────────┤
│ Testing Framework    │ Benchmarking System  │ Troubleshooting       │
│                      │                      │ System                │
│ ┌──────────────┐     │ ┌──────────────┐     │ ┌──────────────┐      │
│ │Unit Tests    │     │ │Latency       │     │ │Health        │      │
│ │5 components  │     │ │Measurement   │     │ │Diagnostics   │      │
│ └──────┬───────┘     │ │100+ iters    │     │ └──────┬───────┘      │
│        │             │ └──────┬───────┘     │        │              │
│ ┌──────▼───────┐     │ ┌──────▼───────┐     │ ┌──────▼───────┐      │
│ │Integration   │     │ │Throughput    │     │ │Bottleneck    │      │
│ │Tests         │     │ │Calculation   │     │ │Detection     │      │
│ │3 workflows   │     │ │ops/sec       │     │ │4 checks      │      │
│ └──────┬───────┘     │ └──────┬───────┘     │ └──────┬───────┘      │
│        │             │        │             │        │              │
│ ┌──────▼───────┐     │ ┌──────▼───────┐     │ ┌──────▼───────┐      │
│ │Coverage      │     │ │Real-time     │     │ │Error         │      │
│ │Calculator    │     │ │Monitoring    │     │ │Logging       │      │
│ │Per Component │     │ │0.5s interval │     │ │              │      │
│ └──────────────┘     │ └──────────────┘     │ └──────────────┘      │
└──────────┬───────────┴──────────┬───────────┴──────────┬────────────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ANALYSIS & METRICS LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  Metrics Computation                                                │
│  ├─ Confusion Matrix (TP, TN, FP, FN)                               │
│  ├─ ROC Curve (FPR, TPR, AUC)                                       │
│  ├─ Precision-Recall Curve                                          │
│  ├─ Statistical Analysis (mean, std, percentiles)                   │
│  └─ Trade-off Analysis (accuracy vs speed, FPR vs FNR)              │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION LAYER                              │
├──────────────────────┬──────────────────────┬───────────────────────┤
│  MATPLOTLIB          │  SEABORN             │  PLOTLY               │
├──────────────────────┼──────────────────────┼───────────────────────┤
│ ┌──────────────┐     │ ┌──────────────┐     │ ┌──────────────┐      │
│ │Training      │     │ │Confusion     │     │ │Interactive   │      │
│ │History Plots │     │ │Matrix        │     │ │Dashboard     │      │
│ └──────────────┘     │ │Heatmaps      │     │ │6 subplots    │      │
│ ┌──────────────┐     │ └──────────────┘     │ └──────────────┘      │
│ │ROC Curves    │     │ ┌──────────────┐     │ ┌──────────────┐      │
│ │              │     │ │Distribution  │     │ │3D Scatter    │      │
│ └──────────────┘     │ │Plots         │     │ │              │      │
│ ┌──────────────┐     │ └──────────────┘     │ └──────────────┘      │
│ │Feature       │     │                      │                       │
│ │Importance    │     │                      │                       │
│ └──────────────┘     │                      │                       │
└──────────┬───────────┴──────────┬───────────┴──────────┬────────────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  PNG Images (300 DPI)  │  HTML Dashboard  │  Text Reports           │
│  ├─ 6 visualization    │  └─ Plotly       │  ├─ Statistical         │
│  │   dashboards        │     interactive  │  │   analysis           │
│  └─ Saved to disk      │     charts       │  ├─ Trade-off           │
│                        │                  │  │   analysis           │
│                        │                  │  └─ Recommendations     │
└────────────────────────┴──────────────────┴─────────────────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  Workspace APIs                                                     │
│  ├─ Gmail API → send_email, schedule_send, draft_response           │
│  ├─ Calendar API → create_event, check_availability, send_invite    │
│  ├─ Drive API → save_file, share_document, create_folder            │
│  ├─ Sheets API → export_data, create_chart, import_data             │
│  └─ Third-party CRM → sync_contacts, log_interaction, update_lead   │
└─────────────────────────────────────────────────────────────────────┘

DATA FLOW:
Input → Collection → Processing → Training → Evaluation → Analysis → Visualization → Output

COMPONENT INTERACTIONS:
├─ Content Understanding feeds Security Framework (feature vectors)
├─ Security Framework triggers Agentic Capabilities (risk scores)
├─ Agentic Capabilities generates actions for Workspace Integration
├─ All modules report metrics to Benchmarking System
├─ Benchmarking System feeds Visualization Layer
└─ Testing Framework validates all components

STORAGE:
├─ In-memory: Model weights, predictions, metrics
├─ Disk: PNG images, HTML files, text reports
└─ No persistent database
```