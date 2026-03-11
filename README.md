# Fuzzy-Monotonic LightGBM for Explainable Credit Default Prediction

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://drive.google.com/file/d/1e3M3xxotAb7ldzuhL9uBTT34yRYYJqxC/view?usp=sharing)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A hybrid explainable AI framework combining fuzzy linguistic reasoning with monotonic gradient boosting for regulatory-compliant credit default prediction.**

---

## 🎯 Overview

This research project addresses the critical trade-off between **predictive accuracy** and **regulatory interpretability** in financial credit risk modeling. We propose a novel **Fuzzy-Monotonic LightGBM** framework that achieves competitive performance (ROC-AUC ~0.77, PR-AUC 0.55) while maintaining structural transparency through:

- 🧩 **Fuzzy Membership Functions**: Human-interpretable linguistic variables (Low/Medium/High)
- 📊 **Monotonic Constraints**: Economic priors enforced in gradient boosting
- 🔧 **Behavioral Feature Engineering**: Domain-driven credit indicators
- 🔍 **Multi-Layer Explainability**: Structural (fuzzy + monotonic) + attributional (SHAP)

### Key Results

| Metric           | Baseline Raw | Baseline + Engineered | Fuzzy  | **Fuzzy-Monotonic** |
| ---------------- | ------------ | --------------------- | ------ | ------------------- |
| **ROC-AUC**      | 0.7744       | 0.7733                | 0.7701 | **0.7700**          |
| **PR-AUC**       | 0.5477       | 0.5496                | 0.5485 | **0.5498** ↑        |
| **Brier Score**  | 0.1725       | 0.1730                | 0.1687 | **0.1696** ↓        |
| **KS Statistic** | 0.4235       | 0.4231                | 0.4208 | **0.4144**          |

✅ **Best PR-AUC** for minority class detection  
✅ **Improved calibration** for probability estimates  
✅ **Economic consistency** via monotonic constraints  
✅ **Regulatory alignment** (Basel II/III, IFRS-9, ECB TRIM)

---

## 📊 Datasets

| Dataset                        | Source            | Size           | Role                           | Key Features                                                                            |
| ------------------------------ | ----------------- | -------------- | ------------------------------ | --------------------------------------------------------------------------------------- |
| **Taiwan Credit Card Default** | UCI ML Repository | 30,000 samples | Primary modeling & ablation    | Temporal repayment behavior (6 months), demographic attributes, billing/payment amounts |
| **German Credit**              | UCI Statlog       | 1,000 samples  | Interpretability demonstration | Categorical financial stability indicators, loan characteristics                        |

### Dataset Characteristics

## 📁 Project Structure

```
Credit-Risk-Analysis-and-Prediction-Framework/
├── Data/                           # Raw datasets
│   ├── german_credit.csv          # German Credit (Statlog)
│   └── taiwan_default_of_credit_card_clients.csv  # Taiwan dataset
├── data/                          # Processed splits
│   ├── processed_baseline_raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed_baseline_engineered/
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── data/
│   │   └── preprocess.py          # Feature engineering pipeline
│   └── models/
│       ├── baseline.py            # Logistic Regression & LightGBM baselines
│       ├── fuzzy_monotonic.py     # Main fuzzy-monotonic model
│       ├── run_ablation.py        # Ablation orchestrator
│       └── plot_ablation_pr_auc.py # Evaluation plot
├── Latex/
│   ├── extended_ieee.tex          # Extended IEEE manuscript
│   ├── ieee_conference.tex        # Conference template
│   ├── eda.tex                    # EDA report
│   └── elsevier_format.tex        # Journal template
├── results/
│   ├── ablation_table.json
│   ├── baseline_engineered_metrics.json
│   ├── baseline_raw_metrics.json
│   ├── fuzzy_metrics.json
│   ├── fuzzy_monotonic_metrics.json
│   ├── pr_curve.png
│   ├── calibration.png
│   └── shap_fuzzy.png
└── docs/
    ├── ARCHITECTURE.md
    ├── DATA.md
    ├── MODELS.md
    ├── USAGE.md
    └── API.md
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preprocessing                        │
│  • Feature alignment & normalization (RobustScaler)          │
│  • Categorical encoding (LabelEncoder)                       │
│  • Train/test stratified split (80/20)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│            Behavioral Feature Engineering                    │
│  • BILL_AMT_AVG: Mean monthly bill statements                │
│  • Utilization: Bill amount / credit limit                   │
│  • Delinquency intensity: Cumulative payment delays          │
│  • Payment trend: Repayment trajectory                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Fuzzy Membership Layer                          │
│  • Linguistic variables: Low / Medium / High                 │
│  • Percentile-based cut-points (training data)               │
│  • Rule activations: min(AND) operators                      │
│  • Human-readable semantics for risk drivers                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│          Monotonic LightGBM Ensemble                         │
│  • Gradient boosted decision trees                           │
│  • Monotonic constraints on economic priors                  │
│  • Class-balanced training (is_unbalance=True)               │
│  • Calibrated probability outputs                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│             Explainability Layer                             │
│  • SHAP: Feature attribution (global + local)                │
│  • Fuzzy rule activations: Structural transparency           │
│  • Monotonicity: Economic consistency guarantees             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │   Outputs & Evaluation   │
        │  • Default probability   │
        │  • Risk label            │
        │  • Feature attributions  │
        │  • Rule activations      │
        │  • Calibration curves    │
        └──────────────────────────┘
```

---

## � Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- (Optional) Docker dev container

### Installation

```bash
# Clone the repository
- Open-source community (scikit-learn, LightGBM, SHAP)
cd Credit-Risk-Analysis-and-Prediction-Framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn shap
```

### Running the Pipeline

#### 1. Data Preprocessing

```bash
python src/data/preprocess.py --engineered
# ⇒ data/processed_baseline_engineered/{train.csv,test.csv}
```

#### 2. Baseline Models

```bash
python src/models/baseline.py
# ⇒ results/metrics_baseline.json, pr_curve.png, calibration.png
```

#### 3. Fuzzy & Fuzzy-Monotonic Models

```bash
# Full fuzzy-monotonic model
python src/models/fuzzy_monotonic.py

# Fuzzy only (no monotonic constraints)
python src/models/fuzzy_monotonic.py --skip-monotonic --variant fuzzy
# ⇒ results/metrics_fuzzy.json, shap_fuzzy.png
```

#### 4. Ablation Study

```bash
python src/models/run_ablation.py
# ⇒ results/ablation_table.json + markdown summary
```

#### 5. Visualization

```bash
python src/models/plot_ablation_pr_auc.py
# ⇒ results/ablation_pr_auc.png
```

---

## 🔬 Methodology

### 1. Behavioral Feature Engineering

```python
# Aggregated spending behavior
BILL_AMT_AVG = mean(BILL_AMT1, ..., BILL_AMT6)

# Credit utilization
utilization = BILL_AMT_AVG / LIMIT_BAL  # clipped [0, 1]

# Repayment discipline
repay_ratio1 = PAY_AMT1 / BILL_AMT1  # clipped [0, 1]

# Delinquency severity


# Payment trend (behavioral drift)
---
```

### 2. Fuzzy Membership Layer

Linguistic variables defined using percentile-based cutpoints (train-only):

```
Low:    [min, 33rd percentile]
Medium: [25th, 75th percentile]  # overlapping transitions
High:   [67th percentile, max]
```

**Example Rule Activation:**

```
IF utilization=High AND delinquency_intensity=High
THEN risk=High (activation = 0.87)
```

### 3. Monotonic Constraints

| Feature                 | Constraint | Economic Rationale                             |
| ----------------------- | ---------- | ---------------------------------------------- |
| `LIMIT_BAL`             | ↑ → risk ↓ | Higher credit limit implies stronger borrowers |
| `AGE`                   | ↑ → risk ↓ | Older applicants typically more stable         |
| `PAY_0`                 | ↑ → risk ↑ | Recent delinquency signals distress            |
| `utilization`           | ↑ → risk ↑ | High utilization indicates stress              |
| `repay_ratio1`          | ↑ → risk ↓ | Higher repayment reduces risk                  |
| `delinquency_intensity` | ↑ → risk ↑ | Historical delinquency compounds risk          |
| `paytrend`              | ↑ → risk ↓ | Improving trend lowers risk                    |

### 4. Explainability

- **Structural**: Fuzzy rules document linguistic reasoning; monotonic constraints enforce economic priors.
- **Attributional**: SHAP TreeExplainer supplies global importances and local driver narratives.

---

## 📈 Evaluation Metrics

| Metric           | Purpose                    | Why It Matters                                   |
| ---------------- | -------------------------- | ------------------------------------------------ |
| **PR-AUC**       | Precision-recall trade-off | Highlights minority class performance (defaults) |
| **ROC-AUC**      | Discrimination power       | Measures ranking ability across thresholds       |
| **Brier Score**  | Probability calibration    | Evaluates quality of probability estimates       |
| **KS Statistic** | Score separation           | Industry-standard for credit scoring validation  |

❌ **Accuracy is intentionally omitted** (misleading for imbalanced datasets).

---

## 🛠️ Tech Stack

| Layer           | Tooling                               |
| --------------- | ------------------------------------- |
| Data Processing | pandas, numpy, scikit-learn           |
| Modeling        | LightGBM, scikit-learn                |
| Fuzzy Reasoning | Custom percentile-based memberships   |
| Explainability  | SHAP (TreeExplainer)                  |
| Visualization   | matplotlib, seaborn                   |
| Documentation   | LaTeX (Overleaf-compatible), Markdown |

---

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** – System design and components
- **[Data Documentation](docs/DATA.md)** – Datasets, preprocessing, feature engineering
- **[Model Documentation](docs/MODELS.md)** – Model variants, methodology, hyperparameters
- **[Usage Guide](docs/USAGE.md)** – Detailed workflows and troubleshooting
- **[API Reference](docs/API.md)** – Code-level function signatures

---

## 📝 Research Papers

All manuscripts live under `Latex/`:

1. **Extended IEEE** (`extended_ieee.tex`) – Full paper with integrated EDA
2. **IEEE Conference** (`ieee_conference.tex`) – Condensed submission format
3. **Elsevier Journal** (`elsevier_format.tex`) – Journal-ready layout
4. **EDA Report** (`eda.tex`) – Standalone exploratory analysis

Compile via:

```bash
cd Latex/
pdflatex extended_ieee.tex
bibtex extended_ieee
pdflatex extended_ieee.tex
pdflatex extended_ieee.tex
```

---

## 🎓 Citation

If you use this framework, please cite:

```bibtex
@inproceedings{dubey2024fuzzy,
  title={Fuzzy-Monotonic LightGBM for Explainable Credit Default Prediction},
  author={Dubey, Utkarsh and Singla, Kanav and Bhardwaj, Dushyant},
  booktitle={Proceedings of IEEE Conference},
  year={2024},
  organization={Netaji Subhas University of Technology}
}
```

### Dataset Citations

```bibtex
@misc{yeh2009default,
  title={Default of credit card clients dataset},
  author={Yeh, I-Cheng and Lien, Che-hui},
  year={2009},
  publisher={UCI Machine Learning Repository}
}

@misc{german1994statlog,
  title={Statlog (German Credit Data) Dataset},
  publisher={UCI Machine Learning Repository}
}
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

```bash
# Install development dependencies
pip install -r requirements-dev.txt  # when available

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

---

## 📄 License

This project is licensed under the MIT License – see [LICENSE](LICENSE).

---

## 🔗 Links & Resources

- **Repository**: [GitHub](https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework)
- **Issues**: [Report bugs or request features](https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework/issues)
- **Discussions**: [Community Q&A](https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework/discussions)

### External References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Basel II/III Guidelines](https://www.bis.org/basel_framework/)
- [IFRS 9 Financial Instruments](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)

---

## 🏆 Acknowledgments

- **Netaji Subhas University of Technology**, Department of Computer Science
- UCI Machine Learning Repository for datasets
- Open-source community (scikit-learn, LightGBM, SHAP)

---

## 🔮 Future Work

- [ ] Automated monotonic prior discovery
- [ ] Uncertainty quantification for probability estimates
- [ ] Macroeconomic stress testing and drift analysis
- [ ] Conversion to compact scorecards for production
- [ ] Multi-dataset cross-portfolio validation
- [ ] Causal structure constraints
- [ ] Real-time streaming model updates

---

**Built with ❤️ for transparent and responsible credit risk assessment**

## 🔮 Future Work

- [ ] Automated monotonic prior discovery
- [ ] Uncertainty quantification for probability estimates
- [ ] Macroeconomic stress testing and drift analysis
- [ ] Conversion to compact scorecards for production
- [ ] Multi-dataset cross-portfolio validation
- [ ] Causal structure constraints
- [ ] Real-time streaming model updates

---

**Built with ❤️ for transparent and responsible credit risk assessment**

```

```
