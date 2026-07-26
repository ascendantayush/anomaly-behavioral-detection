# Local Setup Guide

A step-by-step guide for running the AI-Powered Behavioral Anomaly Detection system on your local machine. This guide assumes no prior setup — follow each section in order.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Create a Virtual Environment](#3-create-a-virtual-environment)
4. [Install Dependencies](#4-install-dependencies)
5. [Verify Installation](#5-verify-installation)
6. [Run the Pipeline](#6-run-the-pipeline)
7. [Expected Outputs](#7-expected-outputs)
8. [Verify Each Stage](#8-verify-each-stage)
9. [Common Errors and Fixes](#9-common-errors-and-fixes)
10. [Reset Everything](#10-reset-everything)

---

## 1. Prerequisites

### Python 3.12 or Later

Check your Python version:

```bash
python3 --version
```

If you see `Python 3.12.x` or higher, you are ready. If not, or if `python3` is not found:

**macOS (using Homebrew):**

```bash
brew install python@3.12
```

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

**Windows:**
Download Python 3.12 from [python.org](https://www.python.org/downloads/). During installation, check the box that says **"Add Python to PATH"**.

### pip (Python Package Manager)

pip is included with Python. Verify:

```bash
pip --version
```

If you see `pip 24.x` or later, you are good.

### Git

Check if Git is installed:

```bash
git --version
```

If not installed:

**macOS:**

```bash
xcode-select --install
```

**Ubuntu/Debian:**

```bash
sudo apt install git
```

**Windows:**
Download from [git-scm.com](https://git-scm.com/downloads).

---

## 2. Clone the Repository

```bash
git clone https://github.com/your-username/ai-behavioral-anomaly-detection.git
cd ai-behavioral-anomaly-detection
```

Verify the structure:

```bash
ls -la
```

You should see:

```
README.md
LOCAL_SETUP.md
DEPLOYMENT.md
anomaly_detection/
    app.py
    config.py
    requirements.txt
    data/
    features/
    detection/
    classification/
    utils/
```

---

## 3. Create a Virtual Environment

A virtual environment isolates this project's dependencies from your system Python.

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt:

```
(venv) user@machine:~/ai-behavioral-anomaly-detection$
```

---

## 4. Install Dependencies

```bash
pip install -r anomaly_detection/requirements.txt
```

This installs:

- `streamlit` — Dashboard framework
- `pandas` — Data manipulation
- `numpy` — Numerical computation
- `faker` — Synthetic data generation
- `scikit-learn` — Isolation Forest model
- `plotly` — Interactive charts

Installation takes 1–3 minutes depending on your connection.

---

## 5. Verify Installation

Run this single command to verify everything is installed correctly:

```bash
python3 -c "
import streamlit
import pandas
import numpy
import sklearn
import plotly
import faker
print('streamlit', streamlit.__version__)
print('pandas', pandas.__version__)
print('numpy', numpy.__version__)
print('scikit-learn', sklearn.__version__)
print('plotly', plotly.__version__)
print('faker', faker.__version__)
print()
print('All dependencies installed successfully.')
"
```

**Expected output:**

```
streamlit 1.32.x
pandas 2.2.x
numpy 1.26.x
scikit-learn 1.4.x
plotly 5.18.x
faker 24.x

All dependencies installed successfully.
```

---

## 6. Run the Pipeline

### Option A: Launch the Dashboard (Recommended)

This runs the entire pipeline automatically and opens the dashboard in your browser:

```bash
cd anomaly_detection
streamlit run app.py
```

**Expected output:**

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser will automatically open to `http://localhost:8501`. If it doesn't, open that URL manually.

The first run takes 1–3 minutes as it:

1. Generates synthetic data
2. Injects all seven attack patterns
3. Extracts 43 features
4. Trains three detection models
5. Classifies all events and computes risk scores

Subsequent runs are nearly instant due to Streamlit caching.

### Option B: Run the Pipeline Programmatically

If you want to see each stage's output in the terminal:

```bash
cd /path/to/ai-behavioral-anomaly-detection
python3 -c "
import sys
sys.path.insert(0, '.')

print('=== Stage 1: Generating Synthetic Data ===')
from anomaly_detection.data.generator import generate_dataset
df, profiles = generate_dataset(num_users=20, days=14)
print(f'  Generated {len(df)} events for {len(profiles)} users')
print(f'  Columns: {list(df.columns)}')
print(f'  Date range: {df[\"timestamp\"].min()} to {df[\"timestamp\"].max()}')
print()

print('=== Stage 2: Injecting Attacks ===')
from anomaly_detection.data.injector import inject_all_attacks
df = inject_all_attacks(df, profiles)
attacks = df[df['is_attack'] == True]
print(f'  Total events after injection: {len(df)}')
print(f'  Attack events injected: {len(attacks)}')
print(f'  Attack types: {list(attacks[\"attack_type\"].unique())}')
print()

print('=== Stage 3: Feature Engineering ===')
from anomaly_detection.features.engineer import build_feature_matrix
df, feature_cols = build_feature_matrix(df)
print(f'  Features generated: {len(feature_cols)}')
print(f'  Feature columns: {feature_cols[:10]}...')
print()

print('=== Stage 4: Isolation Forest ===')
from anomaly_detection.detection.isolation_forest import IsolationForestDetector
iso = IsolationForestDetector()
iso.fit(df[feature_cols])
df = iso.apply_to_dataframe(df, feature_cols)
print(f'  IF anomalies detected: {df[\"iso_is_anomaly\"].sum()}')
print()

print('=== Stage 5: Rule Engine ===')
from anomaly_detection.detection.rule_engine import build_default_engine
engine = build_default_engine()
df = engine.evaluate(df)
print(f'  Rule anomalies detected: {df[\"rule_anomaly\"].sum()}')
print()

print('=== Stage 6: Markov Model ===')
from anomaly_detection.detection.markov_model import MarkovDetector
markov = MarkovDetector()
markov.fit(df)
df = markov.apply_to_dataframe(df)
print(f'  Markov anomalies detected: {df[\"markov_is_anomaly\"].sum()}')
print()

print('=== Stage 7: Classification & Risk Scoring ===')
from anomaly_detection.classification.classifier import AttackClassifier
classifier = AttackClassifier()
df = classifier.classify(df)
print(f'  Total detections: {(df[\"predicted_attack_type\"] != \"normal\").sum()}')
print()
print('  Attack type distribution:')
print(df['predicted_attack_type'].value_counts().to_string())
print()

tp = ((df['predicted_attack_type'] != 'normal') & (df['is_attack'] == True)).sum()
fp = ((df['predicted_attack_type'] != 'normal') & (df['is_attack'] == False)).sum()
fn = ((df['predicted_attack_type'] == 'normal') & (df['is_attack'] == True)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print('=== Evaluation Metrics ===')
print(f'  Precision: {precision:.2%}')
print(f'  Recall: {recall:.2%}')
print(f'  F1 Score: {f1:.2%}')
print()
print('Pipeline completed successfully.')
"
```

---

## 7. Expected Outputs

### First-Time Dashboard Launch

When you run `streamlit run app.py` for the first time, you will see:

1. **Terminal messages:**

   ```
   Generating synthetic data ...
   Training detection models ...
   ```

2. **Browser opens** to `http://localhost:8501` showing:
   - Title: "AI-Powered Behavioral Anomaly Detection for Cybersecurity"
   - Sidebar with dataset stats (events, users, anomalies)
   - Four tabs: Overview, Live Alerts, Entity Explorer, Threat Analytics

3. **Overview tab** shows:
   - Four KPI cards at the top
   - Daily event volume line chart
   - Attack type distribution pie chart
   - Risk score histogram

### Subsequent Launches

After the first run, the pipeline is cached. You will see:

- No loading spinners
- Dashboard appears instantly
- Same data as the first run (to regenerate, see [Reset Everything](#10-reset-everything))

---

## 8. Verify Each Stage

### Stage 1: Data Generation

**What to check:**

- `len(df)` should be > 1,000 events (default: ~1,500 for 50 users × 30 days)
- `df["event_type"].value_counts()` should show all 10 event types
- `df["user_id"].nunique()` should equal `num_users`

**Quick test:**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from anomaly_detection.data.generator import generate_dataset
df, profiles = generate_dataset(num_users=5, days=7)
print(f'Events: {len(df)}')
print(f'Users: {df[\"user_id\"].nunique()}')
print(f'Event types: {list(df[\"event_type\"].unique())}')
print(f'All False is_attack: {(df[\"is_attack\"] == False).all()}')
"
```

### Stage 2: Attack Injection

**What to check:**

- `df["is_attack"].sum()` should be > 0 (typically 300–600 events)
- `df["attack_type"].value_counts()` should show all 7 attack types
- `df["is_attack"].mean()` should be roughly 15–25% of total events

**Quick test:**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from anomaly_detection.data.generator import generate_dataset
from anomaly_detection.data.injector import inject_all_attacks
df, profiles = generate_dataset(num_users=5, days=7)
df = inject_all_attacks(df, profiles)
print(f'Total: {len(df)}, Attacks: {df[\"is_attack\"].sum()}')
print(df['attack_type'].value_counts())
"
```

### Stage 3: Feature Engineering

**What to check:**

- Should produce exactly 43 feature columns
- No NaN or infinity values in the feature matrix
- Feature matrix shape matches event count

**Quick test:**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from anomaly_detection.data.generator import generate_dataset
from anomaly_detection.data.injector import inject_all_attacks
from anomaly_detection.features.engineer import build_feature_matrix
df, profiles = generate_dataset(num_users=5, days=7)
df = inject_all_attacks(df, profiles)
df, cols = build_feature_matrix(df)
print(f'Features: {len(cols)}')
print(f'NaN count: {df[cols].isna().sum().sum()}')
print(f'Inf count: {np.isinf(df[cols].values).sum()}')
"
```

### Stage 4: Isolation Forest

**What to check:**

- `df["iso_is_anomaly"].sum()` should be > 0
- `df["iso_anomaly_score"]` should be between 0 and 1

### Stage 5: Rule Engine

**What to check:**

- `df["rule_anomaly"].sum()` should be > 0
- `df["rule_names"]` should contain rule names for flagged events

### Stage 6: Markov Model

**What to check:**

- `df["markov_is_anomaly"].sum()` should be > 0
- `df["markov_score"]` should be between 0 and 1

### Stage 7: Classification

**What to check:**

- `df["predicted_attack_type"].value_counts()` should show multiple attack types
- `df["risk_score"]` should range from 0 to 100
- `df["risk_level"].value_counts()` should show at least two levels

---

## 9. Common Errors and Fixes

### `ModuleNotFoundError: No module named 'pandas'`

**Cause:** Dependencies not installed in the active virtual environment.

**Fix:**

```bash
source venv/bin/activate          # Reactivate venv
pip install -r anomaly_detection/requirements.txt
```

### `ModuleNotFoundError: No module named 'anomaly_detection'`

**Cause:** Running from the wrong directory.

**Fix:** Make sure you are in the project root directory (the one containing `anomaly_detection/`):

```bash
cd /path/to/ai-behavioral-anomaly-detection
python3 -c "from anomaly_detection.config import DATA_CFG; print('OK')"
```

### `FileNotFoundError: [Errno 2] No such file or directory`

**Cause:** Running Streamlit from the wrong directory.

**Fix:** Always run from inside the `anomaly_detection/` directory:

```bash
cd anomaly_detection
streamlit run app.py
```

### Streamlit shows "ScriptRunCommand: Terminated"

**Cause:** The terminal was closed or Ctrl+C was pressed.

**Fix:** Simply re-run:

```bash
streamlit run app.py
```

### `ValueError: could not convert string to float`

**Cause:** A pandas `.expanding().apply()` was called on string data (this was fixed in the codebase).

**Fix:** Make sure you are using the latest version of the code. If you see this error, the `features/engineer.py` file may have been modified. Re-clone or update.

### `Port 8501 already in use`

**Cause:** Another Streamlit instance is running.

**Fix:**

```bash
# Find and kill the process
lsof -i :8501
kill <PID>

# Or use a different port
streamlit run app.py --server.port 8502
```

### `MemoryError` or slow performance

**Cause:** Large dataset with many users and days.

**Fix:** Reduce dataset size in `config.py`:

```python
num_users: int = 20    # Instead of 50
days: int = 14          # Instead of 30
events_per_user_per_day: int = 20  # Instead of 40
```

### `Pickle / hash error in Streamlit cache`

**Cause:** Cached data from a previous code version is incompatible.

**Fix:** Clear the Streamlit cache:

```bash
rm -rf ~/.streamlit/cache
# Or in the dashboard, press Ctrl+Shift+R to hard refresh
```

---

## 10. Reset Everything

To completely reset the project to a clean state:

```bash
# Deactivate virtual environment if active
deactivate

# Remove the virtual environment
rm -rf venv/

# Remove Streamlit cache
rm -rf ~/.streamlit/cache/

# Remove any generated files
cd anomaly_detection
rm -rf __pycache__/
rm -rf data/__pycache__/
rm -rf features/__pycache__/
rm -rf detection/__pycache__/
rm -rf classification/__pycache__/
rm -rf utils/__pycache__/

# Recreate virtual environment
cd ..
python3 -m venv venv
source venv/bin/activate

# Reinstall dependencies
pip install -r anomaly_detection/requirements.txt

# Verify
python3 -c "import streamlit; print('Reset complete.')"
```

Then re-launch:

```bash
cd anomaly_detection
streamlit run app.py
```

---

## Quick Reference

| Command                                             | What It Does                       |
| --------------------------------------------------- | ---------------------------------- |
| `source venv/bin/activate`                          | Activate the virtual environment   |
| `deactivate`                                        | Deactivate the virtual environment |
| `pip install -r anomaly_detection/requirements.txt` | Install all dependencies           |
| `streamlit run app.py`                              | Launch the dashboard               |
| `streamlit run app.py --server.port 8502`           | Launch on a different port         |
| `rm -rf ~/.streamlit/cache`                         | Clear Streamlit cache              |
| `python3 -m venv venv`                              | Create a fresh virtual environment |

If you encounter any issue not covered here, check the [Troubleshooting section in README.md](README.md#evaluation) or open a GitHub issue.
