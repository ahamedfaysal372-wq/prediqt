# PrediQt - Multi-Domain Fraud Detection Platform

A full-stack fraud detection web application that automatically routes transactions to one of three specialist machine learning models depending on transaction type. Built with Flask, SQLite, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Most fraud detection systems use a single generic model across all transaction types. PrediQt takes a different approach - three specialist Random Forest models, each trained on a different real-world dataset, each optimised for a specific domain.

| Model | Dataset | Transactions | F1 | PR-AUC |
|-------|---------|-------------|-----|--------|
| Credit Card Fraud | Sparkov | 1.85M | 80.12% | 85.43% |
| Online Payment Fraud | IEEE-CIS | 590K | 71.89% | 75.10% |
| Mobile Transfer Fraud | PaySim | 58K | 90.67% | 96.12% |

When a user submits a transaction, PrediQt automatically routes it to the appropriate specialist model based on the transaction type - no manual selection needed.

---

## Features

- **Check Transaction** - Single transaction scoring with auto-routing to the right specialist model
- **Batch Upload** - CSV upload scored row-by-row with a downloadable annotated results file
- **Live Feed** - Real-time transaction monitoring with simulated traffic
- **Alerts** - Fraud-flagged transactions with per-user review and admin notification
- **Explainability** - Leave-one-out feature ablation showing which inputs drove the fraud score
- **Model Performance** - Real precision, recall, F1, and PR-AUC metrics for all three models
- **API Keys** - HMAC-secured key generation with predict/batch scope enforcement and rate limiting (1000/day)
- **Webhooks** - Real-time HMAC-SHA256 signed fraud event delivery to any endpoint
- **API Documentation** - Full endpoint reference at `/api-docs`
- **2FA Authentication** - Email OTP for all regular user accounts (3-minute expiry)
- **Admin Panel** - User management, audit log, role-based access, and fraud notification emails

---

## Tech Stack

- **Backend:** Flask, SQLAlchemy, SQLite
- **ML:** scikit-learn, imbalanced-learn, pandas, numpy, joblib
- **Frontend:** Jinja2, Chart.js, Bootstrap Icons, DM Sans + Fraunces fonts
- **Auth:** Flask-Login, Flask-Mail, bcrypt, HMAC
- **Other:** python-dotenv, requests

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ahamedfaysal372-wq/prediqt.git
cd prediqt
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file inside the `app/` folder

```bash
cd app
cat > .env << 'EOF'
SECRET_KEY=your-secret-key-here
MAIL_PASSWORD=your-gmail-app-password-here
EOF
```

The app uses Gmail to send OTP codes. You'll need a [Gmail app password](https://support.google.com/accounts/answer/185833) to send emails.

### 5. Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Model Files

The three specialist model `.pkl` files are required to run the app. Two are included in this repo (`model_credit_card.pkl` and `model_mobile_transfer.pkl`). The online payment model (`model_online_payment.pkl`) exceeds GitHub's 100MB file size limit and is not included.

To regenerate it, run Section 17 of the Jupyter notebook:

```bash
cd notebook
jupyter notebook fraud_detection.ipynb
```

Run all cells in Section 17 - it will train the model and save it to `model/model_online_payment.pkl` automatically.

---

## API Usage

### Authenticate

Include your API key in every request:

```
Authorization: Bearer pq_live_your_key_here
```

### Check a single transaction

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"transaction_type": "bank_transfer", "amount": 9500, "hour": 3}'
```

### Batch upload

```bash
curl -X POST http://localhost:5000/api/v1/batch \
  -H "Authorization: Bearer YOUR_KEY" \
  -F "transaction_type=bank_transfer" \
  -F "file=@transactions.csv"
```

Full API reference available at `/api-docs` once the app is running.

---

## Webhooks

PrediQt can push real-time notifications to your server the moment fraud is detected. Payloads are signed with HMAC-SHA256 so you can verify they genuinely came from PrediQt.

```python
import hmac, hashlib

def verify_signature(payload_body, signature, secret):
    expected = hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

Set up webhooks at `/webhooks` once logged in.

---

## Project Structure

```
prediqt/
├── app/
│   ├── app.py                  — Main Flask application (~90KB, all routes)
│   ├── templates/              — Jinja2 HTML templates (20 pages)
│   ├── static/
│   │   └── css/                — Per-page stylesheets (18 files)
│   ├── seed_demo_data.py       — Seeds 400 realistic demo predictions
│   └── create_test_users.py    — Creates test user accounts
├── model/
│   ├── model_credit_card.pkl   — Credit card specialist model (Sparkov)
│   ├── model_mobile_transfer.pkl — Mobile transfer specialist model (PaySim)
│   ├── model_online_payment.pkl  — NOT INCLUDED (515MB — regenerate via notebook)
│   ├── scaler_credit_card.pkl
│   ├── scaler_mobile_transfer.pkl
│   ├── scaler_online_payment.pkl
│   └── model_config.json       — Feature lists and thresholds per model
├── notebook/
│   └── fraud_detection.ipynb  — Full ML pipeline (17 sections)
├── requirements.txt
└── README.md
```

---

## Pages

| Section | Pages |
|---------|-------|
| Overview | Landing, Dashboard, Live Feed |
| Detection | Check Transaction, Batch Upload, Alerts |
| Intelligence | Model Performance, Explainability |
| Integration | API Keys, Webhooks, API Docs |
| Admin | Settings, User Management, Audit Log |
| Auth | Login, Register (4-step), 2FA Verify |

---

## Demo Data

To try the app with pre-seeded data:

```bash
cd app
python create_test_users.py
python seed_demo_data.py
```

This creates three test user accounts and seeds 400 realistic predictions across all three models. You'll need Gmail configured in `.env` to receive 2FA codes.

Sample CSVs for batch upload testing are in `app/uploads/`:
- `test_mobile_transfer_fraud.csv` - bank transfer transactions, mostly fraud
- `sample_transactions.csv` - mixed transaction types

---

## Notebook

The Jupyter notebook (`notebook/fraud_detection.ipynb`) covers the full ML pipeline in 17 sections:

- **Sections 1–16:** ULB credit card dataset experiment - baseline models, SMOTE, Random Forest, Gradient Boosting, threshold tuning, feature importance
- **Section 17:** Multi-domain specialist model training - Sparkov, IEEE-CIS, and PaySim datasets with per-model feature engineering, undersampling + SMOTE, and threshold optimisation

---

## Licence

This project is released under the [MIT License](LICENSE). Add a `LICENSE` file to the repo root if you haven't already - GitHub's licence picker makes this a one-click step.
