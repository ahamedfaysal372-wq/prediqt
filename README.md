# PrediQt — Multi-Domain Fraud Detection Platform
## COM668 Individual Project (AT3) | Faysal Ahamed | B00918889 | Ulster University

---

## Project Overview

PrediQt is a Flask-based fraud detection web application that routes transactions 
to one of three specialist machine learning models depending on transaction type:

- **Credit Card Fraud Model** — Sparkov dataset (1.85M transactions, 80% F1)
- **Online Payment Fraud Model** — IEEE-CIS dataset (590K transactions, 72% F1)
- **Mobile Transfer Fraud Model** — PaySim dataset (58K transactions, 91% F1)

---

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
cd app
python app.py
```

### 4. Open in your browser

http://127.0.0.1:5000

---

## Demo Credentials

### Admin Account
| Field | Value |
|-------|-------|
| Email | ahamedfaysal372@gmail.com |
| Password | Admin#123 |
| 2FA | Enabled — a 6-digit verification code is sent by email |

### Test User 

To add a test user you need to register with all the information ased for.

Important: need to use an actual working email as user account will need to go through a 6 digit verification code which will be sent to user's account.

## Authentication Notes

All accounts, including the admin account, use full two-factor authentication.
To test this flow:
1. Log in with any valid account
2. A 6-digit OTP will be sent to the user's email
3. Enter it on the verification page — codes expire after 3 minutes

---

## Project Structure

fraud_detection_project/
├── app/
│   ├── app.py                  — Main Flask application
│   ├── templates/              — All Jinja2 HTML templates
│   ├── static/                 — CSS, JS, images, avatars
│   ├── models/                 — Trained .pkl model files
│   ├── seed_demo_data.py       — Seeds 400 demo predictions
│   └── create_test_users.py    — Creates the three test accounts
├── notebooks/
│   └── fraud_detection.ipynb  — Full ML notebook (17 sections)
├── requirements.txt
└── README.md

---

## Key Features

- **Check Transaction** — Single transaction scoring with auto-routing
- **Batch Upload** — CSV upload scored row-by-row with downloadable results
- **Live Feed** — Real-time transaction monitoring with simulated traffic
- **Alerts** — Fraud-flagged transactions with review and admin notify features
- **Explainability** — Leave-one-out feature ablation per prediction
- **Model Performance** — Real metrics for all three specialist models
- **API Keys** — HMAC-secured key generation with scope enforcement
- **Webhooks** — Real-time HMAC-signed fraud event delivery
- **API Documentation** — Full endpoint reference at /api-docs
- **Admin Panel** — User management, audit log, and role-based access
- **2FA** — Email OTP authentication for all regular user accounts

---

## Notes for Markers

- The database (`fraud_detection.db`) is included so the app runs immediately 
  with pre-seeded demo data across all four accounts.
- The notebook (`fraud_detection.ipynb`) contains 17 sections covering the full 
  ML pipeline — ULB credit card experiment (Sections 1–16) and the specialist 
  model pivot (Section 17).
- Flask-Mail is configured to send from a real Gmail account. All OTP emails 
  and webhook notification emails use this config. If email delivery fails in 
  your environment, admin login still works normally since admin bypasses 2FA.
