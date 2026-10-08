# PrediQt — Fraud Detection Platform
# Author: Faysal Ahamed | B00918889
# Module: COM668 Computing Project | Ulster University | 2026

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import pandas as pd
import os
import json
import re
import random
import string
import secrets
import hmac
import hashlib
import requests as http_requests
from functools import wraps
from datetime import datetime, timedelta
from flask_mail import Mail, Message
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY']                     = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI']        = 'sqlite:///prediqt.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER']                  = 'uploads'
app.config['MAX_CONTENT_LENGTH']             = 16 * 1024 * 1024

# Flask-Mail config
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = 'ahamedfaysal372@gmail.com'
app.config['MAIL_PASSWORD']       = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = ('PrediQt', 'ahamedfaysal372@gmail.com')

mail = Mail(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['AVATAR_FOLDER'] = 'static/avatars'
os.makedirs(app.config['AVATAR_FOLDER'], exist_ok=True)

ALLOWED_AVATAR_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_avatar_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AVATAR_EXTENSIONS

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# =============================================================================
# MODEL LOADING
# =============================================================================

model_credit_card     = joblib.load('../model/model_credit_card.pkl')
model_online_payment  = joblib.load('../model/model_online_payment.pkl')
model_mobile_transfer = joblib.load('../model/model_mobile_transfer.pkl')

scaler_credit_card     = joblib.load('../model/scaler_credit_card.pkl')
scaler_online_payment  = joblib.load('../model/scaler_online_payment.pkl')
scaler_mobile_transfer = joblib.load('../model/scaler_mobile_transfer.pkl')

with open('../model/model_config.json', 'r') as f:
    MODEL_CONFIG = json.load(f)

model_rf   = joblib.load('../model/random_forest_model.pkl')
model_lr   = joblib.load('../model/logistic_regression_model.pkl')
model_gb   = joblib.load('../model/gradient_boosting_model.pkl')
scaler_ulb = joblib.load('../model/scaler.pkl')

SPECIALIST_MODELS = {
    'credit_card': {
        'model'  : model_credit_card,
        'scaler' : scaler_credit_card,
        'name'   : 'Credit Card Fraud Model',
        'dataset': 'Sparkov',
        'f1'     : 0.8012,
        'config' : MODEL_CONFIG['credit_card']
    },
    'online_payment': {
        'model'  : model_online_payment,
        'scaler' : scaler_online_payment,
        'name'   : 'Online Payment Fraud Model',
        'dataset': 'IEEE-CIS',
        'f1'     : 0.7189,
        'config' : MODEL_CONFIG['online_payment']
    },
    'mobile_transfer': {
        'model'  : model_mobile_transfer,
        'scaler' : scaler_mobile_transfer,
        'name'   : 'Mobile Transfer Fraud Model',
        'dataset': 'PaySim',
        'f1'     : 0.9067,
        'config' : MODEL_CONFIG['mobile_transfer']
    }
}

TRANSACTION_TYPE_ROUTING = {
    'card_purchase'   : 'credit_card',
    'atm'             : 'credit_card',
    'online_shopping' : 'credit_card',
    'contactless'     : 'credit_card',
    'grocery'         : 'credit_card',
    'entertainment'   : 'credit_card',
    'ecommerce'       : 'online_payment',
    'digital_payment' : 'online_payment',
    'subscription'    : 'online_payment',
    'online_transfer' : 'online_payment',
    'card_not_present': 'online_payment',
    'bank_transfer'   : 'mobile_transfer',
    'mobile_payment'  : 'mobile_transfer',
    'cash_withdrawal' : 'mobile_transfer',
    'direct_debit'    : 'mobile_transfer',
}

ULB_FEATURE_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
    'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
    'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
    'V25', 'V26', 'V27', 'V28', 'Amount'
]


# =============================================================================
# DATABASE MODELS
# =============================================================================

class User(UserMixin, db.Model):
    id                    = db.Column(db.Integer, primary_key=True)
    # Step 1 — Contact Details
    name                  = db.Column(db.String(100), nullable=False)
    email                 = db.Column(db.String(150), unique=True, nullable=False)
    phone                 = db.Column(db.String(30), nullable=True)
    country               = db.Column(db.String(100), nullable=True)
    city                  = db.Column(db.String(100), nullable=True)
    avatar_filename       = db.Column(db.String(200), nullable=True)
    # Step 2 — Company Details
    company               = db.Column(db.String(100), nullable=True)
    company_reg_number    = db.Column(db.String(50), nullable=True)
    industry              = db.Column(db.String(100), nullable=True)
    company_size          = db.Column(db.String(50), nullable=True)
    annual_revenue        = db.Column(db.String(50), nullable=True)
    business_address      = db.Column(db.String(200), nullable=True)
    postcode              = db.Column(db.String(20), nullable=True)
    # Step 3 — Usage Details
    job_title             = db.Column(db.String(100), nullable=True)
    use_case              = db.Column(db.String(100), nullable=True)
    monthly_transactions  = db.Column(db.String(50), nullable=True)
    current_fraud_solution= db.Column(db.String(100), nullable=True)
    compliance            = db.Column(db.String(100), nullable=True)
    referral              = db.Column(db.String(100), nullable=True)
    fraud_analysts        = db.Column(db.String(50), nullable=True)
    # Step 4 — Security
    password_hash         = db.Column(db.String(200), nullable=False)
    security_question     = db.Column(db.String(200), nullable=True)
    security_answer       = db.Column(db.String(200), nullable=True)
    notifications         = db.Column(db.String(20), default='email')
    terms_accepted        = db.Column(db.Boolean, default=False)
    # System fields
    user_role             = db.Column(db.String(10), default='user', nullable=False)
    created_at            = db.Column(db.DateTime, default=datetime.utcnow)
    last_login            = db.Column(db.DateTime, nullable=True)
    predictions           = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.user_role == 'admin'


class Prediction(db.Model):
    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount            = db.Column(db.Float, nullable=False)
    transaction_type  = db.Column(db.String(50), nullable=True)
    model_used        = db.Column(db.String(50), nullable=True)
    predicted_label   = db.Column(db.Integer, nullable=False)
    fraud_probability = db.Column(db.Float, nullable=False)
    risk_level        = db.Column(db.String(10), nullable=False)
    timestamp         = db.Column(db.DateTime, default=datetime.utcnow)
    is_batch          = db.Column(db.Boolean, default=False)
    is_reviewed       = db.Column(db.Boolean, default=False)
    features_json     = db.Column(db.Text, nullable=True)


class ApiKey(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name            = db.Column(db.String(100), nullable=False)
    key_prefix      = db.Column(db.String(20), nullable=False)
    key_hash        = db.Column(db.String(200), nullable=False)
    key_type        = db.Column(db.String(10), default='live')
    scope_predict   = db.Column(db.Boolean, default=True)
    scope_batch     = db.Column(db.Boolean, default=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    last_used       = db.Column(db.DateTime, nullable=True)
    is_active       = db.Column(db.Boolean, default=True)
    request_count   = db.Column(db.Integer, default=0)
    requests_today  = db.Column(db.Integer, default=0)
    rate_limit_date = db.Column(db.Date, nullable=True)


class ApiRequestLog(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    api_key_id  = db.Column(db.Integer, db.ForeignKey('api_key.id'), nullable=False)
    endpoint    = db.Column(db.String(50), nullable=False)
    result      = db.Column(db.String(20), nullable=True)
    probability = db.Column(db.Float, nullable=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)


class Webhook(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name            = db.Column(db.String(100), nullable=False)
    url             = db.Column(db.String(500), nullable=False)
    secret          = db.Column(db.String(100), nullable=False)
    event_fraud     = db.Column(db.Boolean, default=True)
    event_batch     = db.Column(db.Boolean, default=True)
    is_active       = db.Column(db.Boolean, default=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    last_triggered  = db.Column(db.DateTime, nullable=True)
    delivery_count  = db.Column(db.Integer, default=0)
    failure_count   = db.Column(db.Integer, default=0)


class WebhookDelivery(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    webhook_id  = db.Column(db.Integer, db.ForeignKey('webhook.id'), nullable=False)
    event_type  = db.Column(db.String(50), nullable=False)
    payload     = db.Column(db.Text, nullable=True)
    status_code = db.Column(db.Integer, nullable=True)
    success     = db.Column(db.Boolean, default=False)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)


class AuditLog(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action    = db.Column(db.String(100), nullable=False)
    details    = db.Column(db.String(300), nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)
    user       = db.relationship('User', foreign_keys=[user_id])


class OTP(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    code       = db.Column(db.String(6), nullable=False)
    otp_type   = db.Column(db.String(20), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    used       = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def is_valid(self):
        return not self.used and datetime.utcnow() < self.expires_at


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =============================================================================
# PASSWORD VALIDATION
# =============================================================================

def validate_password(password):
    errors = []
    if len(password) < 8:
        errors.append('Password must be at least 8 characters.')
    if not re.search(r'[A-Z]', password):
        errors.append('Password must contain at least one uppercase letter.')
    if not re.search(r'[0-9]', password):
        errors.append('Password must contain at least one number.')
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append('Password must contain at least one special character.')
    return errors


def generate_otp(user_id, otp_type='2fa'):
    OTP.query.filter_by(user_id=user_id, otp_type=otp_type, used=False).update({'used': True})
    db.session.commit()

    code = ''.join(random.choices(string.digits, k=6))
    otp  = OTP(
        user_id    = user_id,
        code       = code,
        otp_type   = otp_type,
        expires_at = datetime.utcnow() + timedelta(minutes=3)
    )
    db.session.add(otp)
    db.session.commit()
    return code


def send_otp_email(user, code, otp_type='2fa'):
    if otp_type == '2fa':
        subject = 'PrediQt — Your Login Verification Code'
        body    = f'''
        <div style="font-family:'DM Sans',sans-serif;max-width:480px;margin:0 auto;padding:2rem;">
            <div style="background:#1a1916;border-radius:8px;padding:2rem;margin-bottom:1.5rem;">
                <div style="font-family:'Fraunces',serif;font-size:1.5rem;color:#f7f6f3;font-weight:300;">
                    PrediQt
                </div>
                <div style="font-size:12px;color:rgba(247,246,243,0.4);margin-top:4px;">
                    Predict. Detect. Protect.
                </div>
            </div>
            <h2 style="font-size:1.4rem;color:#1a1916;font-weight:600;margin-bottom:0.5rem;">
                Your verification code
            </h2>
            <p style="font-size:14px;color:#8c8580;line-height:1.7;margin-bottom:1.5rem;">
                Hi {user.name.split()[0]}, enter this code to complete your sign in to PrediQt.
                This code expires in 3 minutes.
            </p>
            <div style="background:#f7f6f3;border:2px solid #e2ddd8;border-radius:8px;
                        padding:2rem;text-align:center;margin-bottom:1.5rem;">
                <div style="font-family:'Fraunces',serif;font-size:3rem;font-weight:300;
                            letter-spacing:0.5rem;color:#1a6b4a;">
                    {code}
                </div>
                <div style="font-size:12px;color:#8c8580;margin-top:0.5rem;">
                    Valid for 3 minutes
                </div>
            </div>
            <p style="font-size:13px;color:#8c8580;line-height:1.7;">
                If you did not request this code, please ignore this email.
                Your account remains secure.
            </p>
        </div>
        '''
    msg      = Message(subject, recipients=[user.email])
    msg.html = body
    mail.send(msg)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_for_transaction(transaction_type):
    model_key  = TRANSACTION_TYPE_ROUTING.get(transaction_type, 'credit_card')
    return model_key, SPECIALIST_MODELS[model_key]


def preprocess_universal(features_dict, model_key):
    config       = MODEL_CONFIG[model_key]
    feature_list = config['features']
    scale_cols   = config['scale_cols']

    defaults = {
        'amount': 0, 'amount_log': 0, 'is_high_amount': 0, 'is_micro': 0,
        'hour': 12, 'is_night': 0, 'is_weekend': 0, 'merchant_category': 2,
        'is_international': 0, 'is_new_merchant': 0, 'balance_change': 0,
        'geo_distance': 0, 'is_far_merchant': 0, 'is_online': 0,
        'gender': 0, 'age': 35, 'city_pop_log': 0,
        'card_type': 0, 'card_category': 0, 'card1': 0, 'card2': 0,
        'suspicious_email': 0, 'email_match': 0,
        'count_match': 0, 'count_distinct': 0, 'count_c3': 0, 'count_c4': 0,
        'count_c5': 0, 'count_c6': 0, 'count_c9': 0, 'count_c11': 0,
        'count_c13': 0, 'count_c14': 0,
        'id_01': 0, 'id_02': 0, 'id_03': 0, 'id_05': 0, 'id_06': 0,
        'id_09': 0, 'id_10': 0, 'id_11': 0, 'id_13': 0, 'id_17': 0,
        'id_19': 0, 'id_20': 0,
        'd1': 0, 'd2': 0, 'd3': 0, 'd4': 0, 'd10': 0, 'd15': 0
    }

    row                  = {**defaults, **features_dict}
    amount               = row['amount']
    row['amount_log']    = float(np.log1p(amount))
    row['is_high_amount']= int(amount > 1000)
    row['is_micro']      = int(amount < 1)
    row['balance_change']= row.get('balance_change', amount)

    df_row = pd.DataFrame([row])
    scaler = SPECIALIST_MODELS[model_key]['scaler']

    if hasattr(scaler, 'feature_names_in_'):
        cols_to_scale = list(scaler.feature_names_in_)
    else:
        cols_to_scale = [c for c in scale_cols if c in df_row.columns]

    for col in cols_to_scale:
        if col not in df_row.columns:
            df_row[col] = 0

    df_row[cols_to_scale] = scaler.transform(df_row[cols_to_scale])

    return df_row[feature_list]


def get_risk_level(probability):
    if probability >= 0.7:
        return 'HIGH'
    elif probability >= 0.3:
        return 'MEDIUM'
    else:
        return 'LOW'


def log_action(user_id, action, details=None):
    db.session.add(AuditLog(
        user_id=user_id, action=action, details=details,
        ip_address=request.remote_addr
    ))
    db.session.commit()


FEATURE_LABELS = {
    'amount': 'Transaction amount', 'hour': 'Time of day', 'is_night': 'Late-night timing',
    'is_weekend': 'Weekend timing', 'merchant_category': 'Merchant category',
    'is_international': 'International transaction', 'is_new_merchant': 'Unfamiliar merchant',
    'is_online': 'Online transaction', 'gender': 'Cardholder gender', 'age': 'Cardholder age',
    'geo_distance': 'Distance from usual location', 'is_far_merchant': 'Unusually far merchant',
    'card_type': 'Card type', 'card_category': 'Card category',
    'suspicious_email': 'Suspicious email domain', 'email_match': 'Email/name mismatch',
    'balance_change': 'Balance change size'
}


def explain_prediction(features_dict, model_key):
    specialist = SPECIALIST_MODELS[model_key]

    defaults = {
        'amount': 0, 'hour': 12, 'is_night': 0, 'is_weekend': 0, 'merchant_category': 2,
        'is_international': 0, 'is_new_merchant': 0, 'is_online': 0,
        'gender': 0, 'age': 35, 'geo_distance': 0, 'is_far_merchant': 0,
        'card_type': 0, 'card_category': 0, 'suspicious_email': 0, 'email_match': 1,
        'balance_change': 0
    }

    df_baseline = preprocess_universal(features_dict, model_key)
    base_prob = float(specialist['model'].predict_proba(df_baseline)[0][1])

    contributions = []
    for feat, value in features_dict.items():
        if feat not in defaults or value == defaults[feat]:
            continue
        modified = dict(features_dict)
        modified[feat] = defaults[feat]
        df_modified = preprocess_universal(modified, model_key)
        modified_prob = float(specialist['model'].predict_proba(df_modified)[0][1])
        contributions.append({'feature': feat, 'impact': round((base_prob - modified_prob) * 100, 1)})

    contributions.sort(key=lambda x: abs(x['impact']), reverse=True)
    return round(base_prob * 100, 2), contributions[:6]


def trigger_webhooks(user_id, event_type, payload):
    webhooks = Webhook.query.filter_by(user_id=user_id, is_active=True).all()
    for wh in webhooks:
        if event_type == 'fraud.detected' and not wh.event_fraud:
            continue
        if event_type == 'batch.completed' and not wh.event_batch:
            continue

        body = json.dumps(payload)
        signature = hmac.new(wh.secret.encode(), body.encode(), hashlib.sha256).hexdigest()

        success, status_code = False, None
        try:
            response = http_requests.post(
                wh.url, data=body,
                headers={'Content-Type': 'application/json',
                         'X-PrediQt-Signature': signature,
                         'X-PrediQt-Event': event_type},
                timeout=3
            )
            status_code = response.status_code
            success = 200 <= response.status_code < 300
        except Exception:
            success = False

        wh.delivery_count += 1
        wh.last_triggered = datetime.utcnow()
        if not success:
            wh.failure_count += 1

        db.session.add(WebhookDelivery(
            webhook_id=wh.id, event_type=event_type, payload=body,
            status_code=status_code, success=success
        ))
    db.session.commit()


DAILY_RATE_LIMIT = 1000


def require_api_key(scope=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing or invalid Authorization header. Use: Bearer <api_key>'}), 401

            raw_key = auth_header[7:]
            matched_key = None
            for k in ApiKey.query.filter_by(is_active=True).all():
                if check_password_hash(k.key_hash, raw_key):
                    matched_key = k
                    break

            if not matched_key:
                return jsonify({'error': 'Invalid or revoked API key.'}), 401

            if scope == 'predict' and not matched_key.scope_predict:
                return jsonify({'error': 'This key does not have access to the Predict scope.'}), 403
            if scope == 'batch' and not matched_key.scope_batch:
                return jsonify({'error': 'This key does not have access to the Batch scope.'}), 403

            today = datetime.utcnow().date()
            if matched_key.rate_limit_date != today:
                matched_key.rate_limit_date = today
                matched_key.requests_today = 0

            if matched_key.requests_today >= DAILY_RATE_LIMIT:
                return jsonify({'error': 'Daily rate limit exceeded (' + str(DAILY_RATE_LIMIT) + ' requests/day).'}), 429

            matched_key.requests_today += 1
            matched_key.request_count += 1
            matched_key.last_used = datetime.utcnow()
            db.session.commit()

            request.api_key = matched_key
            return f(*args, **kwargs)
        return wrapped
    return decorator


def get_dashboard_stats(user_id):
    user = User.query.get(user_id)

    if user.is_admin():
        predictions = Prediction.query.all()
        recent      = Prediction.query.order_by(
                        Prediction.timestamp.desc()).limit(10).all()
    else:
        predictions = Prediction.query.filter_by(user_id=user_id).all()
        recent      = Prediction.query.filter_by(user_id=user_id)\
                        .order_by(Prediction.timestamp.desc()).limit(10).all()

    total       = len(predictions)
    fraud_count = sum(1 for p in predictions if p.predicted_label == 1)
    fraud_rate  = round((fraud_count / total * 100), 2) if total > 0 else 0

    return {
        'total'      : total,
        'fraud_count': fraud_count,
        'legitimate' : total - fraud_count,
        'fraud_rate' : fraud_rate,
        'recent'     : recent
    }


@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}


@app.template_filter('enumerate')
def enumerate_filter(iterable):
    return enumerate(iterable)


# =============================================================================
# PUBLIC ROUTES
# =============================================================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'GET':
        requested_step = request.args.get('step')
        if requested_step:
            return render_template('register.html',
                                   step=int(requested_step), saved=session)
        if session.get('reg_email') and session.get('reg_job_title'):
            return render_template('register.html', step=4, saved=session)
        elif session.get('reg_email') and session.get('reg_company') is not None:
            return render_template('register.html', step=3, saved=session)
        elif session.get('reg_email'):
            return render_template('register.html', step=2, saved=session)
        else:
            return render_template('register.html', step=1, saved=session)

    if request.method == 'POST':
        step = int(request.form.get('step', 1))

        if step == 1:
            first_name = request.form.get('first_name', '').strip()
            last_name  = request.form.get('last_name', '').strip()
            session['reg_name']    = first_name + ' ' + last_name
            session['reg_email']   = request.form.get('email', '').strip()
            session['reg_phone']   = request.form.get('phone', '').strip()
            session['reg_country'] = request.form.get('country', '').strip()
            session['reg_city']    = request.form.get('city', '').strip()
            session['reg_dob']     = request.form.get('dob', '').strip()

            if not first_name or not session['reg_email']:
                flash('Please fill in your name and email.', 'danger')
                return render_template('register.html', step=1, saved=session)

            if User.query.filter_by(email=session['reg_email']).first():
                flash('An account with this email already exists.', 'danger')
                return render_template('register.html', step=1, saved=session)

            return render_template('register.html', step=2, saved=session)

        elif step == 2:
            session['reg_company']        = request.form.get('company', '').strip()
            session['reg_company_reg']    = request.form.get('company_reg_number', '').strip()
            session['reg_industry']       = request.form.get('industry', '').strip()
            session['reg_company_size']   = request.form.get('company_size', '').strip()
            session['reg_annual_revenue'] = request.form.get('annual_revenue', '').strip()
            session['reg_address']        = request.form.get('business_address', '').strip()
            session['reg_postcode']       = request.form.get('postcode', '').strip()

            return render_template('register.html', step=3, saved=session)

        elif step == 3:
            session['reg_job_title']      = request.form.get('job_title', '').strip()
            session['reg_use_case']       = request.form.get('use_case', '').strip()
            session['reg_monthly_tx']     = request.form.get('monthly_transactions', '').strip()
            session['reg_fraud_solution'] = request.form.get('current_fraud_solution', '').strip()
            session['reg_compliance']     = request.form.get('compliance', '').strip()
            session['reg_referral']       = request.form.get('referral', '').strip()
            session['reg_fraud_analysts'] = request.form.get('fraud_analysts', '').strip()

            return render_template('register.html', step=4, saved=session)

        elif step == 4:
            password         = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            terms_accepted   = request.form.get('terms_accepted')

            errors = validate_password(password)
            if errors:
                for error in errors:
                    flash(error, 'danger')
                return render_template('register.html', step=4, saved=session)

            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template('register.html', step=4, saved=session)

            if not terms_accepted:
                flash('You must accept the terms and conditions.', 'danger')
                return render_template('register.html', step=4, saved=session)

            user = User(
                name                   = session.get('reg_name', ''),
                email                  = session.get('reg_email', ''),
                phone                  = session.get('reg_phone', ''),
                country                = session.get('reg_country', ''),
                city                   = session.get('reg_city', ''),
                company                = session.get('reg_company', ''),
                company_reg_number     = session.get('reg_company_reg', ''),
                industry               = session.get('reg_industry', ''),
                company_size           = session.get('reg_company_size', ''),
                annual_revenue         = session.get('reg_annual_revenue', ''),
                business_address       = session.get('reg_address', ''),
                postcode               = session.get('reg_postcode', ''),
                job_title              = session.get('reg_job_title', ''),
                use_case               = session.get('reg_use_case', ''),
                monthly_transactions   = session.get('reg_monthly_tx', ''),
                current_fraud_solution = session.get('reg_fraud_solution', ''),
                compliance             = session.get('reg_compliance', ''),
                referral               = session.get('reg_referral', ''),
                fraud_analysts         = session.get('reg_fraud_analysts', ''),
                security_question      = request.form.get('security_question', ''),
                security_answer        = request.form.get('security_answer', ''),
                notifications          = request.form.get('notifications', 'email'),
                terms_accepted         = True,
                user_role              = 'user'
            )
            user.set_password(password)

            for key in list(session.keys()):
                if key.startswith('reg_'):
                    session.pop(key, None)

            db.session.add(user)
            db.session.commit()
            log_action(user.id, 'Account Created', f'New {user.user_role} account: {user.email}')
            login_user(user)
            flash(f'Welcome to PrediQt, {user.name.split()[0]}!', 'success')
            return redirect(url_for('home'))

    return render_template('register.html', step=1, saved=session)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        user     = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            code = generate_otp(user.id, '2fa')
            try:
                send_otp_email(user, code, '2fa')
                session['2fa_user_id']  = user.id
                session['2fa_remember'] = bool(remember)
                return redirect(url_for('verify_2fa'))
            except Exception as e:
                flash(f'Could not send verification email: {str(e)}', 'danger')
                return render_template('login.html')
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/verify-2fa', methods=['GET', 'POST'])
def verify_2fa():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    user_id = session.get('2fa_user_id')
    if not user_id:
        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'resend':
            code = generate_otp(user.id, '2fa')
            try:
                send_otp_email(user, code, '2fa')
                flash('A new verification code has been sent.', 'success')
            except Exception as e:
                flash(f'Could not send email: {str(e)}', 'danger')
            return redirect(url_for('verify_2fa'))

        entered_code = request.form.get('code', '').strip()
        otp = OTP.query.filter_by(
            user_id  = user_id,
            code     = entered_code,
            otp_type = '2fa',
            used     = False
        ).first()

        if otp and otp.is_valid():
            otp.used = True
            db.session.commit()

            remember = session.pop('2fa_remember', False)
            session.pop('2fa_user_id', None)

            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            log_action(user.id, 'Login', f'Signed in from {user.email}')

            flash(f'Welcome back, {user.name.split()[0]}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid or expired code. Please try again.', 'danger')

    return render_template('verify_2fa.html', email=user.email)


@app.route('/logout')
@login_required
def logout():
    log_action(current_user.id, 'Logout', f'{current_user.email} ended their session')
    logout_user()
    return redirect(url_for('index'))


# =============================================================================
# PRIVATE ROUTES
# =============================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    page = request.args.get('page', 1, type=int)
    per_page = 10

    if current_user.is_admin():
        total_predictions = Prediction.query.count()
        fraud_count       = Prediction.query.filter_by(predicted_label=1).count()
        legitimate        = Prediction.query.filter_by(predicted_label=0).count()
        paginated         = Prediction.query\
                                .order_by(Prediction.timestamp.desc())\
                                .paginate(page=page, per_page=per_page, error_out=False)
    else:
        total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
        fraud_count       = Prediction.query.filter_by(user_id=current_user.id, predicted_label=1).count()
        legitimate        = Prediction.query.filter_by(user_id=current_user.id, predicted_label=0).count()
        paginated         = Prediction.query\
                                .filter_by(user_id=current_user.id)\
                                .order_by(Prediction.timestamp.desc())\
                                .paginate(page=page, per_page=per_page, error_out=False)

    fraud_rate = round((fraud_count / total_predictions * 100), 1) if total_predictions > 0 else 0

    stats = {
        'total'       : total_predictions,
        'fraud_count' : fraud_count,
        'legitimate'  : legitimate,
        'fraud_rate'  : fraud_rate,
        'recent'      : paginated.items,
        'pagination'  : paginated
    }

    return render_template('dashboard.html', stats=stats)


@app.route('/export/predictions')
@login_required
def export_predictions():
    import csv
    import io
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                  .order_by(Prediction.timestamp.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Amount', 'Transaction Type', 'Result',
                     'Fraud Probability', 'Risk Level', 'Model Used',
                     'Batch', 'Timestamp'])
    for p in predictions:
        writer.writerow([
            p.id,
            p.amount,
            p.transaction_type or '',
            'FRAUD' if p.predicted_label == 1 else 'LEGITIMATE',
            round(p.fraud_probability * 100, 1),
            p.risk_level,
            p.model_used or '',
            'Yes' if p.is_batch else 'No',
            p.timestamp.strftime('%d %b %Y %H:%M')
        ])
    output.seek(0)
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment;filename=prediqt_predictions.csv'
        }
    )


@app.route('/home')
@login_required
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None

    if request.method == 'POST':
        try:
            transaction_type      = request.form.get('transaction_type', 'card_purchase')
            amount                = float(request.form.get('amount', 0))
            hour                  = int(request.form.get('hour', 12))
            model_key, specialist = get_model_for_transaction(transaction_type)
            threshold             = specialist['config']['threshold']

            features = {
                'amount'           : amount,
                'hour'             : hour,
                'is_night'         : int(0 <= hour <= 5),
                'is_weekend'       : int(request.form.get('is_weekend', 0)),
                'merchant_category': int(request.form.get('merchant_category', 2)),
                'is_international' : int(request.form.get('is_international', 0)),
                'is_new_merchant'  : int(request.form.get('is_new_merchant', 0)),
                'is_online'        : int(transaction_type in [
                    'ecommerce', 'digital_payment', 'subscription',
                    'online_transfer', 'card_not_present', 'online_shopping'
                ]),
            }

            if model_key == 'credit_card':
                features['gender']          = int(request.form.get('gender', 0))
                features['age']             = int(request.form.get('age', 35))
                features['city_pop_log']    = float(np.log1p(
                    float(request.form.get('city_pop', 100000))))
                features['geo_distance']    = float(request.form.get('geo_distance', 0))
                features['is_far_merchant'] = int(features['geo_distance'] > 2)

            elif model_key == 'online_payment':
                card_type_map  = {'visa': 0, 'mastercard': 1,
                                  'american express': 2, 'discover': 3}
                card_cat_map   = {'debit': 0, 'credit': 1,
                                  'debit or credit': 2, 'charge card': 3}
                features['card_type']        = card_type_map.get(
                    request.form.get('card_type', 'visa'), 0)
                features['card_category']    = card_cat_map.get(
                    request.form.get('card_category', 'debit'), 0)
                features['suspicious_email'] = int(request.form.get('suspicious_email', 0))
                features['email_match']      = int(request.form.get('email_match', 1))
                features['geo_distance']     = float(request.form.get('geo_distance', 0))
                features['is_far_merchant']  = int(features['geo_distance'] > 100)

            elif model_key == 'mobile_transfer':
                transfer_type_map             = {
                    'CASH_OUT': 0, 'PAYMENT': 1,
                    'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4
                }
                transfer_type                 = request.form.get('transfer_type', 'PAYMENT')
                features['merchant_category'] = transfer_type_map.get(transfer_type, 1)
                features['is_international']  = int(transfer_type == 'TRANSFER')
                features['is_new_merchant']   = int(transfer_type == 'CASH_OUT')
                old_balance                   = float(request.form.get('old_balance', 0))
                new_balance                   = float(request.form.get('new_balance', 0))
                features['balance_change']    = abs(new_balance - old_balance)

            df_processed = preprocess_universal(features, model_key)
            probability  = specialist['model'].predict_proba(df_processed)[0][1]
            prediction   = 1 if probability >= threshold else 0
            risk_level   = get_risk_level(probability)

            pred = Prediction(
                user_id           = current_user.id,
                amount            = amount,
                transaction_type  = transaction_type,
                model_used        = model_key,
                predicted_label   = prediction,
                fraud_probability = round(probability, 4),
                risk_level        = risk_level
            )
            db.session.add(pred)
            db.session.commit()

            if prediction == 1:
                trigger_webhooks(current_user.id, 'fraud.detected', {
                    'event': 'fraud.detected', 'amount': amount, 'transaction_type': transaction_type,
                    'probability': round(probability, 4), 'risk_level': risk_level,
                    'model_used': specialist['name'], 'timestamp': datetime.utcnow().isoformat()
                })

            result = {
                'label'           : 'FRAUD' if prediction == 1 else 'LEGITIMATE',
                'probability'     : round(probability * 100, 2),
                'risk_level'      : risk_level,
                'amount'          : amount,
                'threshold'       : round(threshold, 4),
                'model_used'      : specialist['name'],
                'model_key'       : model_key,
                'dataset'         : specialist['dataset'],
                'transaction_type': transaction_type
            }

            session['last_predict_result'] = result
            return redirect(url_for('predict_result'))

        except Exception as e:
            flash(f'Error processing transaction: {str(e)}', 'danger')

    return render_template('predict.html', result=None)


@app.route('/predict/result')
@login_required
def predict_result():
    result = session.get('last_predict_result')
    if not result:
        flash('No recent result to show.', 'danger')
        return redirect(url_for('predict'))
    return render_template('predict_result.html', result=result)

 
@app.route('/batch', methods=['GET', 'POST'])
@login_required
def batch():
    last_input  = session.get('last_batch_input')
    last_output = session.get('last_batch_output')

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('batch'))

        file = request.files['file']

        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('batch'))

        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                df_input              = pd.read_csv(filepath)
                transaction_type      = request.form.get('transaction_type', 'card_purchase')
                model_key, specialist = get_model_for_transaction(transaction_type)
                threshold             = specialist['config']['threshold']
                feature_list          = specialist['config']['features']
                scale_cols            = specialist['config']['scale_cols']

                required = ['amount']
                missing  = [c for c in required if c not in
                           [col.lower() for col in df_input.columns]]
                if missing:
                    flash(f'Missing required columns: {missing}', 'danger')
                    return redirect(url_for('batch'))

                df_input.columns = [c.lower() for c in df_input.columns]

                defaults = {
                    'amount': 0, 'amount_log': 0, 'is_high_amount': 0,
                    'is_micro': 0, 'hour': 12, 'is_night': 0, 'is_weekend': 0,
                    'merchant_category': 2, 'is_international': 0,
                    'is_new_merchant': 0, 'balance_change': 0,
                    'geo_distance': 0, 'is_far_merchant': 0, 'is_online': 0,
                    'gender': 0, 'age': 35, 'city_pop_log': 0,
                    'card_type': 0, 'card_category': 0, 'card1': 0, 'card2': 0,
                    'suspicious_email': 0, 'email_match': 0,
                    'count_match': 0, 'count_distinct': 0, 'count_c3': 0,
                    'count_c4': 0, 'count_c5': 0, 'count_c6': 0,
                    'count_c9': 0, 'count_c11': 0, 'count_c13': 0, 'count_c14': 0,
                    'id_01': 0, 'id_02': 0, 'id_03': 0, 'id_05': 0, 'id_06': 0,
                    'id_09': 0, 'id_10': 0, 'id_11': 0, 'id_13': 0, 'id_17': 0,
                    'id_19': 0, 'id_20': 0,
                    'd1': 0, 'd2': 0, 'd3': 0, 'd4': 0, 'd10': 0, 'd15': 0
                }

                df_features = pd.DataFrame()
                for feat in feature_list:
                    if feat in df_input.columns:
                        col = pd.to_numeric(df_input[feat], errors='coerce')
                        df_features[feat] = col.fillna(defaults.get(feat, 0))
                    else:
                        df_features[feat] = defaults.get(feat, 0)

                df_features['amount_log']     = np.log1p(df_features['amount'])
                df_features['is_high_amount'] = (df_features['amount'] > 1000).astype(int)
                df_features['is_micro']       = (df_features['amount'] < 1).astype(int)

                scaler        = specialist['scaler']
                cols_to_scale = [c for c in scale_cols if c in df_features.columns]
                df_features[cols_to_scale] = scaler.transform(df_features[cols_to_scale])

                probabilities = specialist['model'].predict_proba(
                    df_features[feature_list])[:, 1]
                predictions   = (probabilities >= threshold).astype(int)

                df_input['Predicted_Label']   = predictions
                df_input['Fraud_Probability'] = probabilities.round(4)
                df_input['Risk_Level']        = [get_risk_level(p) for p in probabilities]
                df_input['Result']            = ['FRAUD' if p == 1 else 'LEGITIMATE'
                                                  for p in predictions]
                df_input['Model_Used']        = specialist['name']

                for _, row in df_input.iterrows():
                    pred = Prediction(
                        user_id           = current_user.id,
                        amount            = float(row.get('amount', 0)),
                        transaction_type  = transaction_type,
                        model_used        = model_key,
                        predicted_label   = int(row['Predicted_Label']),
                        fraud_probability = float(row['Fraud_Probability']),
                        risk_level        = row['Risk_Level'],
                        is_batch          = True
                    )
                    db.session.add(pred)
                db.session.commit()

                if int(predictions.sum()) > 0:
                    trigger_webhooks(current_user.id, 'batch.completed', {
                        'event': 'batch.completed', 'total': len(df_input),
                        'fraud_count': int(predictions.sum()), 'fraud_rate': round(predictions.mean() * 100, 2),
                        'timestamp': datetime.utcnow().isoformat()
                    })

                output_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'results_' + filename)
                df_input.to_csv(output_path, index=False)

                session['last_batch_input']  = filename
                session['last_batch_output'] = 'results_' + filename
                return redirect(url_for('batch_results', filename='results_' + filename))

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
        else:
            flash('Please upload a valid CSV file.', 'danger')

    return render_template('batch.html', last_input=last_input, last_output=last_output)


@app.route('/batch/results/<filename>')
@login_required
def batch_results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('Results file not found.', 'danger')
        return redirect(url_for('batch'))

    df = pd.read_csv(filepath)
    predictions = df['Predicted_Label']

    page = request.args.get('page', 1, type=int)
    per_page = 20
    total_rows = len(df)
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page

    results = {
        'total'      : total_rows,
        'fraud'      : int(predictions.sum()),
        'legitimate' : int((predictions == 0).sum()),
        'fraud_rate' : round(predictions.mean() * 100, 2),
        'rows'       : df.iloc[start:end].to_dict('records'),
        'output_file': filename,
        'page'       : page,
        'total_pages': total_pages,
        'per_page'   : per_page
    }

    return render_template('batch_results.html', results=results)


@app.route('/batch/clear')
@login_required
def batch_clear():
    session.pop('last_batch_input', None)
    session.pop('last_batch_output', None)
    return redirect(url_for('batch') + '#batchForm')


@app.route('/live-feed')
@login_required
def live_feed():
    return render_template('live_feed.html')


@app.route('/performance')
@login_required
def performance():
    specialist_metrics = {
        'credit_card': {
            'name'     : 'Credit Card Fraud',
            'dataset'  : 'Sparkov (1.85M transactions)',
            'precision': 0.8249, 'recall': 0.7788,
            'f1'       : 0.8012, 'roc_auc': 0.9974, 'pr_auc': 0.8543,
            'fp'       : 319
        },
        'online_payment': {
            'name'     : 'Online Payment Fraud',
            'dataset'  : 'IEEE-CIS (590K transactions)',
            'precision': 0.8197, 'recall': 0.6402,
            'f1'       : 0.7189, 'roc_auc': 0.9327, 'pr_auc': 0.7510,
            'fp'       : 582
        },
        'mobile_transfer': {
            'name'     : 'Mobile Transfer Fraud',
            'dataset'  : 'PaySim (58K transactions)',
            'precision': 0.9177, 'recall': 0.8959,
            'f1'       : 0.9067, 'roc_auc': 0.9936, 'pr_auc': 0.9612,
            'fp'       : 132
        }
    }
    return render_template('performance.html', specialist_metrics=specialist_metrics)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'profile':
            current_user.phone = request.form.get('phone', '').strip()
            current_user.job_title = request.form.get('job_title', '').strip()
            current_user.country = request.form.get('country', '').strip()
            current_user.city = request.form.get('city', '').strip()
            db.session.commit()
            log_action(current_user.id, 'Profile Updated', f'Phone, job title, city, or country changed')
            flash('Profile updated.', 'success')
            return redirect(url_for('settings') + '#settingsLayout')

        elif form_type == 'company':
            current_user.company = request.form.get('company', '').strip()
            current_user.company_reg_number = request.form.get('company_reg_number', '').strip()
            current_user.industry = request.form.get('industry', '').strip()
            current_user.company_size = request.form.get('company_size', '').strip()
            current_user.business_address = request.form.get('business_address', '').strip()
            current_user.postcode = request.form.get('postcode', '').strip()
            db.session.commit()
            log_action(current_user.id, 'Company Details Updated', f'{current_user.company or "Company info"} updated')
            flash('Company details updated.', 'success')
            return redirect(url_for('settings') + '#settingsLayout')

        elif form_type == 'notifications':
            current_user.notifications = request.form.get('notifications', 'none')
            db.session.commit()
            log_action(current_user.id, 'Notification Preferences Updated')
            flash('Notification preferences updated.', 'success')
            return redirect(url_for('settings') + '#settingsLayout')

        elif form_type == 'password':
            current_pw = request.form.get('current_password', '')
            new_pw = request.form.get('new_password', '')
            confirm_pw = request.form.get('confirm_password', '')

            if not current_user.check_password(current_pw):
                flash('Current password is incorrect.', 'danger')
                return redirect(url_for('settings') + '#settingsLayout')

            errors = validate_password(new_pw)
            if errors:
                for e in errors:
                    flash(e, 'danger')
                return redirect(url_for('settings') + '#settingsLayout')

            if new_pw != confirm_pw:
                flash('New passwords do not match.', 'danger')
                return redirect(url_for('settings') + '#settingsLayout')

            current_user.set_password(new_pw)
            db.session.commit()
            log_action(current_user.id, 'Password Changed', 'Password updated via Settings')
            flash('Password updated successfully.', 'success')
            return redirect(url_for('settings') + '#settingsLayout')

    return render_template('settings.html')


@app.route('/settings/avatar/upload', methods=['POST'])
@login_required
def upload_avatar():
    if 'avatar' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('settings') + '#settingsLayout')

    file = request.files['avatar']
    if file.filename == '' or not allowed_avatar_file(file.filename):
        flash('Please upload a PNG, JPG, or WEBP image.', 'danger')
        return redirect(url_for('settings') + '#settingsLayout')

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f'user_{current_user.id}_{secrets.token_hex(6)}.{ext}'

    if current_user.avatar_filename:
        old_path = os.path.join(app.config['AVATAR_FOLDER'], current_user.avatar_filename)
        if os.path.exists(old_path):
            os.remove(old_path)

    file.save(os.path.join(app.config['AVATAR_FOLDER'], filename))
    current_user.avatar_filename = filename
    db.session.commit()
    log_action(current_user.id, 'Avatar Updated', f'New profile picture uploaded ({filename})')
    flash('Profile picture updated.', 'success')
    return redirect(url_for('settings') + '#settingsLayout')


@app.route('/settings/avatar/delete', methods=['POST'])
@login_required
def delete_avatar():
    if current_user.avatar_filename:
        old_path = os.path.join(app.config['AVATAR_FOLDER'], current_user.avatar_filename)
        if os.path.exists(old_path):
            os.remove(old_path)
        current_user.avatar_filename = None
        db.session.commit()
        log_action(current_user.id, 'Avatar Removed', f'Profile picture removed')
        flash('Profile picture removed.', 'success')
    return redirect(url_for('settings') + '#settingsLayout')


@app.route('/users')
@login_required
def users():
    if not current_user.is_admin():
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))

    search = request.args.get('search', '').strip()
    role_filter = request.args.get('role', 'all')
    page = request.args.get('page', 1, type=int)
    per_page = 10

    query = User.query
    if search:
        query = query.filter(db.or_(User.name.ilike(f'%{search}%'), User.email.ilike(f'%{search}%')))
    if role_filter != 'all':
        query = query.filter(User.user_role == role_filter)

    paginated = query.order_by(User.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)

    user_stats = {}
    for u in paginated.items:
        user_stats[u.id] = {
            'total': Prediction.query.filter_by(user_id=u.id).count(),
            'fraud': Prediction.query.filter_by(user_id=u.id, predicted_label=1).count()
        }

    return render_template('users.html', users=paginated, search=search, role_filter=role_filter, user_stats=user_stats,
                            total_users=User.query.count(),
                            admin_count=User.query.filter_by(user_role='admin').count())


@app.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))

    target = User.query.get_or_404(user_id)

    if target.id == current_user.id:
        flash("You can't delete your own account.", 'danger')
        return redirect(url_for('users') + '#usersTable')

    api_key_ids = [k.id for k in ApiKey.query.filter_by(user_id=target.id).all()]
    if api_key_ids:
        ApiRequestLog.query.filter(ApiRequestLog.api_key_id.in_(api_key_ids)).delete(synchronize_session=False)
    ApiKey.query.filter_by(user_id=target.id).delete(synchronize_session=False)

    webhook_ids = [w.id for w in Webhook.query.filter_by(user_id=target.id).all()]
    if webhook_ids:
        WebhookDelivery.query.filter(WebhookDelivery.webhook_id.in_(webhook_ids)).delete(synchronize_session=False)
    Webhook.query.filter_by(user_id=target.id).delete(synchronize_session=False)

    Prediction.query.filter_by(user_id=target.id).delete(synchronize_session=False)
    OTP.query.filter_by(user_id=target.id).delete(synchronize_session=False)
    AuditLog.query.filter_by(user_id=target.id).update({'user_id': None})

    deleted_email = target.email
    db.session.delete(target)
    db.session.commit()

    log_action(current_user.id, 'User Deleted', f'Removed account: {deleted_email}')
    return redirect(url_for('users') + '#usersTable')


@app.route('/audit-log')
@login_required
def audit_log():
    if not current_user.is_admin():
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))

    page = request.args.get('page', 1, type=int)
    per_page = 25
    action_filter = request.args.get('action', 'all')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    query = AuditLog.query.outerjoin(User, AuditLog.user_id == User.id)

    if action_filter != 'all':
        query = query.filter(AuditLog.action == action_filter)
    if start_date:
        try:
            query = query.filter(AuditLog.timestamp >= datetime.strptime(start_date, '%Y-%m-%d'))
        except ValueError:
            pass
    if end_date:
        try:
            query = query.filter(AuditLog.timestamp < datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))
        except ValueError:
            pass

    paginated = query.order_by(AuditLog.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
    distinct_actions = sorted([a[0] for a in db.session.query(AuditLog.action).distinct().all()])

    return render_template('audit_log.html', logs=paginated, action_filter=action_filter,
                            distinct_actions=distinct_actions, start_date=start_date, end_date=end_date)


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/download/<filename>')
@login_required
def download_results(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)


@app.route('/api/dashboard-stats')
@login_required
def api_dashboard_stats():
    stats = get_dashboard_stats(current_user.id)
    return jsonify({
        'total'      : stats['total'],
        'fraud_count': stats['fraud_count'],
        'legitimate' : stats['legitimate'],
        'fraud_rate' : stats['fraud_rate']
    })


@app.route('/api/last-prediction-id')
@login_required
def api_last_prediction_id():
    if current_user.is_admin():
        last = Prediction.query.order_by(Prediction.id.desc()).first()
    else:
        last = Prediction.query.filter_by(user_id=current_user.id)\
                               .order_by(Prediction.id.desc()).first()
    return jsonify({'last_id': last.id if last else 0})


@app.route('/api/recent-predictions')
@login_required
def api_recent_predictions():
    since_id = request.args.get('since', 0, type=int)

    if current_user.is_admin():
        query = Prediction.query.filter(Prediction.id > since_id)
    else:
        query = Prediction.query.filter(
            Prediction.id > since_id,
            Prediction.user_id == current_user.id
        )

    predictions = query.order_by(Prediction.id.asc()).limit(20).all()

    return jsonify([{
        'id'              : p.id,
        'amount'          : p.amount,
        'transaction_type': p.transaction_type or 'unknown',
        'is_fraud'        : bool(p.predicted_label == 1),
        'risk_level'      : p.risk_level,
        'probability'     : p.fraud_probability,
        'timestamp'       : p.timestamp.strftime('%H:%M:%S'),
        'simulated'       : False,
        'model_used'      : SPECIALIST_MODELS.get(p.model_used, {}).get('name', p.model_used or 'Unknown'),
        'user_name'       : p.user.name if current_user.is_admin() else None,
    } for p in predictions])


@app.route('/api/live-feed')
@login_required
def api_live_feed():
    transaction_type      = random.choice(list(TRANSACTION_TYPE_ROUTING.keys()))
    model_key, specialist = get_model_for_transaction(transaction_type)
    threshold             = specialist['config']['threshold']

    if random.random() < 0.85:
        amount = round(random.uniform(1, 1200), 2)
    else:
        amount = round(random.uniform(1200, 5000), 2)
    hour   = random.randint(0, 23)

    features = {
        'amount'           : amount,
        'hour'             : hour,
        'is_night'         : int(0 <= hour <= 5),
        'is_weekend'       : random.choice([0, 1]),
        'merchant_category': random.randint(0, 13),
        'is_international' : random.choice([0, 0, 0, 1]),
        'is_new_merchant'  : random.choice([0, 1]),
        'is_online'        : int(transaction_type in [
            'ecommerce', 'digital_payment', 'subscription',
            'online_transfer', 'card_not_present', 'online_shopping'
        ]),
    }

    if model_key == 'credit_card':
        features['gender']          = random.choice([0, 1])
        features['age']             = random.randint(18, 75)
        features['city_pop_log']    = float(np.log1p(random.uniform(1000, 500000)))
        features['geo_distance']    = random.uniform(0, 5)
        features['is_far_merchant'] = int(features['geo_distance'] > 2)
    elif model_key == 'online_payment':
        features['card_type']        = random.randint(0, 3)
        features['card_category']    = random.randint(0, 3)
        features['suspicious_email'] = random.choice([0, 0, 0, 1])
        features['email_match']      = random.choice([0, 1])
        features['geo_distance']     = random.uniform(0, 200)
        features['is_far_merchant']  = int(features['geo_distance'] > 100)
    elif model_key == 'mobile_transfer':
        features['merchant_category'] = random.randint(0, 4)
        features['is_international']  = random.choice([0, 0, 1])
        features['is_new_merchant']   = random.choice([0, 1])
        features['balance_change']    = random.uniform(0, amount)

    df_processed = preprocess_universal(features, model_key)
    probability  = float(specialist['model'].predict_proba(df_processed)[0][1])
    prediction   = 1 if probability >= threshold else 0
    risk_level   = get_risk_level(probability)

    return jsonify({
        'amount'          : amount,
        'hour'            : hour,
        'transaction_type': transaction_type,
        'is_fraud'        : bool(prediction == 1),
        'risk_level'      : risk_level,
        'probability'     : round(probability, 2),
        'timestamp'       : datetime.utcnow().strftime('%H:%M:%S'),
        'simulated'       : True,
        'model_used'      : specialist['name']
    })


@app.route('/api/routing-info')
@login_required
def api_routing_info():
    return jsonify(TRANSACTION_TYPE_ROUTING)


@app.route('/api/check-email')
def api_check_email():
    email = request.args.get('email', '')
    exists = User.query.filter_by(email=email).first() is not None
    return jsonify({'exists': exists})


@app.route('/alerts')
@login_required
def alerts():
    page = request.args.get('page', 1, type=int)
    per_page = 15
    severity = request.args.get('severity', 'all')
    status = request.args.get('status', 'all')

    if current_user.is_admin():
        query = Prediction.query.filter(Prediction.predicted_label == 1)
    else:
        query = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.user_id == current_user.id)

    total_caught = query.count()

    if severity != 'all':
        query = query.filter(Prediction.risk_level == severity.upper())
    if status == 'reviewed':
        query = query.filter(Prediction.is_reviewed == True)
    elif status == 'unreviewed':
        query = query.filter(Prediction.is_reviewed == False)

    query = query.order_by(Prediction.timestamp.desc())
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    if current_user.is_admin():
        total_unreviewed = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.is_reviewed == False).count()
    else:
        total_unreviewed = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.is_reviewed == False, Prediction.user_id == current_user.id).count()

    model_names = {k: v['name'] for k, v in SPECIALIST_MODELS.items()}
    model_thresholds = {k: v['config']['threshold'] for k, v in SPECIALIST_MODELS.items()}

    high_count = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.risk_level == 'HIGH')
    medium_count = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.risk_level == 'MEDIUM')
    low_count = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.risk_level == 'LOW')
    if not current_user.is_admin():
        high_count = high_count.filter(Prediction.user_id == current_user.id)
        medium_count = medium_count.filter(Prediction.user_id == current_user.id)
        low_count = low_count.filter(Prediction.user_id == current_user.id)

    return render_template('alerts.html', alerts=paginated, severity=severity, status=status,
                            total_unreviewed=total_unreviewed, total_caught=total_caught,
                            high_count=high_count.count(), medium_count=medium_count.count(), low_count=low_count.count(),
                            model_names=model_names, model_thresholds=model_thresholds)


@app.route('/api/alerts/<int:alert_id>/review', methods=['POST'])
@login_required
def api_mark_reviewed(alert_id):
    pred = Prediction.query.get_or_404(alert_id)
    if pred.user_id != current_user.id:
        return jsonify({'error': 'You can only review your own alerts.'}), 403
    pred.is_reviewed = True
    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/alerts/<int:alert_id>/notify', methods=['POST'])
@login_required
def api_notify_alert_owner(alert_id):
    if not current_user.is_admin():
        return jsonify({'error': 'Forbidden'}), 403

    pred = Prediction.query.get_or_404(alert_id)
    owner = User.query.get(pred.user_id)

    if not owner:
        return jsonify({'error': 'Could not find the account this alert belongs to.'}), 404

    try:
        subject = 'PrediQt — A flagged transaction needs your review'
        body = f'''
        <div style="font-family:'DM Sans',sans-serif;max-width:480px;margin:0 auto;padding:2rem;">
            <div style="background:#1a1916;border-radius:8px;padding:2rem;margin-bottom:1.5rem;">
                <div style="font-family:'Fraunces',serif;font-size:1.5rem;color:#f7f6f3;font-weight:300;">PrediQt</div>
            </div>
            <h2 style="font-size:1.4rem;color:#1a1916;font-weight:600;margin-bottom:0.5rem;">
                A flagged transaction on your account
            </h2>
            <p style="font-size:14px;color:#8c8580;line-height:1.7;margin-bottom:1.5rem;">
                Hi {owner.name.split()[0]}, an administrator has flagged the following transaction
                on your PrediQt account and asked that you review it.
            </p>
            <div style="background:#f7f6f3;border:2px solid #e2ddd8;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem;">
                <div style="font-size:13px;color:#8c8580;margin-bottom:4px;">Amount</div>
                <div style="font-size:1.5rem;font-weight:700;color:#1a1916;margin-bottom:1rem;">£{pred.amount:.2f}</div>
                <div style="font-size:13px;color:#8c8580;margin-bottom:4px;">Risk Level</div>
                <div style="font-size:1rem;font-weight:700;color:#dc2626;margin-bottom:1rem;">{pred.risk_level}</div>
                <div style="font-size:13px;color:#8c8580;margin-bottom:4px;">Fraud Probability</div>
                <div style="font-size:1rem;font-weight:700;color:#1a1916;">{round(pred.fraud_probability * 100, 1)}%</div>
            </div>
            <p style="font-size:13px;color:#8c8580;line-height:1.7;">
                Please log in to PrediQt and visit your Alerts page to review and mark it as reviewed.
            </p>
        </div>
        '''
        msg = Message(subject, recipients=[owner.email])
        msg.html = body
        mail.send(msg)

        log_action(current_user.id, 'Alert Notification Sent', f'Notified {owner.email} about alert #{alert_id}')
        return jsonify({'success': True, 'sent_to': owner.email})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/unread-alerts-count')
@login_required
def api_unread_alerts_count():
    if current_user.is_admin():
        count = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.is_reviewed == False).count()
    else:
        count = Prediction.query.filter(Prediction.predicted_label == 1, Prediction.is_reviewed == False, Prediction.user_id == current_user.id).count()
    return jsonify({'count': count})


@app.route('/explainability', methods=['GET', 'POST'])
@login_required
def explainability():
    result = None

    if request.method == 'POST':
        try:
            transaction_type = request.form.get('transaction_type', 'card_purchase')
            amount = float(request.form.get('amount', 0))
            hour = int(request.form.get('hour', 12))
            model_key, specialist = get_model_for_transaction(transaction_type)
            threshold = specialist['config']['threshold']

            features = {
                'amount': amount,
                'hour': hour,
                'is_night': int(0 <= hour <= 5),
                'is_weekend': int(request.form.get('is_weekend', 0)),
                'merchant_category': int(request.form.get('merchant_category', 2)),
                'is_international': int(request.form.get('is_international', 0)),
                'is_new_merchant': int(request.form.get('is_new_merchant', 0)),
                'is_online': int(transaction_type in [
                    'ecommerce', 'digital_payment', 'subscription',
                    'online_transfer', 'card_not_present', 'online_shopping'
                ]),
            }

            if model_key == 'credit_card':
                features['gender'] = int(request.form.get('gender', 0))
                features['age'] = int(request.form.get('age', 35))
                features['geo_distance'] = float(request.form.get('geo_distance', 0))
                features['is_far_merchant'] = int(features['geo_distance'] > 2)
            elif model_key == 'mobile_transfer':
                old_balance = float(request.form.get('old_balance', 0))
                new_balance = float(request.form.get('new_balance', 0))
                features['balance_change'] = abs(new_balance - old_balance)

            df_processed = preprocess_universal(features, model_key)
            probability = float(specialist['model'].predict_proba(df_processed)[0][1])
            prediction = 1 if probability >= threshold else 0
            risk_level = get_risk_level(probability)

            base_prob, contributions = explain_prediction(features, model_key)
            for c in contributions:
                c['label'] = FEATURE_LABELS.get(c['feature'], c['feature'])

            result = {
                'label': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
                'probability': round(probability * 100, 2),
                'risk_level': risk_level,
                'model_used': specialist['name'],
                'contributions': contributions,
                'amount': amount,
                'transaction_type': transaction_type
            }

        except Exception as e:
            flash(f'Error generating explanation: {str(e)}', 'danger')

    return render_template('explainability.html', result=result)


@app.route('/api/v1/predict', methods=['POST'])
@require_api_key(scope='predict')
def api_v1_predict():
    try:
        data = request.get_json(force=True)
        transaction_type = data.get('transaction_type', 'card_purchase')
        amount = float(data.get('amount', 0))
        hour = int(data.get('hour', 12))
        model_key, specialist = get_model_for_transaction(transaction_type)
        threshold = specialist['config']['threshold']

        features = {
            'amount': amount,
            'hour': hour,
            'is_night': int(0 <= hour <= 5),
            'is_weekend': int(data.get('is_weekend', 0)),
            'merchant_category': int(data.get('merchant_category', 2)),
            'is_international': int(data.get('is_international', 0)),
            'is_new_merchant': int(data.get('is_new_merchant', 0)),
            'is_online': int(transaction_type in [
                'ecommerce', 'digital_payment', 'subscription',
                'online_transfer', 'card_not_present', 'online_shopping'
            ]),
        }
        if model_key == 'credit_card':
            features['gender'] = int(data.get('gender', 0))
            features['age'] = int(data.get('age', 35))
            features['geo_distance'] = float(data.get('geo_distance', 0))
            features['is_far_merchant'] = int(features['geo_distance'] > 2)
        elif model_key == 'mobile_transfer':
            old_balance = float(data.get('old_balance', 0))
            new_balance = float(data.get('new_balance', 0))
            features['balance_change'] = abs(new_balance - old_balance)

        df_processed = preprocess_universal(features, model_key)
        probability = float(specialist['model'].predict_proba(df_processed)[0][1])
        prediction = 1 if probability >= threshold else 0
        risk_level = get_risk_level(probability)
        label = 'FRAUD' if prediction == 1 else 'LEGITIMATE'

        is_test = request.api_key.key_type == 'test'
        if not is_test:
            db.session.add(Prediction(
                user_id=request.api_key.user_id, amount=amount, transaction_type=transaction_type,
                model_used=model_key, predicted_label=prediction,
                fraud_probability=round(probability, 4), risk_level=risk_level
            ))

        db.session.add(ApiRequestLog(api_key_id=request.api_key.id, endpoint='/api/v1/predict', result=label, probability=round(probability, 4)))
        db.session.commit()

        if not is_test and prediction == 1:
            trigger_webhooks(request.api_key.user_id, 'fraud.detected', {
                'event': 'fraud.detected', 'amount': amount, 'transaction_type': transaction_type,
                'probability': round(probability, 4), 'risk_level': risk_level,
                'model_used': specialist['name'], 'timestamp': datetime.utcnow().isoformat()
            })

        return jsonify({'label': label, 'probability': round(probability, 4), 'risk_level': risk_level, 'model_used': specialist['name'], 'test_mode': is_test})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/v1/batch', methods=['POST'])
@require_api_key(scope='batch')
def api_v1_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Send as multipart/form-data with key "file".'}), 400

        file = request.files['file']
        transaction_type = request.form.get('transaction_type', 'card_purchase')
        df_input = pd.read_csv(file)
        df_input.columns = [c.lower() for c in df_input.columns]

        if 'amount' not in df_input.columns:
            return jsonify({'error': "Missing required column: 'amount'"}), 400

        model_key, specialist = get_model_for_transaction(transaction_type)
        threshold = specialist['config']['threshold']
        feature_list = specialist['config']['features']
        scale_cols = specialist['config']['scale_cols']

        defaults = {
            'amount': 0, 'amount_log': 0, 'is_high_amount': 0, 'is_micro': 0,
            'hour': 12, 'is_night': 0, 'is_weekend': 0, 'merchant_category': 2,
            'is_international': 0, 'is_new_merchant': 0, 'balance_change': 0,
            'geo_distance': 0, 'is_far_merchant': 0, 'is_online': 0,
            'gender': 0, 'age': 35, 'city_pop_log': 0
        }

        df_features = pd.DataFrame()
        for feat in feature_list:
            if feat in df_input.columns:
                col = pd.to_numeric(df_input[feat], errors='coerce')
                df_features[feat] = col.fillna(defaults.get(feat, 0))
            else:
                df_features[feat] = defaults.get(feat, 0)

        df_features['amount_log'] = np.log1p(df_features['amount'])
        df_features['is_high_amount'] = (df_features['amount'] > 1000).astype(int)
        df_features['is_micro'] = (df_features['amount'] < 1).astype(int)

        scaler = specialist['scaler']
        cols_to_scale = [c for c in scale_cols if c in df_features.columns]
        df_features[cols_to_scale] = scaler.transform(df_features[cols_to_scale])

        probabilities = specialist['model'].predict_proba(df_features[feature_list])[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        is_test = request.api_key.key_type == 'test'
        if not is_test:
            for i in range(len(df_input)):
                db.session.add(Prediction(
                    user_id=request.api_key.user_id, amount=float(df_input.iloc[i].get('amount', 0)),
                    transaction_type=transaction_type, model_used=model_key,
                    predicted_label=int(predictions[i]), fraud_probability=float(probabilities[i]),
                    risk_level=get_risk_level(probabilities[i]), is_batch=True
                ))

        db.session.add(ApiRequestLog(api_key_id=request.api_key.id, endpoint='/api/v1/batch'))
        db.session.commit()

        if not is_test and int(predictions.sum()) > 0:
            trigger_webhooks(request.api_key.user_id, 'batch.completed', {
                'event': 'batch.completed', 'total': len(df_input),
                'fraud_count': int(predictions.sum()), 'fraud_rate': round(predictions.mean() * 100, 2),
                'timestamp': datetime.utcnow().isoformat()
            })

        results = [{'row': i + 1, 'fraud_probability': round(float(probabilities[i]), 4),
                     'risk_level': get_risk_level(probabilities[i]),
                     'result': 'FRAUD' if predictions[i] == 1 else 'LEGITIMATE'} for i in range(len(df_input))]

        return jsonify({'total': len(df_input), 'fraud': int(predictions.sum()), 'legitimate': int((predictions == 0).sum()), 'test_mode': is_test, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api-keys')
@login_required
def api_keys():
    keys = ApiKey.query.filter_by(user_id=current_user.id).order_by(ApiKey.created_at.desc()).all()
    new_key = session.pop('new_api_key', None)
    logs_by_key = {k.id: ApiRequestLog.query.filter_by(api_key_id=k.id).order_by(ApiRequestLog.timestamp.desc()).limit(5).all() for k in keys}
    return render_template('api_keys.html', keys=keys, new_key=new_key, logs_by_key=logs_by_key)


@app.route('/api-keys/create', methods=['POST'])
@login_required
def create_api_key():
    name = request.form.get('name', '').strip() or 'Unnamed Key'
    key_type = request.form.get('key_type', 'live')
    scope_predict = bool(request.form.get('scope_predict'))
    scope_batch = bool(request.form.get('scope_batch'))

    prefix = 'pq_test_' if key_type == 'test' else 'pq_live_'
    raw_key = prefix + secrets.token_urlsafe(32)

    new_key = ApiKey(
        user_id=current_user.id, name=name,
        key_prefix=raw_key[:14] + '••••', key_hash=generate_password_hash(raw_key),
        key_type=key_type, scope_predict=scope_predict, scope_batch=scope_batch
    )
    db.session.add(new_key)
    db.session.commit()
    log_action(current_user.id, 'API Key Created', f'"{name}" ({key_type}, scopes: {"Predict " if scope_predict else ""}{"Batch" if scope_batch else ""})')

    session['new_api_key'] = raw_key
    return redirect(url_for('api_keys') + '#akNewKeyBanner')


@app.route('/api/keys/<int:key_id>/logs')
@login_required
def api_key_logs(key_id):
    key = ApiKey.query.get_or_404(key_id)
    if key.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    logs = ApiRequestLog.query.filter_by(api_key_id=key_id).order_by(ApiRequestLog.timestamp.desc()).limit(5).all()
    return jsonify([{
        'endpoint': l.endpoint,
        'result': l.result,
        'timestamp': l.timestamp.strftime('%d %b, %H:%M')
    } for l in logs])


@app.route('/api-keys/<int:key_id>/revoke', methods=['POST'])
@login_required
def revoke_api_key(key_id):
    key = ApiKey.query.get_or_404(key_id)
    if key.user_id != current_user.id:
        return redirect(url_for('api_keys') + '#akManage')
    key.is_active = False
    db.session.commit()
    log_action(current_user.id, 'API Key Revoked', f'"{key.name}" ({key.key_type})')
    return redirect(url_for('api_keys') + '#akManage')


@app.route('/webhooks')
@login_required
def webhooks():
    hooks = Webhook.query.filter_by(user_id=current_user.id).order_by(Webhook.created_at.desc()).all()
    new_secret = session.pop('new_webhook_secret', None)
    return render_template('webhooks.html', hooks=hooks, new_secret=new_secret)


@app.route('/webhooks/create', methods=['POST'])
@login_required
def create_webhook():
    name = request.form.get('name', '').strip() or 'Unnamed Webhook'
    url = request.form.get('url', '').strip()
    event_fraud = bool(request.form.get('event_fraud'))
    event_batch = bool(request.form.get('event_batch'))

    secret = 'whsec_' + secrets.token_urlsafe(24)

    hook = Webhook(
        user_id=current_user.id, name=name, url=url, secret=secret,
        event_fraud=event_fraud, event_batch=event_batch
    )
    db.session.add(hook)
    db.session.commit()
    log_action(current_user.id, 'Webhook Created', f'"{name}" → {url}')

    session['new_webhook_secret'] = secret
    return redirect(url_for('webhooks') + '#whNewSecretBanner')


@app.route('/webhooks/<int:webhook_id>/toggle', methods=['POST'])
@login_required
def toggle_webhook(webhook_id):
    hook = Webhook.query.get_or_404(webhook_id)
    if hook.user_id != current_user.id:
        return redirect(url_for('webhooks') + '#whManage')
    hook.is_active = not hook.is_active
    db.session.commit()
    return redirect(url_for('webhooks') + '#whManage')


@app.route('/webhooks/<int:webhook_id>/delete', methods=['POST'])
@login_required
def delete_webhook(webhook_id):
    hook = Webhook.query.get_or_404(webhook_id)
    if hook.user_id == current_user.id:
        log_action(current_user.id, 'Webhook Deleted', f'"{hook.name}" ({hook.url})')
        WebhookDelivery.query.filter_by(webhook_id=hook.id).delete()
        db.session.delete(hook)
        db.session.commit()
    return redirect(url_for('webhooks') + '#whManage')


@app.route('/api/webhooks/<int:webhook_id>/test', methods=['POST'])
@login_required
def test_webhook(webhook_id):
    wh = Webhook.query.get_or_404(webhook_id)
    if wh.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403

    payload = {
        'event': 'fraud.detected', 'test': True, 'amount': 1250.00,
        'transaction_type': 'card_purchase', 'probability': 0.91,
        'risk_level': 'HIGH', 'model_used': 'Credit Card Fraud Model',
        'timestamp': datetime.utcnow().isoformat()
    }
    body = json.dumps(payload)
    signature = hmac.new(wh.secret.encode(), body.encode(), hashlib.sha256).hexdigest()

    success, status_code, error_msg = False, None, None
    try:
        response = http_requests.post(wh.url, data=body, headers={
            'Content-Type': 'application/json',
            'X-PrediQt-Signature': signature,
            'X-PrediQt-Event': 'fraud.detected'
        }, timeout=3)
        status_code = response.status_code
        success = 200 <= response.status_code < 300
    except Exception as e:
        error_msg = str(e)

    db.session.add(WebhookDelivery(webhook_id=wh.id, event_type='test', payload=body, status_code=status_code, success=success))
    db.session.commit()

    return jsonify({'success': success, 'status_code': status_code, 'error': error_msg})


@app.route('/api/webhooks/<int:webhook_id>/deliveries')
@login_required
def webhook_deliveries(webhook_id):
    wh = Webhook.query.get_or_404(webhook_id)
    if wh.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    deliveries = WebhookDelivery.query.filter_by(webhook_id=webhook_id).order_by(WebhookDelivery.timestamp.desc()).limit(5).all()
    return jsonify([{
        'event_type': d.event_type, 'success': d.success,
        'status_code': d.status_code, 'timestamp': d.timestamp.strftime('%d %b, %H:%M')
    } for d in deliveries])


@app.route('/api-docs')
@login_required
def api_docs():
    return render_template('api_docs.html', rate_limit=DAILY_RATE_LIMIT)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', code=404, title='Page Not Found',
                            message="The page you're looking for doesn't exist or may have been moved."), 404


@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    return render_template('error.html', code=500, title='Something Went Wrong',
                            message="An unexpected error occurred on our end. Please try again in a moment."), 500


@app.errorhandler(403)
def forbidden(e):
    return render_template('error.html', code=403, title='Access Denied',
                            message="You don't have permission to view this page."), 403


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        if not User.query.filter_by(email='ahamedfaysal372@gmail.com').first():
            admin = User(
                name                   = 'Admin',
                email                  = 'ahamedfaysal372@gmail.com',
                phone                  = '',
                country                = 'United Kingdom',
                city                   = 'Belfast',
                company                = 'PrediQt',
                company_reg_number     = '',
                industry               = 'Technology',
                company_size           = '1-10',
                annual_revenue         = 'Prefer not to say',
                business_address       = '',
                postcode               = '',
                job_title              = 'System Administrator',
                use_case               = 'Real-time Monitoring',
                monthly_transactions   = '100,000+',
                current_fraud_solution = 'In-house Solution',
                compliance             = 'GDPR',
                referral               = 'Other',
                fraud_analysts         = '1',
                security_question      = '',
                security_answer        = '',
                notifications          = 'email',
                terms_accepted         = True,
                user_role              = 'admin'
            )
            admin.set_password('Admin#123')
            db.session.add(admin)
            db.session.commit()
            print('Admin account created: ahamedfaysal372@gmail.com / Admin#123')

    app.run(debug=True)
