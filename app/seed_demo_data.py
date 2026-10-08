import random
from datetime import datetime, timedelta
from app import app, db, User, Prediction, get_risk_level

random.seed()

MODEL_PROFILES = {
    'credit_card': {
        'types': ['card_purchase', 'atm', 'online_shopping', 'contactless', 'grocery', 'entertainment'],
        'fraud_rate': 0.04,
        'amount_range': (5, 800),
        'fraud_amount_range': (400, 4500),
    },
    'online_payment': {
        'types': ['ecommerce', 'digital_payment', 'subscription', 'online_transfer', 'card_not_present'],
        'fraud_rate': 0.06,
        'amount_range': (10, 600),
        'fraud_amount_range': (300, 3000),
    },
    'mobile_transfer': {
        'types': ['bank_transfer', 'mobile_payment', 'cash_withdrawal', 'direct_debit'],
        'fraud_rate': 0.03,
        'amount_range': (5, 1200),
        'fraud_amount_range': (1500, 12000),
    },
}


def random_timestamp(days_back=30):
    days_ago = random.triangular(0, days_back, 0)
    hour = random.choices(
        range(24),
        weights=[2,1,1,1,1,2,4,6,8,9,9,9,9,9,9,9,8,8,7,6,5,4,3,2]
    )[0]
    base = datetime.utcnow() - timedelta(days=days_ago)
    return base.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def generate_prediction(user_id, model_key, profile):
    is_fraud = random.random() < profile['fraud_rate']
    tx_type = random.choice(profile['types'])

    if is_fraud:
        amount = round(random.uniform(*profile['fraud_amount_range']), 2)
        probability = round(random.uniform(0.72, 0.98), 4)
    else:
        amount = round(random.uniform(*profile['amount_range']), 2)
        probability = round(random.uniform(0.01, 0.28), 4)

    risk_level = get_risk_level(probability)
    predicted_label = 1 if is_fraud else 0
    timestamp = random_timestamp()
    is_batch = random.random() < 0.15

    return Prediction(
        user_id=user_id,
        amount=amount,
        transaction_type=tx_type,
        model_used=model_key,
        predicted_label=predicted_label,
        fraud_probability=probability,
        risk_level=risk_level,
        timestamp=timestamp,
        is_batch=is_batch,
        is_reviewed=random.random() < 0.4 if is_fraud else False
    )


def seed(total_records=400):
    with app.app_context():
        all_users = User.query.all()
        if not all_users:
            print('No users found in the database.')
            return

        records = []
        model_keys = list(MODEL_PROFILES.keys())
        weights = [0.45, 0.30, 0.25]

        for _ in range(total_records):
            model_key = random.choices(model_keys, weights=weights)[0]
            profile = MODEL_PROFILES[model_key]
            user = random.choice(all_users)
            records.append(generate_prediction(user.id, model_key, profile))

        db.session.bulk_save_objects(records)
        db.session.commit()

        fraud_count = sum(1 for r in records if r.predicted_label == 1)
        print(f'Seeded {len(records)} predictions ({fraud_count} flagged as fraud) across {len(all_users)} user(s).')


if __name__ == '__main__':
    seed(total_records=400)