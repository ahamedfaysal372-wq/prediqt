"""One-time migration: add new columns to the existing api_key table.
Run once with:  python migrate_api_keys.py
"""
from app.app import app, db

with app.app_context():
    with db.engine.connect() as conn:
        conn.execute(db.text('ALTER TABLE api_key ADD COLUMN key_type VARCHAR(10) DEFAULT "live"'))
        conn.execute(db.text('ALTER TABLE api_key ADD COLUMN scope_predict BOOLEAN DEFAULT 1'))
        conn.execute(db.text('ALTER TABLE api_key ADD COLUMN scope_batch BOOLEAN DEFAULT 1'))
        conn.execute(db.text('ALTER TABLE api_key ADD COLUMN request_count INTEGER DEFAULT 0'))
        conn.execute(db.text('ALTER TABLE api_key ADD COLUMN requests_today INTEGER DEFAULT 0'))
        conn.execute(db.text('ALTER TABLE api_key ADD COLUMN rate_limit_date DATE'))
        conn.commit()
    print('Migration complete — api_key table updated.')
