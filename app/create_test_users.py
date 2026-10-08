from app import app, db, User

TEST_USERS = [
    {'name': 'Sarah Chen', 'email': 'ahamedfaysal372+sarah@gmail.com', 'company': 'Chen Retail Group', 'job_title': 'Fraud Analyst'},
    {'name': 'James Okafor', 'email': 'ahamedfaysal372+james@gmail.com', 'company': 'Okafor Payments Ltd', 'job_title': 'Risk Manager'},
    {'name': 'Maria Santos', 'email': 'ahamedfaysal372+maria@gmail.com', 'company': 'Santos Financial', 'job_title': 'Compliance Officer'},
]

with app.app_context():
    for u in TEST_USERS:
        if User.query.filter_by(email=u['email']).first():
            print(f"Skipping {u['email']} — already exists.")
            continue
        user = User(
            name=u['name'], email=u['email'], company=u['company'],
            job_title=u['job_title'], country='United Kingdom',
            terms_accepted=True, user_role='user'
        )
        user.set_password('TestDemo#123')
        db.session.add(user)
        print(f"Created {u['email']}")
    db.session.commit()
    print('Done.')