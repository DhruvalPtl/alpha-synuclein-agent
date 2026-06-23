
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import VotingClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN

    # Ensemble with two robust boosters
    estimators = [
        ('xgb', Pipeline([
            ('adasyn', ADASYN(random_state=42)),
            ('xgb', XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=7, random_state=42))
        ])),
        ('lgbm', Pipeline([
            ('adasyn', ADASYN(random_state=42)),
            ('lgbm', LGBMClassifier(n_estimators=400, learning_rate=0.03, max_depth=7, random_state=42))
        ]))
    ]
    
    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting.fit(X_train, y_train)
    return voting
