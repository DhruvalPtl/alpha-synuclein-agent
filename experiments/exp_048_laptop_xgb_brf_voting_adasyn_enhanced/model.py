
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN

    # Ensemble: Combining high-depth XGB with BRF
    estimators = [
        ('xgb', Pipeline([
            ('adasyn', ADASYN(random_state=42)),
            ('xgb', XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=8, random_state=42))
        ])),
        ('brf', Pipeline([
            ('adasyn', ADASYN(random_state=42)),
            ('brf', BalancedRandomForestClassifier(n_estimators=500, max_depth=8, random_state=42))
        ]))
    ]
    
    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting.fit(X_train, y_train)
    return voting
