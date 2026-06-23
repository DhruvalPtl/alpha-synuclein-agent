
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import VotingClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN

    # Voting ensemble: BalancedRF + XGBoost with ADASYN pipeline
    clf1 = BalancedRandomForestClassifier(n_estimators=300, random_state=42)
    clf2 = Pipeline([
        ('adasyn', ADASYN(random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42))
    ])
    
    ensemble = VotingClassifier([('brf', clf1), ('xgb_ada', clf2)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble
