
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE, ADASYN
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import VotingClassifier

    # Diverse XGBoost models with different sampling techniques
    pipe1 = Pipeline([('smote', SMOTE(random_state=42)), ('xgb', XGBClassifier(n_estimators=500, max_depth=5, random_state=42))])
    pipe2 = Pipeline([('adasyn', ADASYN(random_state=42)), ('xgb', XGBClassifier(n_estimators=500, max_depth=6, random_state=42))])
    
    ensemble = VotingClassifier([('smote_xgb', pipe1), ('adasyn_xgb', pipe2)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble
