
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE, ADASYN
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import VotingClassifier

    # Models with tuned hyperparameters to reduce overfitting
    xgb1 = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    xgb2 = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    
    pipe1 = Pipeline([('smote', SMOTE(random_state=42)), ('xgb', xgb1)])
    pipe2 = Pipeline([('adasyn', ADASYN(random_state=42)), ('xgb', xgb2)])
    
    ensemble = VotingClassifier([('smote_xgb', pipe1), ('adasyn_xgb', pipe2)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble
