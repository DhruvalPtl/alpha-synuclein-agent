
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from sklearn.ensemble import VotingClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    
    # Create a soft voting ensemble of three different XGBoost configurations
    xgb1 = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    xgb2 = XGBClassifier(n_estimators=200, max_depth=5, random_state=43)
    xgb3 = XGBClassifier(n_estimators=300, max_depth=4, random_state=44)
    
    ensemble = VotingClassifier(
        estimators=[('xgb1', xgb1), ('xgb2', xgb2), ('xgb3', xgb3)],
        voting='soft'
    )
    
    model = Pipeline([
        ('smote', SMOTE(sampling_strategy='all', random_state=42)),
        ('clf', ensemble)
    ])
    
    model.fit(X_train, y_train)
    return model
