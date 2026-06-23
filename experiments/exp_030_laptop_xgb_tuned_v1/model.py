
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    
    # Tuned XGBoost with SMOTE
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
