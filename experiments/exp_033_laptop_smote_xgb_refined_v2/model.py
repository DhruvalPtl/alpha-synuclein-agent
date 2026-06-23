
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    
    # Refining SMOTE and XGBoost
    pipeline = Pipeline([
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=700, 
            learning_rate=0.01, 
            max_depth=5, 
            subsample=0.7, 
            colsample_bytree=0.7, 
            reg_alpha=0.1, 
            reg_lambda=1.0,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
