
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    
    # Refined XGBoost with SMOTE and aggressive hyperparameter tuning
    model = Pipeline([
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    return model
