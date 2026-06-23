
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN
    
    # Trying slightly deeper and more robust XGB ensemble
    model = Pipeline([
        ('adasyn', ADASYN(random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=600,
            learning_rate=0.01,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model
