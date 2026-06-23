
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.combine import SMOTETomek
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    
    # SMOTETomek to balance and clean the dataset
    pipeline = Pipeline([
        ('smotetomek', SMOTETomek(random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=600, 
            learning_rate=0.05, 
            max_depth=5, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
