
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from imblearn.over_sampling import ADASYN
    from imblearn.pipeline import Pipeline
    
    # Using ADASYN to rebalance based on recommendations, with XGBoost tuned
    model = Pipeline([
        ('adasyn', ADASYN(random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=700, 
            learning_rate=0.05, 
            max_depth=5, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            use_label_encoder=False, 
            eval_metric='mlogloss',
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    return model
