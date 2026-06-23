
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from imblearn.over_sampling import ADASYN
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # XGBoost with ADASYN
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('adasyn', ADASYN(random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])
    model.fit(X_train, y_train)
    return model
