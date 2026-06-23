
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from imblearn.over_sampling import ADASYN
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('adasyn', ADASYN(random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=1,
            gamma=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])
    model.fit(X_train, y_train)
    return model
