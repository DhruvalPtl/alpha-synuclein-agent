
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from imblearn.over_sampling import ADASYN
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier

    # Feature selection + XGBoost with ADASYN
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('adasyn', ADASYN(random_state=42)),
        ('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=42), max_features=20)),
        ('xgb', XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])
    model.fit(X_train, y_train)
    return model
