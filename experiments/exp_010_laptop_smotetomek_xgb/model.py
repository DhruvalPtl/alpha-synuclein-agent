
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    
    # SMOTETomek with XGBoost
    clf = Pipeline([
        ('smotetomek', SMOTETomek(random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    clf.fit(X_train, y_train)
    return clf
