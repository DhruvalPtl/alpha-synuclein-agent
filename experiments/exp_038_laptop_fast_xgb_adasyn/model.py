
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import ADASYN
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline

    # Using XGBoost which is faster than scikit-learn's GradientBoosting
    pipe = Pipeline([
        ('adasyn', ADASYN(sampling_strategy='auto', random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1))
    ])
    
    pipe.fit(X_train, y_train)
    return pipe
