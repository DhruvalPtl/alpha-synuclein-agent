
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import GradientBoostingClassifier
    from imblearn.pipeline import Pipeline
    
    # GradientBoosting with SMOTE
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
