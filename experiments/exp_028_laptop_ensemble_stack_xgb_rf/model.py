
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42))
    ]
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('stack', StackingClassifier(estimators=estimators, final_estimator=XGBClassifier()))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
