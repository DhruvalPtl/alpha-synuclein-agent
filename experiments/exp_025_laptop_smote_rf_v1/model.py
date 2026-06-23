
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.pipeline import Pipeline
    
    # Using SMOTE for balancing + RF for classification
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
