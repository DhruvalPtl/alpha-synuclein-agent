
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import GradientBoostingClassifier
    from imblearn.pipeline import Pipeline

    pipe = Pipeline([
        ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=700, learning_rate=0.03, max_depth=4, random_state=42))
    ])
    
    pipe.fit(X_train, y_train)
    return pipe
