
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    
    # Simpler, stable ensemble
    model = Pipeline([
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('voting', VotingClassifier(
            estimators=[
                ('et', ExtraTreesClassifier(n_estimators=500, max_depth=10, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.02, random_state=42))
            ],
            voting='soft'
        ))
    ])
    model.fit(X_train, y_train)
    return model
