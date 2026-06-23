
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE, ADASYN
    
    # Building a more robust 3-model ensemble
    model = Pipeline([
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('adasyn', ADASYN(sampling_strategy='minority', random_state=42)),
        ('voting', VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, random_state=42)),
                ('lgbm', LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, random_state=42))
            ],
            voting='soft'
        ))
    ])
    model.fit(X_train, y_train)
    return model
