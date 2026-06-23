
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    
    # Diverse ensemble stack
    model = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('voting', VotingClassifier(
            estimators=[
                ('et', ExtraTreesClassifier(n_estimators=300, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)),
                ('lgbm', LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42))
            ],
            voting='soft'
        ))
    ])
    model.fit(X_train, y_train)
    return model
