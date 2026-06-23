
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    
    # Refined ensemble: BRF + XGBoost with SMOTE
    model = Pipeline([
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('voting', VotingClassifier(
            estimators=[
                ('brf', BalancedRandomForestClassifier(n_estimators=500, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=42))
            ],
            voting='soft'
        ))
    ])
    model.fit(X_train, y_train)
    return model
