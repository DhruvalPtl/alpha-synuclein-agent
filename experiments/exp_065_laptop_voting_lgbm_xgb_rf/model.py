
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Ensemble of high-performing models
    estimators = [
        ('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ]
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('voting', VotingClassifier(estimators=estimators, voting='soft'))
    ])
    model.fit(X_train, y_train)
    return model
