
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    from imblearn.pipeline import Pipeline
    
    # Using voting with hard voting
    clf1 = RandomForestClassifier(n_estimators=200, random_state=42)
    clf2 = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('voting', VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2)], voting='hard'))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
