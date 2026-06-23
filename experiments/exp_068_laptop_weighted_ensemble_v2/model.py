
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    # Weighted Ensemble of boosting and bagging
    # Boosting to focus on hard instances, bagging to reduce variance
    estimators = [
        ('gbc', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42))
    ]
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('voting', VotingClassifier(estimators=estimators, voting='soft', weights=[0.6, 0.4]))
    ])
    model.fit(X_train, y_train)
    return model
