
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from sklearn.ensemble import VotingClassifier
    
    # Create an ensemble of XGBoost classifiers with slightly different parameters
    clf1 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    clf2 = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=43)
    clf3 = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=44)
    
    ensemble = VotingClassifier(estimators=[
        ('xgb1', clf1), ('xgb2', clf2), ('xgb3', clf3)
    ], voting='soft')
    
    ensemble.fit(X_train, y_train)
    return ensemble
