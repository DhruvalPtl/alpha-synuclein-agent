
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import VotingClassifier

    # Voting between BalancedRandomForest and BalancedBaggingClassifier with SMOTE
    clf1 = BalancedRandomForestClassifier(n_estimators=400, random_state=42)
    clf2 = BalancedBaggingClassifier(base_estimator=None, n_estimators=400, random_state=42)
    
    ensemble = VotingClassifier([('brf', clf1), ('bbc', clf2)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble
