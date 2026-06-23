
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    # BalancedRandomForest handles imbalance internally
    clf = BalancedRandomForestClassifier(n_estimators=500, random_state=42, replacement=True)
    clf.fit(X_train, y_train)
    return clf
