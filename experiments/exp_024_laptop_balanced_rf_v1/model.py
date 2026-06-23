
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    # BalancedRandomForest handles class imbalance internally
    # and is generally very effective for small tabular datasets
    clf = BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        sampling_strategy='auto',
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf
