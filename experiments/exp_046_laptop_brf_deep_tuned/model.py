
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    # Using BalancedRandomForest which handles imbalance internally
    brf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        sampling_strategy='all',
        replacement=True,
        random_state=42
    )
    brf.fit(X_train, y_train)
    return brf
