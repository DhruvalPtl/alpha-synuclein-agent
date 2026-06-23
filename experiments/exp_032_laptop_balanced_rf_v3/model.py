
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    # Using BalancedRandomForest to directly handle the imbalance mentioned in the leaderboard
    model = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        sampling_strategy='all',
        replacement=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
