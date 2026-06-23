
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    # Using BalancedRandomForestClassifier to naturally handle imbalance
    model = BalancedRandomForestClassifier(
        n_estimators=200,
        sampling_strategy='all',
        replacement=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model
