
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Calculate sample weights to handle imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model
