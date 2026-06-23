
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    # Using BalancedRandomForestClassifier directly for intrinsic handling of imbalance
    model = Pipeline([
        ('brf', BalancedRandomForestClassifier(
            n_estimators=500, 
            sampling_strategy='all', 
            replacement=True, 
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    return model
