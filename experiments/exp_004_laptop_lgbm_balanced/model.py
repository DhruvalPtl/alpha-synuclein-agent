
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from lightgbm import LGBMClassifier
    
    # Using 'balanced' class weights
    clf = LGBMClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42, 
        verbose=-1
    )
    
    clf.fit(X_train, y_train)
    return clf
