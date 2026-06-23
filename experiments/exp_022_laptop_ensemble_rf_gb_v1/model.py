
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Calculate sample weights for the base learners
    sample_weights = compute_sample_weight(class_weight=class_weights, y=y_train)
    
    # Base models
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    
    # Voting ensemble
    clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    return clf
