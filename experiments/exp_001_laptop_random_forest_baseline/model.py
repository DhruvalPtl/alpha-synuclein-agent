
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import RandomForestClassifier
    # Convert dict class_weights to a list for the classifier if necessary, 
    # but RF handles 'balanced' string well.
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    return clf
