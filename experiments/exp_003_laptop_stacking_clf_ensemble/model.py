
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    # Base estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, class_weight='balanced', random_state=42))
    ]
    
    # Final estimator
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    clf.fit(X_train, y_train)
    return clf
