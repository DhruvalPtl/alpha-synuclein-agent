
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Logistic Regression with balanced class weights
    # We use a pipeline for scaling as it's critical for linear models
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(
            class_weight='balanced', 
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=1000
        ))
    ])
    
    clf.fit(X_train, y_train)
    return clf
