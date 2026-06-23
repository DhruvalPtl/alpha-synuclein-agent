
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Define base models
    estimators = [
        ('lr', LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ]
    
    # Stacking classifier
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42)
    )
    
    # Wrap in pipeline for scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacking', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline
