
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Simple MLP with class weights in mind
    # Since MLP doesn't have class_weight natively, 
    # we can try to compensate by sampling or just basic scaling.
    # Actually, we can use the sample_weight parameter in fit.
    
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])
    
    # Calculate sample weights
    import numpy as np
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    clf.fit(X_train, y_train, mlp__sample_weight=sample_weights)
    return clf
