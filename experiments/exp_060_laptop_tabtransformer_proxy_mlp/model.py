
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    
    # Simple MLP as a proxy for attention-based tabular architectures 
    # to maintain complexity control given small N=390
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            activation='relu', 
            solver='adam', 
            max_iter=500, 
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model
