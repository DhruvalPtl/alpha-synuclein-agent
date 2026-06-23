
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=800, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model
