
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    # A dense MLP behaves as a basic sequence/global feature learner
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model
