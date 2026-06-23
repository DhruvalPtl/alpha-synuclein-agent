
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    # Nystroem approximation acts as a protein embedding projection
    model = Pipeline([
        ('feat', Nystroem(gamma=0.1, n_components=100)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model
