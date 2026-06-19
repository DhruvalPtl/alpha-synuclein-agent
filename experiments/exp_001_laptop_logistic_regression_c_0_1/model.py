
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        C=0.1,
        class_weight=class_weights,
        max_iter=2000,
        solver='lbfgs',
        multi_class='auto',
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
