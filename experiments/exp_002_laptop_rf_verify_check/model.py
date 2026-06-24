
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=50,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
