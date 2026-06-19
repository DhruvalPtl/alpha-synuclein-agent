
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    model = LinearSVC(
        C=0.1,
        class_weight=class_weights,
        max_iter=5000,
        random_state=42,
    )
    # Wrap for .predict() consistency (LinearSVC always has .predict anyway)
    model.fit(X_train, y_train)
    return model
