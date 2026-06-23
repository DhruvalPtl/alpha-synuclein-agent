
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import GradientBoostingClassifier
    from imblearn.over_sampling import ADASYN
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('adasyn', ADASYN(random_state=42)),
        ('gbc', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model
