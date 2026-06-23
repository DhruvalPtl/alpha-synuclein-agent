
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=1
        ))
    ])
    model.fit(X_train, y_train)
    return model
