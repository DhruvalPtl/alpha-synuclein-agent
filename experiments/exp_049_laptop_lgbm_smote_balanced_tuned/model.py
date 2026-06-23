
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from lightgbm import LGBMClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    model = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=9,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    return model
