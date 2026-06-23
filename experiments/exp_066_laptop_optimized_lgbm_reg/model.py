
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Refined LightGBM with higher regularization
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model
