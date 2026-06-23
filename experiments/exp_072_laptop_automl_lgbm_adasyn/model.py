
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import ADASYN
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Using a robust, balanced LightGBM setup
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('adasyn', ADASYN(random_state=42)),
        ('lgbm', LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            boosting_type='gbdt',
            class_weight='balanced',
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model
