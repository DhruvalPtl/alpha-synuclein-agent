
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    
    # SMOTE with LGBM
    clf = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42))
    ])
    
    clf.fit(X_train, y_train)
    return clf
