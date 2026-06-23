
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Create individual pipelines
    xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
    ])
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42))
    ])

    # Stacking
    estimators = [('xgb', xgb), ('rf', rf)]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3
    )
    
    # We apply SMOTE separately before fitting the stack
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train_res, y_train_res)
    return model
