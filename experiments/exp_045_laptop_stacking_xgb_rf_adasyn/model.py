
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN

    # Stacking structure
    estimators = [
        ('xgb', Pipeline([
            ('adasyn', ADASYN(random_state=42)),
            ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
        ])),
        ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42))
    ]
    
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
    stack.fit(X_train, y_train)
    return stack
