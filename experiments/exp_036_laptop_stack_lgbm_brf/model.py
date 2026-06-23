
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from lightgbm import LGBMClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    estimators = [
        ('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight='balanced', random_state=42)),
        ('brf', BalancedRandomForestClassifier(n_estimators=300, random_state=42))
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    stack.fit(X_train, y_train)
    return stack
