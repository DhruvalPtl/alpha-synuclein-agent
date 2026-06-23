
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    estimators = [
        ('brf', BalancedRandomForestClassifier(n_estimators=500, max_depth=12, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', random_state=42))
    ]
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    )
    clf.fit(X_train, y_train)
    return clf
