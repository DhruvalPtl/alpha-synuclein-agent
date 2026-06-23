
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # SMOTE is often more robust for smaller datasets
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    level0 = [
        ('xgb', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01, n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=300, num_leaves=20, learning_rate=0.01, n_jobs=-1, verbose=-1)),
        ('gbc', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.01))
    ]
    level1 = LogisticRegression()
    clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    clf.fit(X_res, y_res)
    return clf
