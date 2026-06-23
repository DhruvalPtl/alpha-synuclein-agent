
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import VotingClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN, SMOTE

    # Branch 1: XGBoost with ADASYN
    clf1 = Pipeline([
        ('adasyn', ADASYN(random_state=42)),
        ('xgb', XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42))
    ])
    
    # Branch 2: LightGBM with SMOTE
    clf2 = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42))
    ])
    
    ensemble = VotingClassifier([('xgb', clf1), ('lgbm', clf2)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble
