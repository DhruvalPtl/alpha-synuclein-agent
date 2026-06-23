
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN
    
    # Ensemble of XGB and LGBM with PCA and ADASYN
    model = Pipeline([
        ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
        ('adasyn', ADASYN(random_state=42)),
        ('voting', VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)),
                ('lgbm', LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42))
            ],
            voting='soft'
        ))
    ])
    model.fit(X_train, y_train)
    return model
