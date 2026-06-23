
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import ADASYN
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
    ]
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('adasyn', ADASYN(random_state=42)),
        ('stack', StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression(),
            cv=3
        ))
    ])
    model.fit(X_train, y_train)
    return model
