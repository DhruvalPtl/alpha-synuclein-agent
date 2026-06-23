
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('sampling', ADASYN(n_neighbors=5, random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=1.2,
            n_jobs=-1,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
