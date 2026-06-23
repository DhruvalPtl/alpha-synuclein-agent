
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('sampling', ADASYN(random_state=42, n_neighbors=5)),
        ('clf', XGBClassifier(
            n_estimators=800,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            n_jobs=-1,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
