
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Using a pipeline with ADASYN for imbalance
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('adasyn', ADASYN(sampling_strategy='minority', random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            objective='multi:softprob',
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
