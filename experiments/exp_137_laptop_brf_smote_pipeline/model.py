
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Using a pipeline with SMOTE for minority classes and BalancedRandomForest for bagging
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('brf', BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            sampling_strategy='all',
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
