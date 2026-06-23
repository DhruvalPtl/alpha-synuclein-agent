
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', HistGradientBoostingClassifier(
            loss='log_loss',
            learning_rate=0.05,
            max_iter=1000,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
