
from xgboost import XGBClassifier
import numpy as np

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Calculate scale_pos_weight: ratio of negative to positive samples
    # Assuming classes 0, 1, 2, 3. 
    # Let's use multiclass objective and set weights via sample_weight during fit
    
    # Calculate sample weights to balance the classes
    unique_classes, counts = np.unique(y_train, return_counts=True)
    weights_dict = {cls: sum(counts)/count for cls, count in zip(unique_classes, counts)}
    sample_weights = np.array([weights_dict[y] for y in y_train])
    
    clf = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.5,
        objective='multi:softprob',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf
