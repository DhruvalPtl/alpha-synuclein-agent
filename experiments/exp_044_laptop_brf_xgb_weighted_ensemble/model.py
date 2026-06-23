
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import VotingClassifier

    # Calculate class weights for XGBoost
    import numpy as np
    cw = [class_weights[i] for i in range(4)]
    
    # Use balanced weights for both learners
    clf1 = BalancedRandomForestClassifier(n_estimators=500, random_state=42)
    clf2 = XGBClassifier(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42,
        class_weight=None # XGBoost uses sample_weight during fit, or scale_pos_weight
    )
    
    # Pass sample weights for XGBoost to handle class imbalance
    sample_weights = np.array([cw[y] for y in y_train])
    
    ensemble = VotingClassifier([('brf', clf1), ('xgb', clf2)], voting='soft')
    ensemble.fit(X_train, y_train, sample_weight=sample_weights) # VotingClassifier supports sample_weight if all underlying estimators do
    return ensemble
