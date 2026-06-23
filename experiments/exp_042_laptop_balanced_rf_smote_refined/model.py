
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    # Pipeline with SMOTE + BalancedRandomForest
    model = Pipeline([
        ('smote', SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)),
        ('brf', BalancedRandomForestClassifier(
            n_estimators=1000, 
            criterion='gini',
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    return model
