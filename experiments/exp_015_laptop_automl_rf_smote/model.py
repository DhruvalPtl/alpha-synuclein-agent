
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    
    # AutoML-like approach: GridSearch for Random Forest with SMOTE
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [3, 5, 10]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro')
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_
