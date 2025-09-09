from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
from typing import Dict, Any, Tuple


class ModelTrainer:
    def __init__(self):
        """Initialize ModelTrainer with various regression models."""
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(max_iter=2000, tol=1e-3),
            'lasso': Lasso(max_iter=2000, tol=1e-3),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='linear', max_iter=2000)
        }
        self.best_model = None
        self.best_model_name = None
        
    def cv_train_and_select(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict[str, float]]:
        #train different models with cross validation
        # Split the data
        cv_results = {}
        best_score = float('-inf')
        
        for name, model in self.models.items():
            if name == 'gbr':
                continue
            # perform cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=4)
            cv_avg = cv_scores.mean()
            cv_results[name] = {
                'cv_mean': cv_avg,
            }
            
            # Update best model
            if cv_avg > best_score:
                best_score = cv_avg
                self.best_model = model
                self.best_model_name = name

        self.best_model = self.best_model.fit(X_train, y_train)
        
        return self.best_model_name, self.best_model, cv_results

    def grid_search_train(self, X_train, Y_train, model, param_grid, cv=4, refit=True):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        self.best_model= grid_search.fit(X_train, Y_train)
        return self.best_model


    def get_best_model(self) -> Tuple[str, Any]:

        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        return self.best_model_name, self.best_model
    