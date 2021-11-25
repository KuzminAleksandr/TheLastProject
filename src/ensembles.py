import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from tqdm import trange
from time import time


class RandomForestRMSE:
    def __init__(
            self,
            n_estimators: int,
            max_depth: int = None,
            bagging_fraction: float = None,
            feature_subsample_size: float = None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        bagging_fraction : float
            The fraction of dataset used for bagging. If None is 0.66.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        if feature_subsample_size is not None:
            self.feature_subsample_size = feature_subsample_size
        else:
            self.feature_subsample_size = 1 / 3

        if bagging_fraction is not None:
            self.bagging_fraction = bagging_fraction
        else:
            self.bagging_fraction = 0.66

        self.trees_parameters = trees_parameters

        self.estimators = []

    def fit(self, X, y, X_val=None, y_val=None, trace=False, verbose=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        trace : bool
            Return of no results of training stage. False by default.
        verbose : bool
            Whether to print or no intermediate results. False by default.
        """
        val_preds = None
        if X_val is not None:
            val_preds = np.zeros_like(y_val)

        train_time = []
        val_time = []
        val_metric = []

        for i in trange(self.n_estimators):
            start_time = time()
            ###------Measure training time------###
            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.feature_subsample_size,
                **self.trees_parameters
            )

            new_idx = np.random.choice(
                np.arange(X.shape[0]),
                size=[int(self.bagging_fraction * X.shape[0])]
            )
            estimator.fit(X[new_idx], y[new_idx])
            ###---------------------------------###
            train_time.append(time() - start_time)
            if verbose:
                print(f"Время тренировки {i}-ого дерева: {train_time[-1]: .3f} c.")
            ###------Measure evaluation time------###
            if X_val is not None:
                start_time = time()
                preds = estimator.predict(X_val)
                val_preds += preds
                val_time.append(time() - start_time)
                rmse_estimator = self.get_metric(preds, y_val)
                val_metric.append(rmse_estimator)
                if verbose:
                    print(f"Время валидации {i}-ого дерева: {val_time[-1]: .3f} c.")
                    print(f"RMSE на валидационной выборке для {i}-ого дерева: {rmse_estimator: .3f}")
                    print("#------------------------------#")
            ###---------------------------------- ###
            self.estimators.append(estimator)

        ###------Print results of experiments------###
        self.print_results(val_preds, y_val, train_time)

        ###------Return statistics------###
        if trace:
            return val_preds, rmse_estimator, train_time
        else:
            return

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        preds = np.zeros(size=[X.shape[0], 1])
        for estimator in self.estimators:
            preds += estimator.predict(X).reshape(-1, 1) / self.n_estimators

        return preds

    @staticmethod
    def get_metric(y_preds, y_true):
        return np.mean((y_preds - y_true) ** 2) ** 0.5

    def print_results(self, val_preds, y_val, train_time):
        print(f"Метод: Bagging")
        print(f"Параметры:")
        print(f"|-> Число деревьев: {self.n_estimators}")
        print(f"|-> Макс. глубина дерева: {self.max_depth}")
        print(f"|-> Размерность пространства признаков: {self.feature_subsample_size: .3f}")
        print(f"|-> Доля объектов выборки в каждой подвыборке: {self.bagging_fraction: .3f}")
        print(f"Время тренировки ансамбля: {np.sum(train_time): .2f} c.")
        if val_preds is not None:
            val_preds /= self.n_estimators
            print(f"RMSE ансамбля на валидации: {self.get_metric(val_preds, y_val): .3f}")


class GradientBoostingRMSE:
    def __init__(
            self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        pass

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        pass

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pass
