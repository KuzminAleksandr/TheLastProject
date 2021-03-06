import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from tqdm import trange
from time import time


class RandomForestRMSE:
    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = None,
            bagging_fraction: float = None,
            feature_subsample_size: float = None,
            **trees_parameters
    ) -> None:
        """
        n_estimators : int
            The number of trees in the forest. By default 100.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        bagging_fraction : float
            The fraction of dataset used for bagging. If None is 1.0.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        if n_estimators is None:
            self.n_estimators = 100
        self.max_depth = max_depth

        if feature_subsample_size is not None:
            self.feature_subsample_size = feature_subsample_size
        else:
            self.feature_subsample_size = 1 / 3

        if bagging_fraction is not None:
            self.bagging_fraction = bagging_fraction
        else:
            self.bagging_fraction = 1

        self.trees_parameters = trees_parameters

        self.estimators = []

    def fit(self, X, y, X_val=None, y_val=None, trace=True, verbose=0) -> None or tuple:
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
            Return of no results of training stage. True by default.
        verbose : int
            How often print intermediate results. Lower more often. 0 by default.
        """
        val_preds = None
        train_preds = np.zeros_like(y)
        if X_val is not None:
            val_preds = np.zeros_like(y_val)

        train_time = []
        train_metric = []
        val_time = []
        val_metric = []

        ran = np.arange(0, X.shape[1])

        for i in trange(self.n_estimators):
            # Sample subset of features
            rand_gen = np.random.RandomState(i)
            rand_gen.shuffle(ran)
            features = ran[:int(X.shape[1] * self.feature_subsample_size)]
            # Measure training time
            start_time = time()

            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                **self.trees_parameters
            )

            new_idx = np.random.choice(
                np.arange(X.shape[0]),
                size=[int(self.bagging_fraction * X.shape[0])]
            )
            estimator.fit(X[new_idx][:, features], y[new_idx])

            self.estimators.append(estimator)
            train_time.append(time() - start_time)

            train_preds += estimator.predict(X[:, features])
            train_metric.append(self.get_metric(train_preds / (i + 1), y))

            if val_preds is not None:
                # Measure evaluation time
                start_time = time()

                preds = estimator.predict(X_val[:, features])
                val_preds += preds
                val_time.append(time() - start_time)

                rmse_estimator = self.get_metric(val_preds / (i + 1), y_val)
                val_metric.append(rmse_estimator)

            if verbose and i % verbose == 0:
                print(f"?????????? ???????????????????? {i}-?????? ????????????: {train_time[-1]: .3f} c.")
                if val_preds is not None:
                    print(f"?????????? ?????????????????? {i}-?????? ????????????: {val_time[-1]: .3f} c.")
                    print(f"RMSE ???? ?????????????????????????? ?????????????? ?????? ???????????????????? ???? {i + 1} ????????????????: {rmse_estimator: .3f}")
                print("#------------------------------#")

        # Print results of experiments
        self.print_results(val_preds, y_val, train_time)

        # Return training statistics
        if trace:
            return train_metric, train_time, val_metric
        else:
            return


    def predict(self, X) -> np.ndarray:
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        preds = np.zeros(shape=[X.shape[0], 1])
        ran = np.arange(0, X.shape[1])
        for i, estimator in enumerate(self.estimators):
            rand_gen = np.random.RandomState(i)
            rand_gen.shuffle(ran)
            features = ran[:int(X.shape[1] * self.feature_subsample_size)]
            preds += estimator.predict(X[:, features]).reshape(-1, 1) / self.n_estimators

        return preds

    @staticmethod
    def get_metric(y_preds, y_true) -> float:
        """
        Return RMSE.
        """
        return np.mean((y_preds - y_true) ** 2) ** 0.5

    def print_results(self, val_preds, y_val, train_time) -> None:
        """
        Print training results
        """
        print(f"??????????: Random Forest")
        print(f"??????????????????:")
        print(f"|-> ?????????? ????????????????: {self.n_estimators}")
        print(f"|-> ????????. ?????????????? ????????????: {self.max_depth}")
        print(f"|-> ?????????????????????? ???????????????????????? ??????????????????: {self.feature_subsample_size: .3f}")
        print(f"|-> ???????? ???????????????? ?????????????? ?? ???????????? ????????????????????: {self.bagging_fraction: .3f}")
        print(f"?????????? ???????????????????? ????????????????: {np.sum(train_time): .2f} c.")
        if val_preds is not None:
            val_preds /= self.n_estimators
            print(f"RMSE ???????????????? ???? ??????????????????: {self.get_metric(val_preds, y_val): .3f}")


class GradientBoostingRMSE:
    def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: int = 5,
            feature_subsample_size: float = None,
            **trees_parameters
    ) -> None:
        """
        n_estimators : int
            The number of trees in the forest. By default 100.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        if learning_rate is None:
            self.learning_rate = 0.1

        self.n_estimators = n_estimators
        if n_estimators is None:
            self.n_estimators = 100
        self.trees_parameters = trees_parameters
        if feature_subsample_size is not None:
            self.feature_subsample_size = feature_subsample_size
        else:
            self.feature_subsample_size = 1 / 3

        self.estimators = []
        self.alphas = []

    def fit(self, X, y, X_val=None, y_val=None, trace=True, verbose=0) -> None or tuple:
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
            Return of no results of training stage. True by default.
        verbose : int
            How often print intermediate results. Lower more often. 0 by default.
        """
        current_preds_train = np.zeros_like(y)

        metric_train = []
        train_time = []

        current_preds_val = None
        metric_val = []
        if y_val is not None:
            current_preds_val = np.zeros_like(y_val)

        ran = np.arange(0, X.shape[1])

        for i in trange(self.n_estimators):
            # Sample subset of features
            rand_gen = np.random.RandomState(i)
            rand_gen.shuffle(ran)
            features = ran[:int(X.shape[1] * self.feature_subsample_size)]
            # Measure training time
            start_time = time()

            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                **self.trees_parameters
            )

            # Approximate anti-gradient
            current_antigradient = self.get_antigradient(current_preds_train, y)
            estimator.fit(X[:, features], current_antigradient)
            preds = estimator.predict(X[:, features])

            # Find alpha
            minimizer = minimize_scalar(
                self.alpha_optimize,
                method="bounded",
                bounds=[0, 10],
                args=(current_preds_train, self.learning_rate, preds, y)
            )
            alpha = minimizer.x

            train_time.append(time() - start_time)

            # Save model and alpha
            self.estimators.append(estimator)
            self.alphas.append(alpha * self.learning_rate)

            # Save metric stats
            current_preds_train += alpha * self.learning_rate * preds
            metric_train.append(self.get_metric(current_preds_train, y))

            if current_preds_val is not None:
                current_preds_val += alpha * self.learning_rate * estimator.predict(X_val[:, features])
                metric_val.append(self.get_metric(current_preds_val, y_val))

            if verbose and i % verbose == 0:
                print(f"?????????? ???????????????????? {i}-?????? ????????????: {train_time[-1]: .3f}")
                print(f"RMSE ???? ?????????????????????????? ?????????????? ?????? ???????????????? ???? {i+1} ????????????: {metric_train[-1]: .3f}")
                if current_preds_val is not None:
                    print(f"RMSE ???? ?????????????????????????? ?????????????? ?????? ???????????????? ???? {i + 1} ????????????: {metric_val[-1]: .3f}")
                print("#------------------------------------------------------#")

        # Print results of experiments
        self.print_results(metric_train[-1], train_time, current_preds_val, y_val)

        if trace:
            return metric_train, train_time, metric_val
        else:
            return

    def predict(self, X) -> np.ndarray:
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        prediction = np.zeros(shape=[X.shape[0], 1])
        ran = np.arange(0, X.shape[1])
        for i, (estimator, alpha) in enumerate(zip(self.estimators, self.alphas)):
            rand_gen = np.random.RandomState(i)
            rand_gen.shuffle(ran)
            features = ran[:int(X.shape[1] * self.feature_subsample_size)]
            prediction += alpha * estimator.predict(X[:, features]).reshape(-1, 1)

        return prediction

    @staticmethod
    def get_antigradient(y_preds, y_true) -> np.ndarray:
        """
        Return anti-gradient for MSE.
        """
        return y_true - y_preds

    @staticmethod
    def alpha_optimize(
            alpha,
            prev_preds,
            learning_rate,
            cur_pred,
            y_true
    ) -> float:
        """
        Function for one-dimension optimization of alpha
        alpha : float
            Scalar for optimization.
        """
        upd_pred = prev_preds + alpha * cur_pred
        return np.mean((upd_pred - y_true) ** 2)

    @staticmethod
    def get_metric(y_preds, y_true) -> float:
        """
        Return RMSE.
        """
        return np.mean((y_preds - y_true) ** 2) ** 0.5

    def print_results(self, train_metric,  train_time, val_preds, y_val) -> None:
        """
        Print training results
        """
        print(f"??????????: Gradient Boosting")
        print(f"??????????????????:")
        print(f"|-> ?????????? ????????????????: {self.n_estimators}")
        print(f"|-> ????????. ?????????????? ????????????: {self.max_depth}")
        print(f"|-> ?????????????????????? ???????????????????????? ??????????????????: {self.feature_subsample_size: .3f}")
        print(f"|-> Learning rate: {self.learning_rate}")
        print(f"?????????? ???????????????????? ????????????????: {np.sum(train_time): .2f} c.")
        print(f"RMSE ???????????????? ???? ????????????????????: {train_metric: .3f}")
        if val_preds is not None:
            print(f"RMSE ???????????????? ???? ??????????????????: {self.get_metric(val_preds, y_val): .3f}")
