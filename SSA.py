# This file will contain the multivariate SSA algorithm 
from typing import Literal
import numpy as np
from scipy.linalg import hankel

class SSA:
    def __init__(self, L: int | str = "auto", r : int | float | str = "auto", type = "HSSA"):
        self.L = L
        self.r = r
        assert type in ["HSSA", "VSSA"], "type must be either HSSA or VSSA"
        self.type = type
    def fit(self, data):
        """
            Fit the SSA model to the given data.

            This function takes in the data, computes the principal components, and assigns them to `self.U`. It also computes the associated eigenvalues and assigns them to `self.weights`.

            Parameters
            ----------
            data : np.ndarray
                The input data. It should be a 2D array where the first dimension is the number of features and the second dimension is the number of samples.

            Returns
            -------
            None

            Notes
            -----
            The function modifies the object by setting the `self.U` and `self.weights` attributes. `self.U` contains the principal components and `self.weights` contains the associated eigenvalues.

            Raises
            ------
            AssertionError
                If the input data is not a 2D array or if the number of samples is less than the number of features.
        """
        
        # assuming data is of shape (n_features, n_samples)
        assert len(data.shape) == 2, "data must be 2D"
        assert data.shape[1] > data.shape[0], "data must have more samples than features"
        N = data.shape[1]
        M = data.shape[0]
        self.M = M
        if self.L == "auto":
            # do some transformation to get the L value
            if self.type == "HSSA":
                self.L = int(M * (N+1)/(M+1))
            else:
                self.L = int((N+1)/(M+1))
        # constructing the trajectory matrix for each dimension
        X = [self.get_trajectory(data[i], self.L) for i in range(M)]
        ranks = [np.linalg.matrix_rank(X[i]) for i in range(M)]
        if self.type == "HSSA":
            # we first horizontally stack the data
            # then we do the trajectory matrix
            X = np.hstack(X)
        else: 
            # we first vertically stack the data
            # then we do the trajectory matrix
            X = np.vstack(X)
        lag_cov = X @ X.T
        eigvals, eigvecs = np.linalg.eigh(lag_cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        eigvecs = np.real(eigvecs)
        if type(self.r) == float:
            cumsum = np.cumsum(eigvals)
            self.r = np.argmax(cumsum >= self.r) + 1
        elif self.r == "auto":
            self.r = np.max(ranks)
        eigvals, eigvecs = eigvals[:self.r], eigvecs[:, :self.r]
        self.weights = eigvals
        self.U = eigvecs
    
    def transform(self, data):
        """
            Transform the given data using the SSA model.

            This function takes in the data and transforms it using the SSA model. It first computes the trajectory matrix and then projects it onto the principal components.

            Parameters
            ----------
            data : np.ndarray
                The input data. It should be a 2D array where the first dimension is the number of features and the second dimension is the number of samples.

            Returns
            -------
            np.ndarray
                The transformed data. It is a 2D array where the first dimension is the number of features and the second dimension is the number of samples.

            Notes
            -----
            The function does not modify the object.

            Raises
            ------
            AssertionError
                If the input data is not a 2D array or if the number of samples is less than the number of features.
        """
        X = [self.get_trajectory(data[i], self.L) for i in range(len(data))]
        if self.type == "HSSA":
            # we first horizontally stack the data
            # then we do the trajectory matrix
            X = np.hstack(X)
        else: 
            # we first vertically stack the data
            # then we do the trajectory matrix
            X = np.vstack(X)
        temp = self.hankelization(self.U @ self.U.T @ X)
        if self.type == "HSSA":
            temp = np.hsplit(temp, self.M)
        else:
            temp = np.vsplit(temp, self.M)
        return np.array([np.concatenate([temp[i][0,:-1], temp[i][:, -1]]) for i in range(len(temp))])
        
    def fit_transform(self, data):
        """
            Fit the SSA model to the given data and then transform it.

            This function takes in the data, computes the principal components, and assigns them to `self.U`. It also computes the associated eigenvalues and assigns them to `self.weights`. It then transforms the data using the SSA model.

            Parameters
            ----------
            data : np.ndarray
                The input data. It should be a 2D array where the first dimension is the number of features and the second dimension is the number of samples.

            Returns
            -------
            np.ndarray
                The transformed data. It is a 2D array where the first dimension is the number of features and the second dimension is the number of samples.

            Notes
            -----
            The function modifies the object by setting the `self.U` and `self.weights` attributes. `self.U` contains the principal components and `self.weights` contains the associated eigenvalues.

            Raises
            ------
            AssertionError
                If the input data is not a 2D array or if the number of samples is less than the number of features.
        """
        self.fit(data)
        return self.transform(data)
    def get_trajectory(self, data, L):
        # assuming data is of shape (n_samples,)
        assert len(data.shape) == 1, "data must be 1D"
        assert data.shape[0] > L, "data must have more samples than L"
        N = data.shape[0]
        # constructing the trajectory matrix
        X = np.zeros((N-L+1, L))
        for i in range(N-L+1):
            X[i] = data[i:i+L]
        return X.T
    def hankelization(self, X):
        if self.type == "HSSA":
            X = np.hsplit(X, self.M)
        else:
            X = np.vsplit(X, self.M)
        temp = np.zeros((self.M, X[0].shape[0] + X[0].shape[1] - 1))
        for i in range(len(X)):
            for j in range(X[i].shape[0]):
                for k in range(X[i].shape[1]):
                    temp[i, j+k] += X[i][j, k]
        for j in range(temp.shape[1]):
            temp[:,j] /= min(j+1, temp.shape[1]-j)
        temp = [hankel(temp[i, :self.L], temp[i, self.L-1:]) for i in range(self.M)]
        return np.hstack(temp) if self.type == "HSSA" else np.vstack(temp)


            
    def predict(point):
        pass
    def score(data):
        pass
