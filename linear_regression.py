from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class LinearRegression:
    """
    A (fairly) minimimal class for fitting linear regression via least squares that also implements
    leverage scores and other things. Ideally, we would use a trusted package for this,
    or test this code against a package.
    """

    x: NDArray  # [p, n] Covariates/Inputs for regression
    y: NDArray  # [n, k] Response/Outputs for regression
    beta: Optional[
        NDArray
    ] = None  # [p, k] coefficients, usually this should starts as none and will be assigned when fit

    def fit(self) -> None:
        self.beta, _, _, _ = np.linalg.lstsq(
            self.x.T, self.y, rcond=None
        )  # Use numpy for least squares, it expects x to be [n, p]

    def predict(self, xnew: NDArray) -> NDArray:
        """
        Make predictions at points xnew ([p, m]). Fit the coefficients first if not already fit
        """
        if self.beta is None:
            self.fit()
        return xnew.T @ self.beta
    
    def residual(self) -> NDArray:
        """
        Compute residuals for the fit.
        """
        residuals = self.y - self.predict(self.x)
        return residuals.flatten()

    def influence_scores(self) -> NDArray: # zaminfluence, or compute leverage scores manually, write unit tests.
        """
        Computes {IF(x_n; beta, F_N)}_{n=1}^{N}.
        Returns:
            NDArray:  [P, N]; Influence function of parameters for each datapoint at empirical distribution.
            Note that if we want to estimate the effect of removing the datapoint, we need a negative sign.
        """
        residuals = self.y - self.predict(self.x)  # [N, 1]
        xxt = self.x @ self.x.T
        xxtinvx = np.linalg.solve(xxt, self.x)  # [P, N]
        return xxtinvx * residuals.T  # [P, N]
    
    def one_step_newton(self) -> NDArray: # zaminfluence, or compute leverage scores manually, write unit tests.
        """
        Computes {IF(x_n; beta, F_N)}_{n=1}^{N}.
        Returns:
            NDArray:  [P, N]; Influence function of parameters for each datapoint at empirical distribution.
            Note that if we want to estimate the effect of removing the datapoint, we need a negative sign.
        """
        residuals = self.y - self.predict(self.x)  # [N, 1]
        xxt = self.x @ self.x.T
        xxtinvx = np.linalg.solve(xxt, self.x)  # [P, N]
        ### leverage correction term
        hii = np.sum(self.x * xxtinvx, axis=0)
        ###
        return xxtinvx * residuals.T / (1 - hii)  # [P, N]

    def leverage_scores(self) -> NDArray:
        """
        Leverage score for each datapoint
        Returns:
            NDArray: [N]
        """
        xxt = self.x @ self.x.T
        xxtinvx = np.linalg.solve(xxt, self.x)  # [P, N]
        return np.sum(self.x * xxtinvx, axis=0)
    
    def schmeverage_scores(self, ei: NDArray) -> NDArray:
        """
        ei(xtx)-1xi score for each datapoint
        Arg:
            dim: the direction (e1, e2 etc.)
        Returns:
            NDArray: [N]
        """
        xxt = self.x @ self.x.T
        xxtinvx = np.linalg.solve(xxt, self.x)  # [P, N]
        return np.sum(ei * xxtinvx, axis=0)
    
    def standard_error_ols(self) -> NDArray:
        """
        Compute the standard error of the OLS estimator.
        Returns:
            NDArray: [p] Standard errors for each coefficient.
        """
        if self.beta is None:
            self.fit()
        p, n = self.x.shape
        residuals = self.residual()
        sigma_squared_hat = np.sum(residuals**2) / (n - p)
        print("sigma_squared_hat: ", sigma_squared_hat)
        xtx_inv = np.linalg.inv(self.x @ self.x.T)
        standard_errors = np.sqrt(np.diagonal(sigma_squared_hat * xtx_inv))
        return standard_errors

