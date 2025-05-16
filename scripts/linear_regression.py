# This file contains the LinearRegression class and helper functions 
# to compute influence scores, leverage scores, and other quantities.

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def compute_scores(x, y, lr):
    '''
    x: design matrix.
    y: response vector.
    lr: linear regression model.
    '''
    if_scores = -lr.influence_scores()[1]
    newton_scores = -lr.one_step_newton()[1]

    # sort indices in ascending order.
    if_inds = np.argsort(if_scores)
    newton_inds = np.argsort(newton_scores)

    # sort scores according to indices.
    sorted_scores = if_scores[if_inds]
    sorted_newton_scores = newton_scores[newton_inds]

    # compute residuals and leverages.
    residuals = lr.residual()
    leverages = lr.leverage_scores()

    x1 = [pt[1] for pt in x]

    # create a df with residuals, leverages, and coordinates.
    orig_df = pd.DataFrame({'x': x1, 'y': y, 
                            'residual': residuals, 'leverage': leverages, 
                            'IF': if_scores, '1Exact': newton_scores,
                            'sorted_IF_indices': if_inds, 'sorted_1Exact_indices': newton_inds,
                            'sorted_IF_scores': sorted_scores, 'sorted_1Exact_scores': sorted_newton_scores})
    return orig_df

def create_orig_df(x, y, lr):
    '''
    x: design matrix.
    y: response vector.
    lr: linear regression object.
    '''
    # compute IF/1Exact scores.
    if_scores = -lr.influence_scores()[1]
    newton_scores = -lr.one_step_newton()[1]

    # sort indices in ascending order.
    if_inds = np.argsort(if_scores)
    newton_inds = np.argsort(newton_scores)

    # residuals
    residuals = lr.residual()
    # leverages
    leverages = lr.leverage_scores()
    # x's
    x1 = [pt[1] for pt in x]

    # create a df with residuals, leverages, and coordinates.
    orig_df = pd.DataFrame({'x1': x1, 'y': y, 
                                      'residual': residuals, 'leverage': leverages, 
                            'influence': if_scores, 'newton': newton_scores})
    
    # print(orig_df[:5])
    return orig_df, if_inds, if_scores, newton_inds, newton_scores

def create_plot(orig_df):
    '''
    orig_df: output from the helper function above.
    '''

    # Sort DF by influence score
    sorted_if_df = orig_df.sort_values(by='influence', ascending=True)
    print("Point to Drop: ", sorted_if_df.index[0])
    
    # sorted_if_df: the indices here are the order which amip drops points.
    index = range(0, len(sorted_if_df))
    sorted_if_df['sorted_idx'] = index
    
    # Plot points dropped
    plt.figure(figsize=(9, 7))
    plt.scatter(orig_df[:2]['x1'], orig_df[:2]['y'], marker='x', color='black', label='Pop. A') # pop A
    plt.scatter(orig_df[2:]['x1'], orig_df[2:]['y'], marker='x', color='r', label='Pop. B') # pop B

    # Add dropped order positioned by each point
    for index, row in sorted_if_df.iterrows():
        if row['sorted_idx'] <= 9: # indices 2,3,4 are right on top of one another.
            plt.text(row['x1'], row['y'], int(row['sorted_idx']), ha='left', va='top', fontsize=20)


    plt.xlabel('x1')
    plt.ylabel('y')
    plt.title('Scores Ordered')
    plt.legend()
    plt.show()
    
    return

def Run_Greedy(x, y, orig_if_inds, orig_newton_inds, lr, alphaN, method='IF'):
    '''
    x: design matrix.
    y: response vector.
    orig_inds: indices sorted by the first round.
    lr: linear regression object.
    method: 'IF' or '1Exact'.
    '''
    ctr = 0
    prev_beta = lr.beta[1] # initialize to the original beta estimate.
    dropped_order = []
    exact_changes_beta = []
    beta_estimates_greedy = []
    if_inds = orig_if_inds
    newton_inds = orig_newton_inds

    #for _ in range(3):
    while prev_beta > 0 and ctr < alphaN:
        # print(f'interation {ctr}')

        if method == 'IF':
            inds = if_inds
        else:
            inds = newton_inds
        
        # 1. drop the datapoint with the most negative influence:
        index_to_remove = inds[0]
        # print("index to remove", index_to_remove)

        dropped_order.append(index_to_remove)

        new_x = np.concatenate((x[:index_to_remove], x[index_to_remove + 1:]))
        new_y = np.concatenate((y[:index_to_remove], y[index_to_remove + 1:]))

        x = new_x
        y = new_y

        # 2. calculate the exact perturbation (ie. refit the lr to get the change in the coefficient.)
        lr = LinearRegression(x=x.T, y=y)
        lr.fit()
        # print(f'fitted beta1: {lr.beta[1]}')

        # 3. compute scores and create plot.
        orig_df, if_inds, if_scores, newton_inds, newton_scores = create_orig_df(x, y, lr)
        # create_plot(orig_df)

        # 4. record: the exact change in beta.
        beta_change = lr.beta[1] - prev_beta
        exact_changes_beta.append(beta_change)
        prev_beta = lr.beta[1]
        beta_estimates_greedy.append(lr.beta[1])

        # counter
        ctr += 1

    return dropped_order, exact_changes_beta, beta_estimates_greedy


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

    def influence_scores(self) -> NDArray:
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
    
    def one_step_newton(self) -> NDArray:
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
    
    def hat_matrix_algorithm(self, hatMatrix, alphaN) -> Tuple[set, list]:
        """
        Returns points belonging to pairs that have the largest hat matrix elements,
        until alphaN unique points are identified.

        Args:
            hatMatrix: [n, n] Hat matrix of the linear regression.
            alphaN: int, number of unique data points to include.
        Returns:
            Tuple[set, list]: the Most Influential Set and an associated list of leverages/cross-leverages.
        """
        n = hatMatrix.shape[0]
        UR_hatMatrix = []
        # Place all unique absolute value cross-leverage values into a list, along with the associated data point pairs.
        for i in range(n):
            for j in range(i, n):
                UR_hatMatrix.append((i, j, np.abs(hatMatrix[i, j])))
        
        # Sort the list in descending order of leverage and cross-leverage value magnitudes. 
        sorted_UR_hatMatrix = sorted(UR_hatMatrix, key=lambda x: x[2], reverse=True)
        
        cross_leverages = []
        unique_indices = set()
        # Iterate through the sorted cross leverages until alphaN unique data points are included.
        for i, j, value in sorted_UR_hatMatrix:
            if len(unique_indices) >= alphaN:
                break
            if i not in unique_indices:
                unique_indices.add(i)
                cross_leverages.append(value)
                if j != i and len(unique_indices) <= alphaN:
                    print(f'off-diagonal value included for {(i, j)}')
            if j not in unique_indices and len(unique_indices) <= alphaN:
                unique_indices.add(j)
                cross_leverages.append(value)
                if j != i and len(unique_indices) <= alphaN:
                    print(f'off-diagonal value included for {(i, j)}')
        
        return unique_indices, cross_leverages

