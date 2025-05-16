## This script contains scripts for post-processing computed scores.
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

def compute_scores(x, y, lr, positivebeta):
    '''
    x: design matrix.
    y: response vector.
    lr: linear regression model.
    positivebeta: boolean, True if fit to the full data is positive.
    '''
    if_scores = -lr.influence_scores()[0]
    newton_scores = -lr.one_step_newton()[0]

    # sort indices in ascending order.
    if positivebeta:
        # sort indices in ascending order.
        if_inds = np.argsort(if_scores)
        newton_inds = np.argsort(newton_scores)
    else:
        # sort indices in descending order.
        if_inds = np.argsort(if_scores)[::-1]
        newton_inds = np.argsort(newton_scores)[::-1]

    # sort scores according to indices.
    sorted_scores = if_scores[if_inds]
    sorted_newton_scores = newton_scores[newton_inds]

    # compute residuals and leverages.
    residuals = lr.residual()
    leverages = lr.leverage_scores()

    x1 = [pt[0] for pt in x]

    # create a df with residuals, leverages, and coordinates.
    orig_df = pd.DataFrame({'x': x1, 'y': y, 
                            'residual': residuals, 'leverage': leverages, 
                            'IF': if_scores, '1Exact': newton_scores,
                            'sorted_IF_indices': if_inds, 'sorted_1Exact_indices': newton_inds,
                            'sorted_IF_scores': sorted_scores, 'sorted_1Exact_scores': sorted_newton_scores})
    return orig_df

def create_orig_df(x, y, lr, positivebeta):
    '''
    x: design matrix.
    y: response vector.
    lr: linear regression object.
    '''
    # compute IF/1Exact scores.
    if_scores = -lr.influence_scores()[1]
    newton_scores = -lr.one_step_newton()[1]

    # sort indices in ascending order.
    if positivebeta:
        # sort indices in ascending order.
        if_inds = np.argsort(if_scores)
        newton_inds = np.argsort(newton_scores)
    else:
        # sort indices in descending order.
        if_inds = np.argsort(if_scores)[::-1]
        newton_inds = np.argsort(newton_scores)[::-1]

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
    plt.scatter(orig_df[:10]['x1'], orig_df[:10]['y'], marker='o', color='black', label='Pop. A') # pop A
    plt.scatter(orig_df[10:]['x1'], orig_df[10:]['y'], marker='x', color='red', label='Pop. B') # pop B

    # Add dropped order positioned by each point
    for index, row in sorted_if_df.iterrows():
        if row['sorted_idx'] <= 9:
            plt.text(row['x1'], row['y'], int(row['sorted_idx']), ha='left', va='top', fontsize=20)


    plt.xlabel('x1')
    plt.ylabel('y')
    plt.title('Scores Ordered')
    plt.legend()
    plt.show()
    
    return
