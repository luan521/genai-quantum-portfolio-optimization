from itertools import product
import numpy as np
import pandas as pd

def cost_function(x, expected_value, cov_matrix, q, lamb, B):
    """
    Args:
        x (np.array(int)): Array of binary values.
        expected_value (list[float]): List of asset returns.
        cov_matrix (DataFrame-like): Covariance matrix.
        q (float): Covariance scaling factor.
        B (float): Budget/threshold parameter.
        lamb (float): Penalty parameter.
        
    Returns:
        float
    """
    response = -(expected_value @ x) + q*(x @ cov_matrix @ x) + lamb*(B-np.sum(x))**2
    return response

def sort_by_cost_function(expected_value, cov_matrix, q, lamb, B):
    """
    Args:
        expected_value (list[float]): List of asset returns.
        cov_matrix (DataFrame-like): Covariance matrix.
        q (float): Covariance scaling factor.
        B (float): Budget/threshold parameter.
        lamb (float): Penalty parameter.
        
    Returns:
        pandas.DataFrame: Dataframe sorted by cost_function applied to all possible solutions
    """
    data = []
    for x in list(product([0,1], repeat=len(expected_value))):
        c = cost_function(np.array(x), expected_value, cov_matrix, q, lamb, B)
        data.append({'solution': x, 'cost_function': c})
    response = pd.DataFrame(data).sort_values('cost_function')
    return response