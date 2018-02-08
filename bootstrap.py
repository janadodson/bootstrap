"""
Functions related to running a vectorized bootstrap of standard errors.

Author : Jana Dodson
Email : janadodson1@gmail.com
Version : 1.0

Dependencies
------------
numpy : version 1.11.3
pandas : version 0.19.2
"""


import numpy as np
import pandas as pd


def bootstrap_se(x, wts=None, n_reps=10000):
    """
    Compute bootstrapped standard error.

    Parameters
    ----------
    x : array-like with shape = (n_observations, ) or (n_observations, 
    n_features)
        1D or 2D array containing the value(s) for each observation.
    wts : array-like with shape = (n_observations, ), optional
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    n_reps : positive int, optional
        Count of random sample replicates. When unspecified, default is 
        10,000.

    Returns
    -------
    se : numpy.float64 or array-like with shape (n_features, )
        
    """
    
    # Convert input arrays to numpy arrays
    temp_x = np.asanyarray(x)
    temp_wts = np.asanyarray(wts)
        
    # By default, all observations are weighted equally
    if wts is None:
        temp_wts = np.ones(len(temp_x))

    # Check that input arrays have the same length
    if len(temp_x) != len(temp_wts):
        raise Exception("Input arrays 'x' and 'wts' must have the same length.")
        
    # Check input array dimensions
    if temp_x.ndim not in (1, 2):
        raise Exception("Input array 'x' must be 1- or 2-dimensional.")
    if temp_wts.ndim != 1:
        raise Exception("Input array 'wts' must be 1-dimensional.")
        
    # Check that n_reps is a positive int
    if not isinstance(n_reps, int) or n_reps <= 0:
         raise Exception("Input value 'n_reps' must be a positive integer.")
    
    # Create array of observation frequencies for each random sample replicate
    boot_counts = np.random.multinomial(n=len(temp_x), pvals=np.ones(len(temp_x)) / len(temp_x), size=n_reps)
    
    # Calculate SE for each column in data
    if x.ndim == 1:
        se = (
            np.multiply(boot_counts, np.multiply(temp_wts, temp_x)).sum(1) 
            / np.multiply(boot_counts, np.multiply(temp_wts, ~np.isnan(temp_x))).sum(1)
        ).std()
    elif x.ndim == 2:
        se = np.array([
            (
                np.multiply(boot_counts, np.multiply(temp_wts, col)).sum(1) 
                / np.multiply(boot_counts, np.multiply(temp_wts, ~np.isnan(col))).sum(1)
            ).std()
            for col in temp_x.T
        ])
        if isinstance(x, pd.DataFrame):
            se = pd.Series(se, index=x.columns)
            
    return se

