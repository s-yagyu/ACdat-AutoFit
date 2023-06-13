# ReLU absolute fit

import numpy as np
from scipy.optimize import minimize

# m_relu関数の定義
@np.vectorize
def m_relu(x, Nor, Ip, Bg):
    u = (x - Ip)
    return Nor * u * (u > 0.0) + Bg

# 絶対誤差の定義
def absolute_error(params, x, y):

    y_pred = m_relu(x, *params)
    return np.sum(np.abs(y - y_pred))

# 最小二乗誤差の定義
def least_squares_error(params, x, y):
    y_pred = m_relu(x, *params)
    return np.sum((y - y_pred) ** 2)


def fit_m_relu(x, y, params_init=None):
    """Absolute fitting
    Args:
        x (ndarray): _description_
        y (ndarray): _description_
        params_init (list or ndarray, optional): _description_. Defaults to None.

    Returns:
        float: Nor_opt, Ip_opt, Bg_opt
        
    examples:
    fit_param = fit_m_relu(xdata,ydata, params_init=[1, 4.5, 0.0])
    
    """
    
    if params_init is None:
        params_init = np.array([1, 4.5, 0.0])

    result = minimize(absolute_error, params_init, args=(x, y))
    # result = minimize(least_squares_error, params_init, args=(x, y))

    Nor_opt, Ip_opt, Bg_opt = result.x

    return Nor_opt, Ip_opt, Bg_opt