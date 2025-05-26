"""
Fitting and Plot Library
This module provides various functions for data fitting and plotting, 
including linear fitting, ReLU fitting, and utility functions for data preprocessing and evaluation.

"""
import re

from scipy.optimize import  minimize, differential_evolution
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter('ignore')
# warnings.resetwarnings()
# warnings.simplefilter('ignore', FutureWarning)

# ------ fitting functions
def line_func(xdata: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Linear function for fitting purposes.

    Args:
        xdata (np.ndarray): Independent variable data.
        slope (float): Slope of the line.
        intercept (float): Intercept of the line.

    Returns:
        np.ndarray: Computed y values.
    """
    return slope * xdata + intercept

def abs_line_fit(xdata: np.ndarray, ydata: np.ndarray, params_init: np.ndarray = None) -> tuple:
    """Performs absolute error fitting using minimization.

    Args:
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.
        params_init (np.ndarray, optional): Initial parameters for fitting. Defaults to None.

    Returns:
        tuple: Slope and intercept of the fitted line.
    """
    if params_init is None:
        params_init = np.polyfit(xdata, ydata, 1)

    result = minimize(_line_absolute_error, params_init, args=(xdata, ydata))
    slope, intercept = result.x

    return slope, intercept

def _line_absolute_error(params: np.ndarray, xdata: np.ndarray, ydata: np.ndarray) -> float:
    """Computes the absolute error between model predictions and actual data.

    Args:
        params (np.ndarray): Parameters (slope, intercept).
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.

    Returns:
        float: Sum of absolute errors.
    """
    y_pred = line_func(xdata, *params)
    return np.sum(np.abs(ydata - y_pred))

def y_intercept(Nor: float, Ip: float, Bg: float) -> float:
    """Calculates the y-intercept for a ReLU function.

    Args:
        Nor (float): Normalization factor.
        Ip (float): Input parameter.
        Bg (float): Background value.

    Returns:
        float: Calculated y-intercept.
    """
    intercept = Bg - Nor * Ip
    return intercept

def zero_cross(Nor: float, Ip: float, Bg: float) -> dict:
    """Calculates the zero-crossing point for a ReLU function.

    Args:
        Nor (float): Normalization factor.
        Ip (float): Input parameter.
        Bg (float): Background value.

    Returns:
        dict: Dictionary containing coefficients and zero-crossing point.
    """
    b = Bg - Nor * Ip
    x0cross = -b / Nor
    return {'a': Nor, 'b': b, 'cross': x0cross}

#------ReLU fitting
def relu_func(x: np.ndarray, Nor: float, Ip: float, Bg: float) -> np.ndarray:
    """ReLU function for fitting purposes.

    Args:
        x (np.ndarray): Independent variable data.
        Nor (float): Normalization factor.
        Ip (float): Input parameter.
        Bg (float): Background value.

    Returns:
        np.ndarray: Computed y values.
    """
    u = (x - Ip)
    return Nor * u * (u > 0.0) + Bg

def _relu_absolute_error(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Computes the absolute error for ReLU fitting.

    Args:
        params (np.ndarray): Parameters (Nor, Ip, Bg).
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.

    Returns:
        float: Sum of absolute errors.
    """
    y_pred = relu_func(x, *params)
    return np.sum(np.abs(y - y_pred))

def _relu_least_squares_error(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Computes the least squares error for ReLU fitting.

    Args:
        params (np.ndarray): Parameters (Nor, Ip, Bg).
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.

    Returns:
        float: Sum of squared errors.
    """
    y_pred = relu_func(x, *params)
    return np.sum((y - y_pred) ** 2)

def relu_fit(x: np.ndarray, y: np.ndarray, params_init: np.ndarray = None, 
             min_error: str = 'mae', bounds: list = None) -> tuple:
    """Fits data using a ReLU function with minimization.

    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        params_init (np.ndarray, optional): Initial parameters for fitting. Defaults to None.
        min_error (str, optional): Error metric ('mae' or 'mse'). Defaults to 'mae'.
        bounds (list, optional): Bounds for parameters. Defaults to None.

    Returns:
        tuple: Optimized parameters (Nor, Ip, Bg).
    """
    if params_init is None:
        nor = 1
        ip = np.percentile(x, 50)
        bg = np.nanmean(y[:5]) 
        params_init = np.array([nor, ip, bg])

    if bounds is None:
        s_para = np.max(y) / (np.abs(x[-1] - x[-2]))
        bg_para = np.abs(np.mean(y[:4]) * 10)
        bounds = [(0, s_para), (np.min(x), np.max(x)), (-1 * bg_para, bg_para)]

    if min_error == 'mae':
        result = minimize(_relu_absolute_error, params_init, bounds=bounds, args=(x, y))
    else:
        result = minimize(_relu_least_squares_error, params_init, bounds=bounds, args=(x, y))

    Nor_opt, Ip_opt, Bg_opt = result.x
    return Nor_opt, Ip_opt, Bg_opt

def relu_de_fit(x: np.ndarray, y: np.ndarray, bounds: list = None) -> tuple:
    """Fits data using a ReLU function with differential evolution.

    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        bounds (list, optional): Bounds for parameters. Defaults to None.

    Returns:
        tuple: Optimized parameters (Nor, Ip, Bg).
    """
    if bounds is None:
        s_para = np.max(y) / (np.abs(x[-1] - x[-2]))
        bg_para = np.abs(np.mean(y[:4]) * 10)
        bounds = [(0, s_para), (np.min(x), np.max(x)), (-1 * bg_para, bg_para)]

    result = differential_evolution(_relu_absolute_error, bounds, args=(x, y))
    Nor_opt, Ip_opt, Bg_opt = result.x

    return Nor_opt, Ip_opt, Bg_opt

# -----
def get_nearest_value(lst: list, num: float) -> int:
    """Finds the element in the list closest to the given number.

    Args:
        lst (list): List of numbers.
        num (float): Target number.

    Returns:
        int: Index of the closest element.
    """
    idx = np.abs(np.asarray(lst) - num).argmin()
    return idx

def nan_inf_rm(xdata: np.ndarray, ydata: np.ndarray, zero_replace: bool = False, 
               info: bool = False) -> tuple:
    """Removes NaN and Inf values from data.

    Args:
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.
        zero_replace (bool, optional): Replace NaN with 0 if True. Defaults to False.
        info (bool, optional): Display information if True. Defaults to False.

    Returns:
        tuple: Cleaned xdata and ydata.
    """
    xarray = np.array(xdata)
    yarray = np.array(ydata)
    tr_inf_x = np.where(np.isinf(xarray), np.nan, xarray)
    tr_inf_y = np.where(np.isinf(yarray), np.nan, yarray)

    re_xdata = tr_inf_x.copy()
    re_ydata = tr_inf_y.copy()

    if zero_replace:
        re_xdata = np.nan_to_num(re_xdata, copy=False)
        re_ydata = np.nan_to_num(re_ydata, copy=False)
    else:
        nan_ind_x = np.where(~np.isnan(tr_inf_x))
        re_xdata = re_xdata[nan_ind_x[0]]
        re_ydata = re_ydata[nan_ind_x[0]]

        nan_ind_y = np.where(~np.isnan(re_ydata))
        re_xdata = re_xdata[nan_ind_y[0]]
        re_ydata = re_ydata[nan_ind_y[0]]

    if info:
        print(f'Number of NaN in (x, y): ({np.count_nonzero(np.isnan(tr_inf_x))}, {np.count_nonzero(np.isnan(tr_inf_y))})')
        print(f'Original shape x, y: {xarray.shape}, {yarray.shape}')
        print(f'Output shape x, y: {re_xdata.shape}, {re_ydata.shape}')

    return re_xdata, re_ydata

def r2n_score(y_data: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the R^2 score for true and predicted values.

    Args:
        y_data (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: R^2 score.
    """
    rss = np.sum((y_data - y_pred) ** 2)
    y_mean = np.mean(y_data)
    tss = np.sum((y_data - y_mean) ** 2)
    r2 = 1 - rss / tss
    return r2

#---- plot

# Note: figure size
# 黄金比 1：1.62 wdth:height = 4.86:3
# 白銀比 1：1.43  wdth:height = 4.3:3 
# figsize=(width,height) wdth:height= 4:3 
# (row,col)->figsize(col*4,row*3): (3,4)->figsize(16,9)
#  w_in, h_in = plt.rcParams[cParams["figure.figsize"]


def re_replace(text: str) -> str:
    """Removes special symbols from text using regular expressions.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text with special symbols removed.

    Examples:
        text = '4 inch $\\phi$=0'
        re_replace(text)
        >>> '4_inch_phi0'
    """
    code_regex = re.compile('[!"#$%&\\\'\\()*+,-./:;<=>?@[\\]^_`{|}~]')
    cleaned_text = code_regex.sub('', text).replace(' ', '_')
    return cleaned_text
