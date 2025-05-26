"""
Log-Log Fit Analysis Module
Analysis module for inverse power fitting and ReLU-based fitting.

This module provides functions for performing inverse power transformations
and fitting data using rectified linear unit (ReLU) models. It includes
support for handling negative values, plotting results, and calculating
various fitting metrics.

This module provides tools for analyzing data using log-log fitting and 
inverse power transformations. It includes methods for estimating shifts, 
removing background noise, and evaluating fitting results with various metrics.
"""
import math
import numpy as np
import scipy.optimize as optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from pfitlib.fit_lib import (relu_func, relu_fit, relu_de_fit,  
                                line_func, _line_absolute_error,
                                get_nearest_value, r2n_score, 
                                nan_inf_rm, re_replace)

import warnings
warnings.simplefilter('ignore')

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

class LogAnalysis:
    """
    A class for performing log-log and inverse power fitting analysis.
    """

    def __init__(self, xdata: np.ndarray, ydata: np.ndarray):
        """
        Initializes the LogAnalysis object.

        Args:
            xdata (np.ndarray): Array of x-values.
            ydata (np.ndarray): Array of y-values.
        """
        self.xdata = xdata
        self.ydata = ydata

    def inv_n_scan(self, search_range: list = None, zero_replace: bool = False, 
                   min_error: str = 'mae', likely_evaluation: str = 'r2_bp', 
                   info: bool = False, plot: bool = True, plot_fig_step: int = 1):
        """
        Performs inverse power fitting scan over a range of powers.

        Args:
            search_range (list, optional): Range of powers to evaluate. Defaults to None.
            zero_replace (bool, optional): Replace NaN values with zero. Defaults to False.
            min_error (str, optional): Error metric to minimize ('mae' or 'mse'). Defaults to 'mae'.
            likely_evaluation (str, optional): Evaluation metric ('r2' or 'r2_bp'). Defaults to 'r2_bp'.
            info (bool, optional): Print additional information. Defaults to False.
            plot (bool, optional): Plot the results. Defaults to True.
            plot_fig_step (int, optional): Step size for plotting figures. Defaults to 1.
        """
        self.res_inv = xpower_search(self.xdata, self.ydata, 
                                     search_range=search_range,
                                     r2_plot=plot, process_plot=plot,
                                     min_error=min_error,
                                     zero_replace=zero_replace,
                                     likely_evaluation=likely_evaluation,
                                     info=info, 
                                     every_plot=False, 
                                     plot_save=False,
                                     plot_fig_step=plot_fig_step)

    def dual_log_scan(self, search_range: list, fit_type: str = 'weight', 
                      bg_num: int = 3, lim_val: float = 0.5, 
                      min_error: str = 'mae', info: bool = True, 
                      plot: bool = True, plot_fig_step: int = 1):
        """
        Performs shifted log-log fitting scan over a range of shifts.

        Args:
            search_range (list): Range of shifts to evaluate.
            fit_type (str, optional): Type of fitting ('mae', 'weight', or 'mse'). Defaults to 'weight'.
            bg_num (int, optional): Number of points for background averaging. Defaults to 3.
            lim_val (float, optional): Threshold value for filtering x-axis values. Defaults to 0.5.
            min_error (str, optional): Error metric to minimize ('mae' or 'mse'). Defaults to 'mae'.
            info (bool, optional): Print additional information. Defaults to True.
            plot (bool, optional): Plot the results. Defaults to True.
            plot_fig_step (int, optional): Step size for plotting figures. Defaults to 1.
        """
        self.res_dlog = xshift_search(self.xdata, self.ydata, 
                                      search_range,
                                      bg_num=bg_num, 
                                      lim_val=lim_val, 
                                      fit_type=fit_type,
                                      r2_plot=plot, 
                                      process_plot=plot,
                                      min_error=min_error,
                                      info=info,
                                      every_plot=False, 
                                      plot_save=False,
                                      plot_fig_step=plot_fig_step)
# inv_n functions
def const_inv_power_fit(xdata: np.ndarray, ydata: np.ndarray, power_num: float = 2, 
                        ini_params: list = None, min_error: str = 'mae', 
                        retry_r2: float = 0.9, negative: bool = False, 
                        zero_replace: bool = False, plot: bool = False, 
                        info: bool = False) -> dict:
    """
    Performs inverse power transformation and ReLU-based fitting.

    Args:
        xdata (np.ndarray): Array of x-values.
        ydata (np.ndarray): Array of y-values.
        power_num (float, optional): The power number for transformation. Defaults to 2.
        ini_params (list, optional): Initial parameters for fitting (Nor, Ip, Bg). Defaults to None.
        min_error (str, optional): Error metric to minimize ('mae' or 'mse'). Defaults to 'mae'.
        retry_r2 (float, optional): R^2 threshold for retrying fitting. Defaults to 0.9.
        negative (bool, optional): Handle negative values during transformation. Defaults to False.
        zero_replace (bool, optional): Replace NaN values with zero. Defaults to False.
        plot (bool, optional): Plot the data and fitting results. Defaults to False.
        info (bool, optional): Print additional information. Defaults to False.

    Returns:
        dict: Dictionary containing the fitted results and additional information.
    """
    if ini_params is None:
        nor = 1
        ip = np.percentile(xdata, 50)
        bg = np.nanmean(ydata[:5]) 
        ini_params = np.array([nor, ip, bg])

    def none_negative_power(ydata, pw):
        """
        Apply a power transformation without resulting in NaN for negative values.

        Args:
            ydata (ndarray): Array of y-values.
            pw (float): Power to apply.

        Returns:
            ndarray: Transformed y-values.
        """
        ysign = np.sign(ydata)
        y_power = ysign * np.power(np.abs(ydata), pw)
        return y_power
    
    if negative:
        yydata = none_negative_power(ydata, pw=1/power_num)
    else:
        yydata = np.power(ydata, 1/power_num) 
     
    re_xdata, re_ydata = nan_inf_rm(xdata, yydata, 
                                    zero_replace=zero_replace, 
                                    info=info)
    re_fit = re_ydata.copy()
    res_params = relu_fit(re_xdata, re_ydata, ini_params, min_error)
    re_fit = relu_func(re_xdata, *res_params)
    
    # Breakpoint index
    idx = get_nearest_value(re_xdata, res_params[1])
    
    try: 
        r2 = r2n_score(re_ydata, re_fit) 
        r2_bp = r2n_score(re_ydata[idx:], re_fit[idx:])
    except:
        r2, r2_bp = -1, -1

    if r2 < retry_r2:
        # differential_evolution
        res_params = relu_de_fit(re_xdata, re_ydata )
        re_fit = relu_func(re_xdata, *res_params)
        idx = get_nearest_value(re_xdata, res_params[1])
        try:
            r2 = r2n_score(re_ydata, re_fit)
            r2_bp = r2n_score(re_ydata[idx:], re_fit[idx:])
        except:
            r2, r2_bp = -1, -1

    if info:
        print(f'Power (n): {power_num:.2f}, 1/Power (1/n): {1/power_num:.2f}')
        print(f'R^2: {r2:.2f}')
        print(f'R^2 > threshold: {r2_bp:.2f}')
        print(f'Params (Nor, Ip, Bg): {res_params}')

    if plot:
        _plot_results(xdata, ydata, re_xdata, re_ydata, re_fit, res_params, power_num)

    return {'rex': re_xdata, 'rey': re_ydata, 'fit': re_fit, 
            'n': power_num, 'popt': res_params, 'r2': r2, 'r2_bp': r2_bp}

def _plot_results(xdata, ydata, re_xdata, re_ydata, re_fit, res_params, power_num):
    """
    Plot the original and fitted data.

    Args:
        xdata (ndarray): Original x-data.
        ydata (ndarray): Original y-data.
        re_xdata (ndarray): Transformed x-data.
        re_ydata (ndarray): Transformed y-data.
        re_fit (ndarray): Fitted y-data.
        res_params (list): Fitted parameters.
        power_num (float): Power number used for transformation.
    """
    fig = plt.figure(figsize=(12, 5), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(xdata, ydata, 'ro-', label='Data')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(which='both')

    ax2 = plot_ax(re_xdata, re_ydata, re_xdata, re_fit, 
                  breakpoints=[res_params[1]],
                  lgtitle=f'n: {power_num:.2f}\n1/n: {1/power_num:.2f}',
                  title='', axi=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel(f'y$^{{1/{power_num:.1f}}}$')
    plt.show()

def plot_ax(xdata, ydata, m_xdata, m_ydata, breakpoints=None, lgtitle=None, title='', axi=None):
    """
    Plot data with optional breakpoints and legends.

    Args:
        xdata (ndarray): Original x-data.
        ydata (ndarray): Original y-data.
        m_xdata (ndarray): Processed x-data.
        m_ydata (ndarray): Processed y-data.
        breakpoints (list, optional): List of breakpoints to mark. Defaults to None.
        lgtitle (str, optional): Legend title. Defaults to None.
        title (str, optional): Plot title. Defaults to ''.
        axi (matplotlib.axes.Axes, optional): Existing axis to plot on. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    if axi is None:
        fig_ = plt.figure()
        ax_ = fig_.add_subplot(111)
    else:
        ax_ = axi
                  
    ax_.set_title(f'{title}')    
    ax_.plot(xdata, ydata,'ro',label='Data')
    ax_.plot(m_xdata, m_ydata,'bo-',label='Processed')
    
    if breakpoints is not None:
        for i, bp in enumerate(breakpoints):
            ax_.axvline(bp)
            ax_.text(bp,np.max(ydata)*0.8, f'{bp:.2f}')
        
    ax_.grid(which='both')
    
    if lgtitle is None:
        ax_.legend() 
        
    else:   
        ax_.legend(title=lgtitle)
    
    if axi is None:
        plt.show()
        return 
    
    return ax_

# Shift-Log-Log functions
def val_rm(xdata: np.ndarray, ydata: np.ndarray, lim_val: float, info: bool = True) -> tuple:
    """
    Removes values smaller than the specified threshold along the x-axis.

    Args:
        xdata (np.ndarray): Array of x-values.
        ydata (np.ndarray): Array of y-values.
        lim_val (float): Threshold value for filtering x-axis values.
        info (bool, optional): Print information about the operation. Defaults to True.

    Returns:
        tuple: Filtered x and y data arrays.
        
    Note:
        Delete values smaller than the specified value on the x-axis.
        There is a large variation in the data near the threshold. 
        Shift Factor that determines how far away the value is from the moved location to perform the evaluation.
        lim_val: Value before taking the log of the x-axis (0.1, 0.5, etc.)
       
        -inf < ln(x) < 0  at 0 < x < 1 
        When x is less than 1, taking the log result in a negative value.
        
        lim_val: Value before taking the log of the x-axis (0.1, 0.5, etc.)
        
        ln(lim_val)= val
        ln(1) = 0
        ln(0.1) = -2.3
        ln(0.5) = -0.69
        ln(0) = -inf
        lim_val
        
        np.e**(val)= lim_val
        np.e**(0) = 1
        np.e**(-0.69) = 0.5
        np.e**(-1) = 0.36
        np.e**(-2.3) = 0.1
        np.e**(-3) = 0.05

    Example:
        val_rm(np.log(xf), np.log(yf), 0.3)
    """
    val = np.log(lim_val)
    tr_val_x = np.where(xdata > val)
    
    re_xdata = xdata.copy()
    re_ydata = ydata.copy()
    re_xdata = xdata[tr_val_x[0]]
    re_ydata = ydata[tr_val_x[0]]
    
    if info:
        print(f'Number of values removed under {val} on xdata: {np.count_nonzero(tr_val_x)}')
        print(f'Shape of x, y: {re_xdata.shape}, {re_ydata.shape}')

    return re_xdata, re_ydata


def _apply_power_transformation(ydata: np.ndarray, power: float, negative: bool = False) -> np.ndarray:
    """
    Applies a power transformation to the data.

    Args:
        ydata (np.ndarray): Array of y-values.
        power (float): Power to apply.
        negative (bool, optional): Handle negative values. Defaults to False.

    Returns:
        np.ndarray: Transformed y-values.
    """
    if negative:
        ysign = np.sign(ydata)
        return ysign * np.power(np.abs(ydata), power)
    return np.power(ydata, power)

def _filter_data(xdata: np.ndarray, ydata: np.ndarray, lim_val: float, info: bool = True) -> tuple:
    """
    Filters data based on a threshold value for the x-axis.

    Args:
        xdata (np.ndarray): Array of x-values.
        ydata (np.ndarray): Array of y-values.
        lim_val (float): Threshold value for filtering x-axis values.
        info (bool, optional): Print information about the operation. Defaults to True.

    Returns:
        tuple: Filtered x and y data arrays.
    """
    val = np.log(lim_val)
    valid_indices = xdata > val
    filtered_x = xdata[valid_indices]
    filtered_y = ydata[valid_indices]

    if info:
        print(f'Filtered {np.sum(~valid_indices)} values below {lim_val}.')
    return filtered_x, filtered_y

def _fit_data(xdata: np.ndarray, ydata: np.ndarray, fit_type: str = 'mae') -> tuple:
    """
    Fits data using the specified fitting type.

    Args:
        xdata (np.ndarray): Array of x-values.
        ydata (np.ndarray): Array of y-values.
        fit_type (str, optional): Type of fitting ('mae', 'weight', or 'mse'). Defaults to 'mae'.

    Returns:
        tuple: Fitted parameters and R^2 value.
    """
    try:
        if fit_type == 'mae':
            result = optimize.minimize(_line_absolute_error, (1, 1), args=(xdata, ydata))
            popt = result.x
        elif fit_type == 'weight':
            popt, _ = curve_fit(line_func, xdata, ydata, p0=(1, 1), sigma=1/xdata)
        else:
            popt, _ = curve_fit(line_func, xdata, ydata, p0=(1, 1))
        fity = line_func(xdata, *popt)
        r2 = r2n_score(ydata, fity)
        return popt, r2
    except Exception as e:
        print(f"Fitting error: {e}")
        return (np.nan, np.nan), np.nan

# log_log_fit function
def log_log_fit(xdata: np.ndarray, ydata: np.ndarray, bg_num: int = None, 
                lim_val: float = 0.36, fit_type: str = 'mae', 
                comment: str = '', info: bool = True, 
                plot: bool = True) -> dict:
    """
    Performs log-log fitting to estimate power and shift.

    Args:
        xdata (np.ndarray): Array of x-values.
        ydata (np.ndarray): Array of y-values.
        bg_num (int, optional): Number of points for background averaging. Defaults to None.
        lim_val (float, optional): Threshold value for filtering x-axis values. Defaults to 0.36.
        fit_type (str, optional): Type of fitting ('mae', 'weight', or 'mse'). Defaults to 'mae'.
        comment (str, optional): Additional comment to print. Defaults to ''.
        info (bool, optional): Print additional information. Defaults to True.
        plot (bool, optional): Plot the fitting results. Defaults to True.

    Returns:
        dict: Dictionary containing fitting results.
    """
    if bg_num is not None:
        ydata = ydata - np.nanmean(ydata[:bg_num])

    log_xdata, log_ydata = nan_inf_rm(np.log(xdata), np.log(ydata), zero_replace=False, info=info)
    log_xdata, log_ydata = _filter_data(log_xdata, log_ydata, lim_val, info=info)

    popt, r2 = _fit_data(log_xdata, log_ydata, fit_type=fit_type)

    if info:
        print(comment)
        print(f'Slope: {popt[0]:.3f}, Intercept: {popt[1]:.3f}, $R^2$: {r2:.3f}')

    if plot:
        plt.plot(log_xdata, log_ydata, 'ro', label='data')
        plt.plot(log_xdata, line_func(log_xdata, *popt), 'b-', label='fit')
        plt.grid(True)
        plt.legend(title=f'Slope: {popt[0]:.3f}\nIntercept: {popt[1]:.3f}\n$R^2$: {r2:.3f}')
        plt.show()

    return {'popt': popt, 'r2': r2, 'rex': log_xdata, 'rey': log_ydata, 'fit': line_func(log_xdata, *popt)}

def create_search_range(center: float, range_: float, step: float) -> np.ndarray:
    """
    Creates a range of values for searching.

    Args:
        center (float): Center value of the range.
        range_ (float): Range to search.
        step (float): Step size within the range.

    Returns:
        np.ndarray: Array of search range values.

    Example:
        s_range = create_search_range(center=5, range_=1, step=0.2)
        >>> array([4. , 4.2, 4.4, 4.6, 4.8, 5. , 5.2, 5.4, 5.6, 5.8, 6. ])
    """
    return np.arange(center - range_, center + range_ + step, step)

# Scan functions

def xshift_search(xdata: np.ndarray, ydata: np.ndarray, search_range: list, 
                  bg_num: int = None, lim_val: float = 0.36, fit_type: str = 'mae',
                  r2_plot: bool = True, process_plot: bool = True, min_error: str = 'mae', 
                  info: bool = False, every_plot: bool = False, plot_save: bool = False, 
                  plot_fig_step: int = 1) -> dict:
    """
    Performs shifted log-log fitting over a range of shifts.

    Args:
        xdata (np.ndarray): Array of x-values.
        ydata (np.ndarray): Array of y-values.
        search_range (list): List of shift values to evaluate.
        bg_num (int, optional): Number of points for background averaging. Defaults to None.
        lim_val (float, optional): Threshold value for filtering x-axis values. Defaults to 0.36.
        fit_type (str, optional): Type of fitting ('mae', 'weight', or 'mse'). Defaults to 'mae'.
        r2_plot (bool, optional): Plot R2 results. Defaults to True.
        process_plot (bool, optional): Plot fitting process. Defaults to True.
        min_error (str, optional): Error metric to minimize. Defaults to 'mae'.
        info (bool, optional): Print additional information. Defaults to False.
        every_plot (bool, optional): Plot each fitting process. Defaults to False.
        plot_save (bool, optional): Save the plots. Defaults to False.
        plot_fig_step (int, optional): Step size for plotting figures. Defaults to 1.

    Returns:
        dict: Dictionary containing fitting results.
    """
    power = []
    r2 = []
    res_list = []
    applied_search_range = []
    
    for shift in search_range:
        res_para = log_log_fit(xdata - shift, ydata, bg_num=bg_num,
                               lim_val=lim_val, fit_type=fit_type, 
                               comment=f'shift: {shift:.2f}', info=info, plot=every_plot)
        
        if not math.isnan(res_para['r2']):
            applied_search_range.append(shift)
            power.append(res_para['popt'][0])
            r2.append(res_para['r2'])
            res_list.append(res_para)
 
    if process_plot:
        multi_plots(applied_search_range, res_list, title='Shift-Log-Log', save=plot_save,
                    plot_fig_step=plot_fig_step)    
    
    max_r2_idx = get_nearest_value(r2, 1)
    likeli_power = power[max_r2_idx]
    likeli_shift = applied_search_range[max_r2_idx]
    likeli_r2 = r2[max_r2_idx]
    likeli_rex = res_list[max_r2_idx]['rex']
    likeli_rey = res_list[max_r2_idx]['rey']
    likeli_fit = res_list[max_r2_idx]['fit']
    
    msg = f'Likelihood shift: {likeli_shift:.3f}, power: {likeli_power:.3f}'
    
    if info:
        print(msg)
    
    if r2_plot:
        search_r2_plot(shifts=applied_search_range, powers=power, r2=r2, msg=msg, save=plot_save)
        
        const_inv_power_fit(xdata, ydata, power_num=likeli_power,
                            ini_params=(1, likeli_shift, 0), 
                            min_error=min_error, retry_r2=0.9, 
                            zero_replace=True, plot=True, info=False)
          
    return {'r2': likeli_r2, 'shift': likeli_shift, 'power': likeli_power,
            'rex': likeli_rex, 'rey': likeli_rey, 'fit': likeli_fit}


def xpower_search(xdata: np.ndarray, ydata: np.ndarray, search_range: list = None, 
                  r2_plot: bool = True, process_plot: bool = True, min_error: str = 'mae', 
                  zero_replace: bool = False, likely_evaluation: str = 'r2', 
                  info: bool = True, every_plot: bool = False, plot_save: bool = False, 
                  plot_fig_step: int = 1) -> dict:
    """
    Performs inverse power fitting over a range of powers.

    Args:
        xdata (np.ndarray): Array of x-values.
        ydata (np.ndarray): Array of y-values.
        search_range (list, optional): Range of powers to evaluate. Defaults to None.
        r2_plot (bool, optional): Plot R2 results. Defaults to True.
        process_plot (bool, optional): Plot fitting process. Defaults to True.
        min_error (str, optional): Error metric to minimize. Defaults to 'mae'.
        zero_replace (bool, optional): Replace NaN values with zero. Defaults to False.
        likely_evaluation (str, optional): Evaluation metric ('r2' or 'r2_bp'). Defaults to 'r2'.
        info (bool, optional): Print additional information. Defaults to True.
        every_plot (bool, optional): Plot each fitting process. Defaults to False.
        plot_save (bool, optional): Save the plots. Defaults to False.
        plot_fig_step (int, optional): Step size for plotting figures. Defaults to 1.

    Returns:
        dict: Dictionary containing fitting results.
    """
    shift = []
    r2 = []
    r2_bp = []
    res_list = []
    applied_search_range = []
    
    if search_range is None:
        search_range = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    if len(search_range) == 3:
        ncols = 3
    elif len(search_range) == 2:
        ncols = 2 
    else:
        ncols = 4 
      
    for power_num in search_range:
        res_para = const_inv_power_fit(xdata, ydata, power_num=power_num, 
                                       ini_params=None, min_error=min_error,
                                       zero_replace=zero_replace, plot=every_plot, info=info)
        
        if not math.isnan(res_para['r2']):
            applied_search_range.append(power_num)
            shift.append(res_para['popt'][1])
            r2.append(res_para['r2'])
            r2_bp.append(res_para['r2_bp'])
            res_list.append(res_para)
    
    if process_plot:
        multi_plots(applied_search_range, res_list, title='1/n-Power', ncols=ncols, 
                    para_inv=True, save=plot_save, plot_fig_step=plot_fig_step)
    
    if likely_evaluation == 'r2_bp':
        eva_r2 = r2_bp
        decision_eva = r'$R^{2}_{bp}$' 
    else:
        eva_r2 = r2
        decision_eva = r'$R^{2}$'
        
    max_r2_idx = get_nearest_value(eva_r2, 1)
 
    likeli_shift = shift[max_r2_idx]
    likeli_power = applied_search_range[max_r2_idx]
    likeli_r2 = eva_r2[max_r2_idx]
    likeli_rex = res_list[max_r2_idx]['rex']
    likeli_rey = res_list[max_r2_idx]['rey']
    likeli_fit = res_list[max_r2_idx]['fit']
    likeli_popt = res_list[max_r2_idx]['popt']
    
    msg = f'{decision_eva}, likelihood shift: {likeli_shift:.3f}, power: {likeli_power:.1f}, 1/power: {1/likeli_power:.3f}'
    
    if info:
        print(msg)
    
    if r2_plot:
        search_r2_plot(shifts=shift, powers=applied_search_range, r2=eva_r2, msg=msg, save=plot_save)
        
        with_bg_fit = const_inv_power_fit(xdata, ydata, power_num=likeli_power,
                                          ini_params=(1, likeli_shift, 0), 
                                          zero_replace=zero_replace, retry_r2=0.9, 
                                          plot=True, info=False) 
        # Backgroundを差し引いて解析を行う
        const_inv_power_fit(xdata, ydata - with_bg_fit['popt'][2]**likeli_power, 
                            power_num=likeli_power, ini_params=(1, likeli_shift, 0), 
                            zero_replace=zero_replace, retry_r2=0.9, plot=True, info=False)
        
    return {'r2': likeli_r2, 'shift': likeli_shift, 'power': likeli_power,
            'rex': likeli_rex, 'rey': likeli_rey, 'fit': likeli_fit, 'popt': likeli_popt}


def search_r2_plot(shifts: list, powers: list, r2: list, msg: str, save: bool = False):
    """
    Plots R2 results for shifts and powers.

    Args:
        shifts (list): Shift values.
        powers (list): Power values.
        r2 (list): R2 values.
        msg (str): Title for the plot.
        save (bool, optional): Save the plot. Defaults to False.
    """
        
    fig = plt.figure(figsize=(12,5), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(shifts,r2,'ro-')
    ax1.set_xlabel('shift')
    ax1.set_ylabel('$R^2$')
    ax1.grid(which='both')
    
    ax2.plot(powers,r2,'b^-')
    ax2.set_xlabel('n')
    ax2.set_ylabel('$R^2$')
    ax2.grid(which='both')
    
    fig.suptitle(msg)
    
    if save:
        filename = re_replace(msg)
        fig.savefig(f'{filename}.png', dpi=300)
        
    plt.show()
      
       
def multi_plots(search_lists: list, res_lists: list, title: str = 'Shift evaluation', 
                nrows: int = None, ncols: int = 3, save: bool = False, 
                para_inv: bool = False, plot_fig_step: int = 1):
    """
    Creates multiple subplots for fitting results.

    Args:
        search_lists (list): List of search parameters.
        res_lists (list): List of fitting results.
        title (str, optional): Title for the plots. Defaults to 'Shift evaluation'.
        nrows (int, optional): Number of rows in the subplot grid. Defaults to None.
        ncols (int, optional): Number of columns in the subplot grid. Defaults to 3.
        save (bool, optional): Save the plots. Defaults to False.
        para_inv (bool, optional): Whether the search parameter is inverse power. Defaults to False.
        plot_fig_step (int, optional): Step size for plotting figures. Defaults to 1.
    """
    all_data = len(search_lists)
    total = len(search_lists[::plot_fig_step])
    
    if nrows == None:
        nrows = (total + ncols -1)//ncols

    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.8,nrows*3.6), squeeze=False, tight_layout=True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*5), squeeze=False, tight_layout=True)
    
    fig.suptitle(title,y=1)
    
    ax_list=[]    
    for i in range(nrows):
        for j in range(ncols):
            ax_list.append(ax[i,j])

    for  idx, (i_sea, i_res) in enumerate(zip(search_lists[::plot_fig_step], res_lists[::plot_fig_step])): 
        
        p_para = i_res['popt']
        ax_list[idx].plot(i_res['rex'],i_res['rey'],'ro',label='data')
        ax_list[idx].plot(i_res['rex'],i_res['fit'],'b-',label='fit')
        ax_list[idx].grid(which='both')
        
        if para_inv:
            comment = f'No.{idx*plot_fig_step+1}/{all_data}\nshift: {i_res["popt"][1]:.2f}\nn: {i_sea:.2f}\n$R^2$: {i_res["r2"]:.3f}\n$R^{2}_{{bp}}$: {i_res["r2_bp"]:.3f}'
            ax_list[idx].set_xlabel('x')
            ax_list[idx].set_ylabel(f'y${{1/{i_sea:.1f}}}$')
        else:    
            comment = f'No.{idx*plot_fig_step+1}/{all_data}\nshift: {i_sea:.2f}\nn: {i_res["popt"][0]:.2f}\nslice: {i_res["popt"][1]:.2f}\n$R^2$: {i_res["r2"]:.3f}'
            ax_list[idx].set_xlabel('$ln{x}$')
            ax_list[idx].set_ylabel('$ln{y}$')
        ax_list[idx].legend(title=comment)
        
    if len(ax_list) != total:
        for ij in range(len(ax_list)-total):
            newi= ij + total
            ax_list[newi].axis("off")

    plt.tight_layout()
    
    if save:
        filename=re_replace(title)
        plt.savefig(f'{filename}.png', dpi=300)
        
    plt.show()