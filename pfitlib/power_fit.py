"""
Inv-Power-Plot, Shift-log-log-plot
analysis module

"""				
import time
from datetime import datetime
import re
from pathlib import Path

import scipy as sp
import scipy.optimize as optimize
from scipy.optimize import curve_fit, minimize

# from sklearn.metrics import r2_score
# # from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter('ignore')
# warnings.resetwarnings()
# warnings.simplefilter('ignore', FutureWarning)

# Note: figure size
# 黄金比 1：1.62 wdth:height = 4.86:3
# 白銀比 1：1.43  wdth:height = 4.3:3 
# figsize=(width,height) wdth:height= 4:3 
# (row,col)->figsize(col*4,row*3): (3,4)->figsize(16,9)
#  w_in, h_in = plt.rcParams[cParams["figure.figsize"]

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

class InvAnalysis():
    # single inv_n analysis
    
    def __init__(self,xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata
        
    def fit(self,
            power_num=2,
            ini_para=None, 
            retry_r2=0.9, min_error='mae',zero_replace=False,
            info=False, plot=False):
        
        # 1/n Power
        self.s_res_inv = const_inv_power_fit(self.xdata,
                                                self.ydata,
                                                power_num,
                                                ini_params=ini_para, 
                                                retry_r2=retry_r2,
                                                min_error= min_error,
                                                zero_replace=zero_replace,
                                                plot=plot, 
                                                info=info)
class DlogAnalysis(InvAnalysis):
    # dual logarism plot analysis
    
    def shift_estimate_by_power_scan(self, search_range=None, 
                                     zero_replace=False,
                                     min_error='mae', 
                                     likely_evaluation='r2',
                                     info=False, plot=True,
                                     plot_fig_step=1):
        # 1/n Power
        self.res_inv = xpower_search(self.xdata, self.ydata, 
                                        search_range=search_range,
                                        r2_plot=plot, process_plot=plot,
                                        min_error=min_error,
                                        zero_replace=zero_replace,
                                        likely_evaluation =likely_evaluation,
                                        info=info, 
                                        every_plot=False, 
                                        plot_save=False,
                                        plot_fig_step=plot_fig_step)
        
        # res_inv:keys
        # ['r2','shift', 'power', 'rex', 'rey', 'fit','popt']
        
    def power_estimate_by_shift_scan(self, search_range, 
                                       fit_type='weight', 
                                       bg_num=3,
                                       lim_val=0.5, 
                                       min_error='mae',
                                       info=True, plot=True,
                                       plot_fig_step=1):
        # Shifted-log-log algorithm
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
        
        # res_dlog:keys
        # ['r2','shift', 'power', 'rex','rey','fit']
 
        
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
    """Perform absolute error fitting using minimization.

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
    """Compute the absolute error between model predictions and actual data.

    Args:
        params (np.ndarray): Parameters (slope, intercept).
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.

    Returns:
        float: Sum of absolute errors.
    """
    
    y_pred = line_func(xdata, *params)
    return np.sum(np.abs(ydata - y_pred))

def y_intercept(Ip, Nor, Bg):
    # for ReLu
    intercept = Bg - Nor*Ip
    return intercept

#------ReLU fitting
def relu_func(x, Nor, Ip, Bg):
    u = (x - Ip)
    return Nor * u * (u > 0.0) + Bg

def _relu_absolute_error(params, x, y):
    y_pred = relu_func(x, *params)
    return np.sum(np.abs(y - y_pred))

def _relu_least_squares_error(params, x, y):
    y_pred = relu_func(x, *params)
    return np.sum((y - y_pred) ** 2)

def relu_fit(x, y, params_init=None, min_error='mae'):
    # min_error:(str,option): 
    # 'mae':absolute error, 'mse':squares error, defalut is 'mae'
    if params_init is None:
        params_init = np.array([1, 4.5, 0.0])

    if min_error == 'mae':
        result = minimize(_relu_absolute_error, params_init, args=(x, y))
    
    else:   
        result = minimize(_relu_least_squares_error, params_init, args=(x, y))
    
    Nor_opt, Ip_opt, Bg_opt = result.x
    return Nor_opt, Ip_opt, Bg_opt

# -----
def get_nearest_value(lst: list, num: float):
    """
    Returns the element from the input list that is closest to the given number.
    
    Args:
        lst: A list of numbers.
        num: A float number.
    
    Returns:
        The index of the element in the list that is closest to the given number.
    """
    # Calculate the difference between each element in the list and the target number,
    # and get the index of the minimum difference
    idx = np.abs(np.asarray(lst) - num).argmin()
    
    return idx

def nan_inf_rm(xdata,ydata,zero_replace=False,info=False):
    """remove nan and inf values
        If you take Log
            Negative value: Nan for not defined
            0: Negative infinity because it is not defined
            If these values are included, remove them (Index) to avoid an error

    Args:
        xdata (ndarray or list): xdata
        ydata (ndarrayor list): ydata
        zero_replace(bool): True: Nan -> 0 replace  False:Nan remove. default False
        info (bool): show infomation
        
    Returns:
        trimmed data (ndarray): re_xdata, re_ydata
    """
    # inf -> nan
    # nan_inf_rm(np.log(xf),np.log(yf))
    xarray = np.array(xdata)
    yarray = np.array(ydata)
    tr_inf_x = np.where(np.isinf(xarray),np.nan,xarray)
    tr_inf_y = np.where(np.isinf(yarray),np.nan,yarray)
    
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
        print(f'number of Nan of (x, y) : ({np.count_nonzero(np.isnan(tr_inf_x))}, {np.count_nonzero(np.isnan(tr_inf_y))})')
        print(f'Original shape x, y: {xarray.shape}, {yarray.shape}')
        print(f'output shape x, y: {re_xdata.shape}, {re_ydata.shape}')
        
    return re_xdata, re_ydata

def r2n_score(y_data, y_pred):
    """
    Calculate the coefficient of determination, R^2, for the given true and predicted values.

    The R^2 score, also known as the coefficient of determination, is a statistical measure that 
    represents the proportion of the variance for a dependent variable that's explained by an 
    independent variable or variables in a regression model.

    Args:
        y_data (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: The R^2 score.

    """
    # Residual Sum of Squares
    rss = np.sum((y_data - y_pred)**2)

    # Total Sum of Squares
    y_mean = np.mean(y_data)
    tss = np.sum((y_data - y_mean)**2)

    # Coefficient of Determination
    r2 = 1 - rss / tss

    return r2

#---- plot
def re_replace(text):
    """Remove special symbols with regular expressions 

    Args:
        text (str): text
    Returns:
        str: Text with special symbols removed
    Examples:
        text = '4 inch $\phi$=0'
        re_replace(text)
        >>> '4_inch_phi0
    Ref:
        https://qiita.com/ganyariya/items/42fc0ed3dcebecb6b117 
    """
    # code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')

    code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~]')
    cleaned_text = code_regex.sub('', text).replace(' ', '_')
    # print(cleaned_text)

    return cleaned_text

def search_r2_plot(shifts,powers,r2,msg,save=False):
    """shift vs r2, power vs r2 plot

    Args:
        shifts (array like): shift data
        powers (array like): power data
        r2 (array like): r2 data
        msg (str): graph title
        save (bool, optional): figure save. Defaults to False.
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
    
    
def inv_power_plot(xdata, ydata, shift, power):
    # power 2-> 1/2 plot
    #---
    inv_power = 1/power
    fig = plt.figure(figsize=(12,5), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    ax1.plot(xdata,ydata,'ro-',label='Data')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(which='both')
    
    ax2.plot(xdata,np.power(ydata,1/power),'b^-',label='Data')
    ax2.set_xlabel('x')
    ax2.set_ylabel('$y^{{{1/power}}}$')
    ax2.legend(title=f'Shift: {shift:.2f}\nn: {power:.2f}')
    ax2.grid(which='both')
        
    plt.show()
    
def inv_power_plot_ax(xdata, ydata, yfitdata, n_power, shift, axi=None):
    # power 2-> 1/2 plot
    #---
    if axi is None:
        fig_, ax_ = plt.subplots()
    else:
        ax_ = axi
        
    # inv_power = 1/n_power
    # fig = plt.figure(figsize=(12,5), tight_layout=True)
    # ax_ = fig.add_subplot(1,1,1)
    
    ax_.plot(xdata,np.power(ydata,1/n_power),'ro',label='Data')
    ax_.plot(xdata, yfitdata,'b-', label='Fit')
    ax_.set_xlabel('x')
    ax_.set_ylabel('$y^{{{1/n_power}}}$')
    ax_.legend(title=f'Shift: {shift:.2f}\nn: {n_power:.2f}')
    ax_.grid(which='both')
    
    if axi is None:    
        plt.show()
        return 
    else:
        return ax_

def plot_ax(xdata, ydata, 
            m_xdata, m_ydata, 
            breakpoints=None, 
            lgtitle=None, title='', 
            axi=None):
    
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

def plot3_pw_ax(xdata, ydata, 
                m_xdata=None, m_ydata=None, 
                n_xdata=None, n_ydata=None, 
                breakpoints=None, title='', axi=None):
    
    if axi is None:
        fig_ = plt.figure()
        ax_ = fig_.add_subplot(111)
    else:
        ax_ = axi
                  
    ax_.set_title(title)    
    ax_.plot(xdata, ydata, color=cycle[0], marker="o",label='Data')
    
    if m_xdata is not None:
        ax_.plot(m_xdata, m_ydata, color=cycle[1], linestyle = '-', label='Fit')
    
    if n_xdata is not None:
        ax_.plot(n_xdata, n_ydata, color=cycle[2], linestyle = '-', label='User')
        
    
    if breakpoints is not None:
        for i, bp in enumerate(breakpoints):
            ax_.axvline(bp,color=cycle[i+1] )
            ax_.text(bp,np.max(ydata)*0.1*(i+5), f'{bp:.2f}')
        
    ax_.grid()
    ax_.legend()     
    
    if axi is None:
        plt.show()
        return 
    
    return ax_   
   
def multi_plots(search_lists, res_lists, title='Shift evaluation', 
                nrows=None, ncols=3, 
                save=False, para_inv=False, 
                plot_fig_step=1):
    """Multiplot
    para_inv =True -> search_list= Power number
    
    Args:
        search_lists (array like): shift data -> para_inv=False
                                    power data -> para_inv=True
        res_lists (array like): if shift data, power data. if power data, shift data
        title (str, optional): graph title. Defaults to 'Shift evaluation'.
        nrows (int, optional): rows. Defaults to None.
        ncols (int, optional): cols. Defaults to 4.
        save (bool, optional): save. Defaults to False.
        para_inv (bool, optional): para_inv =True -> search_list= Power number. Defaults to False.
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

    
# ----
def const_inv_power_fit(xdata, ydata, power_num=2, ini_params=None,
                        min_error='mae', retry_r2=0.9, 
                        negative=False, zero_replace=False,
                        plot=False, info=False):
    """
    Perform a rectified linear unit (ReLU) and 
    mean absolute error (MAE) fitting on data^(1/power_num).
    
    Args:
        xdata (ndarray): Array of x-values.
        ydata (ndarray): Array of y-values.
        power_num (float, optional): The power number. Defaults to 2.
        ini_params (list, optional): Initial parameters for fitting (Nor, Ip, Bg). 
                                     Defaults to [1, 4.5, 0.0].
        min_error (str, optional): Error metric to minimize: 'mae' for mean absolute error or 'mse' for mean squared error. 
                                   Default is 'mae'.
        retry_r2 (float, optional): R^2 lower limit for retrying the fitting. Defaults to 0.9.
        negative (bool, optional): Handle negative values by taking the absolute value before applying the power, 
                                   and retaining the sign. Defaults to False.
        zero_replace (bool, optional): Replace NaN values with zero. Defaults to False.
        plot (bool, optional): Plot the data and fitting results. Defaults to False.
        info (bool, optional): Print additional information. Defaults to False.
    
    Returns:
        dict: Dictionary containing the fitted results and additional information:
            - 'rex': Transformed x-data
            - 'rey': Transformed y-data
            - 'fit': Fitted values
            - 'popt': Fitted parameters
            - 'r2': R^2 score
            - 'r2_bp': R^2 score after break point
            - 'error_range': Error range
            - 'cv': Coefficient of variation
    """

    if ini_params is None:
        bg = np.nanmean(ydata[:5])  # Background estimation
        # nor = np.max(ydata)/10
        nor = 1  # Normalization factor estimation
        # ip = np.percentile(xdata,50)
        ip = 5  # Default initial breakpoint
        ini_params = (nor, ip, bg) 

    def none_negative_power(ydata, pw):
        """Apply a power transformation without resulting in NaN for negative values."""
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

    # TODO:Consider whether this item is necessary for real data 
    # (if fitting does not work, try changing the initial parameters and fitting).
    if r2 < retry_r2:
        bg = np.mean(re_ydata[:5])
        nor = np.max(re_ydata) / 2
        for p in [45, 55, 40, 60, 35, 65]:
            ip = np.percentile(re_xdata, p)
            res_params = relu_fit(re_xdata, re_ydata, (nor, ip, bg), min_error)
            re_fit = relu_func(re_xdata, *res_params)
            idx = get_nearest_value(re_xdata, res_params[1])
            try:
                r2 = r2n_score(re_ydata, re_fit)
                r2_bp = r2n_score(re_ydata[idx:], re_fit[idx:])
            except:
                r2, r2_bp = -1, -1
                
            if r2 > retry_r2 + 0.02:
                break

    if info:
        print(f'power(n): {power_num:.2f}, 1/power(1/n):{1/power_num:.2f}')
        print(f'$R^2$: {r2:.2f}')
        print(f'$R^2$ > threshold {r2_bp:.2f}')
        print(f'params(Nor,Ip,bg): {res_params}')     
         
    if plot:
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
                      title='',
                      axi=ax2)
        ax2.set_xlabel('x')
        ax2.set_ylabel(f'y$^{{1/{power_num:.1f}}}$')
        plt.show()
         
    return {'rex': re_xdata, 'rey': re_ydata, 'fit': re_fit, 
            'n': power_num, 'popt': res_params, 'r2': r2, 'r2_bp': r2_bp}


# --- log-log
def val_rm(xdata, ydata, lim_val, info=True):
    """
    Remove values smaller than the specified value along the x-axis.

    Args:
        xdata (ndarray): Array of x-values.
        ydata (ndarray): Array of y-values.
        lim_val (float): The specified threshold value.
                         Value before taking the log of the x-axis (0.1, 0.5, etc.)
        info (bool, optional): Display information about the operation. Defaults to True.

    Returns:
        tuple: Filtered x and y data arrays.
        
    Note:
        Delete values ​​smaller than the specified value on the x-axis.
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


def log_log_fit(xdata, ydata, bg_num=None, lim_val=0.36, fit_type='mae', 
                comment='', info=True, plot=True):
    """
    Estimate power and shift using log-log fitting.

    Args:
        xdata (ndarray): Array of x-values.
        ydata (ndarray): Array of y-values.
        bg_num (int, optional): Number of points for background averaging. Defaults to None.
        lim_val (float, optional): Threshold value for filtering x-axis values. Defaults to 0.36.
        fit_type (str, optional): Type of fitting. 'mae' for mean absolute error, 
                                  'weight' for weighted fitting, 'mse' for mean squared error. Defaults to 'mae'.
        comment (str, optional): Additional comment to print. Defaults to ''.
        info (bool, optional): Display information about the fitting. Defaults to True.
        plot (bool, optional): Plot the fitting results. Defaults to True.

    Returns:
        dict: Dictionary containing fitting results:
            - 'popt': Optimized parameters [slope, intercept]
            - 'r2': R^2 score
            - 'rex': Filtered x data
            - 'rey': Filtered y data
            - 'fit': Fitted y values
    """
    
    bg_rm_ydata = ydata
    if bg_num is not None: #remove background
        ybg = np.nanmean(ydata[:bg_num])
        bg_rm_ydata = ydata - ybg
        
    re_xdata, re_ydata = nan_inf_rm(np.log(xdata), np.log(bg_rm_ydata),
                                    zero_replace=False, info=info)
    
    re_xdata, re_ydata = val_rm(re_xdata, re_ydata, lim_val, info=info)
    e_re_xdata = np.exp(re_xdata)
    
    r_popt = [0, 0]
    if len(re_xdata) >= 5 and len(re_ydata) >= 5: # Fittigする点が5つ以上
        try:
            if fit_type == 'mae':
                r_popt_ = optimize.minimize(_line_absolute_error, (1, 1), args=(re_xdata, re_ydata))
                r_popt = r_popt_.x
            elif fit_type == 'weight':
                r_popt, _ = curve_fit(line_func, re_xdata, re_ydata, p0=(1, 1), sigma=1/re_xdata)
            else:
                r_popt, _ = curve_fit(line_func, re_xdata, re_ydata, p0=(1, 1))
                
            fity = line_func(re_xdata, *r_popt)
            r2 = r2n_score(re_ydata, fity)
            
        except:
            r_popt[0], r_popt[1] = np.nan, np.nan
            r2, fity = np.nan, np.nan

        if info:
            print(comment)
            print(f'Slope: {r_popt[0]:.3f}, Intercept: {r_popt[1]:.3f}, $R^2$: {r2:.3f}')
        
        if plot:
            plt.plot(re_xdata, re_ydata, 'ro', label='data')
            plt.plot(np.log(e_re_xdata), line_func(np.log(e_re_xdata), *r_popt), 'b-', label='fit')
            plt.grid(True)
            plt.legend(title=f'Slope: {r_popt[0]:.3f}\nIntercept: {r_popt[1]:.3f}\n$R^2$: {r2:.3f}')
            plt.show()
    else:
        r_popt[0], r_popt[1] = np.nan, np.nan
        r2 = np.nan
        re_xdata, re_ydata, fity = np.nan, np.nan, np.nan
        
    return {'popt': r_popt, 'r2': r2, 'rex': re_xdata, 'rey': re_ydata, 'fit': fity}


def create_search_range(center, range_, step):
    """
    Create a search range.

    Args:
        center (float): Estimated center position.
        range_ (float): Range to search.
        step (float): Step size within the range.

    Returns:
        ndarray: Array of search range values.

    Example:
        s_range = create_search_range(center=5, range_=1, step=0.2)
        >>> array([4. , 4.2, 4.4, 4.6, 4.8, 5. , 5.2, 5.4, 5.6, 5.8, 6. ])
    """
    return np.arange(center - range_, center + range_ + step, step)

def xshift_search(xdata, ydata, search_range,
                  bg_num=None, lim_val=0.36, fit_type='mae',
                  r2_plot=True, process_plot=True, 
                  min_error='mae', info=False, every_plot=False, plot_save=False,
                  plot_fig_step=1):
    """
    Perform shift and power estimation using a specified shift list.
    
    Args:
        xdata (ndarray): Array of x-values.
        ydata (ndarray): Array of y-values.
        search_range (list or tuple): List of shift values to evaluate.
        bg_num (int, optional): Number of points for background averaging. Defaults to None.
        lim_val (float, optional): Threshold value for filtering x-axis values. Defaults to 0.36.
        fit_type (str, optional): Type of fitting. 
                                  'mae' for mean absolute error, 'weight' for weighted fitting, 
                                  'mse' for mean squared error. Default is 'mae'.
        r2_plot (bool, optional): Display result plots for shift vs. R2 and power vs. R2. Defaults to True.
        process_plot (bool, optional): Display the fitting process plots. Defaults to True.
        min_error (str, optional): Minimum error type. Default is 'mae'.
        info (bool, optional): Print additional information. Defaults to False.
        every_plot (bool, optional): Plot each fitting process individually. Defaults to False.
        plot_save (bool, optional): Save the plots. Defaults to False.
        plot_fig_step (int, optional): Step size for figure plot. Defaults to 1.

    Returns:
        dict: A dictionary containing the fitting results:
              - 'r2': The highest R2 value.
              - 'shift': The estimated shift.
              - 'power': The estimated power.
              - 'rex': The transformed x-data.
              - 'rey': The transformed y-data.
              - 'fit': The fitted values.

    Example:
        res_data = xshift_search(xdata, ydata, search_range=[3, 3.5, 4, 4.5, 5],
                                 bg_num=None, lim_val=0.36, fit_type='mae',
                                 r2_plot=True, process_plot=True, plot_save=False)
    """
    power = []
    r2 = []
    res_list = []
    applied_search_range = []
    
    for shift in search_range:
        res_para = log_log_fit(xdata - shift, ydata, bg_num=bg_num,
                               lim_val=lim_val, fit_type=fit_type, 
                               comment=f'shift: {shift:.2f}', info=info, plot=every_plot)
        
        if not np.isnan(res_para['r2']):
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


def xpower_search(xdata, ydata, search_range=None,
                  r2_plot=True, process_plot=True,
                  min_error='mae', zero_replace=False,
                  likely_evaluation='r2', info=True, every_plot=False, plot_save=False,
                  plot_fig_step=1):
    """
    Estimate shift thresholds while varying the power.
    
    Args:
        xdata (ndarray): Array of x-values.
        ydata (ndarray): Array of y-values.
        search_range (list or tuple, optional): Power range to evaluate. 
                                                Defaults to [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5].
        r2_plot (bool, optional): Display result plots for slope, shift, and R2. Defaults to True.
        process_plot (bool, optional): Display plots for each trial. Defaults to True.
        likely_evaluation (str, optional): Evaluation method. 'r2' or 'r2_bp'. Defaults to 'r2'.
                                           'r2' evaluates all regions, 'r2_bp' evaluates after break point.
        min_error (str, optional): Minimum error type. Defaults to 'mae'.
        zero_replace (bool, optional): Replace NaNs with zero. Defaults to False.
        info (bool, optional): Print additional information. Defaults to True.
        every_plot (bool, optional): Plot each fitting process individually. Defaults to False.
        plot_save (bool, optional): Save the plots. Defaults to False.
        plot_fig_step (int, optional): Step size for figure plot. Defaults to 1.

    Returns:
        dict: A dictionary containing the fitting results:
              - 'r2': The highest R2 value.
              - 'shift': The estimated shift.
              - 'power': The estimated power.
              - 'rex': The transformed x-data.
              - 'rey': The transformed y-data.
              - 'fit': The fitted values.
              - 'popt': The optimized fitting parameters.

    Examples:
        res_data = xpower_search(xdata, ydata, search_range=[2, 3, 4, 5],
                                 r2_plot=True, process_plot=True, plot_save=False)
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
        
        if not np.isnan(res_para['r2']):
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
