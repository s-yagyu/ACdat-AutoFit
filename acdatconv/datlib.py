import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
from pandas.io.json import json_normalize

from acdatconv import datconv as dv

def dat_list_make(data_path, figout=True, out_file_name=None):
    """Creating a metadata list of data in a folder
    Args:
        data_path (str or pathlib): Data foldar path 
        figout (bool, optional): Figure show. Defaults to True.
        out_file_name (str, optional):Output filename. ex:'test.xlsx'.  Defaults to None.

    output: Excel file containing dat metadata
    
    Returns:
        dataframe
        
    """
    f_path = Path(data_path)
    tg_list = list(f_path.glob('*.dat'))
    
    meta_list=[]
    meta_wo_list=[]
    for i,fl in enumerate(tg_list):
        
        try:
            acdata = dv.AcConv(fl)
            acdata.convert()
            meta_ = acdata.metadata
            meta_wo = acdata.metadata_wo_calc
            plot_data = acdata.calcdata
            meta_list.append(meta_)
            meta_wo_list.append(meta_wo)
       
        except:
            print(f'file error: {fl.name}')
        
        if figout:  
            make_plot(plot_data,meta_wo)

    df_meta = json_normalize(meta_list)
    df_meta_wo = json_normalize(meta_wo_list)
 
        
    if out_file_name is None:
        out_file_name = 'dat_list.xlsx'
        
    sp = out_file_name.split('.')
    out_file_name_wo = f'{sp[-2]}_wo.{sp[-1]}'
 
    print(f'output file with spectrum :{out_file_name}')
    print(f'output file without spectrum :{out_file_name_wo}')
    
    df_meta.to_excel(out_file_name,index=False)
    df_meta_wo.to_excel(out_file_name_wo,index=False)
    
    return df_meta

def make_plot(plotdata,metadata):
    
    width_u = 5.2
    height_u = 4
    nrows=1
    ncols=1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*width_u,nrows*height_u), squeeze=False, tight_layout=True)
    
    ax[0,0].plot(plotdata["uvEnergy"], plotdata["nayield"],'ro-',label='Data')
    ax[0,0].plot(plotdata["uvEnergy"], plotdata["guideline"],'b-',label='Guide')
    ax[0,0].set_title(f" {metadata['file_name']}")
    ax[0,0].set_xlabel('Energy [eV]')
    ax[0,0].set_ylabel(f"Yield^{metadata['powerNumber']:.2f}")
    ax[0,0].grid()
    ax[0,0].legend(title=f"Power: {metadata['uvIntensity59']:.2f}nw\nth: {metadata['thresholdEnergy']:.2f}eV\nslop: {metadata['slope']:.2f}",
                   loc='upper left')
    
    # fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()
    
if __name__ =="__ main__":
    # set folder
    data_folder=r"RDEdevelop"
    f_path = Path(data_folder)
    # tg_list = list(f_path.glob('*.dat'))
    # print(tg_list)
    
    df = dat_list_make(data_path=f_path, figout=True, out_file_name="develop.xlsx")