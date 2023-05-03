
from pathlib import Path
from tempfile import NamedTemporaryFile
import io
# import zipfile

import streamlit as st
import matplotlib.pyplot as plt
import japanize_matplotlib

from acdatconv import datconv as dv
from acdatconv import datlib as dlib

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----
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
    if params_init is None:
        params_init = np.array([1, 4.5, 0.0])

    result = minimize(absolute_error, params_init, args=(x, y))
    # result = minimize(least_squares_error, params_init, args=(x, y))

    Nor_opt, Ip_opt, Bg_opt = result.x

    return Nor_opt, Ip_opt, Bg_opt


# -----

st.title('AC Dat File Converter')

# Fileの拡張子をチェックしてくれる
uploaded_file = st.file_uploader("dat file upload", type='dat')


if uploaded_file is not None:
    file_name = uploaded_file.name
    save_name = file_name.split('.')[0]

    
    with NamedTemporaryFile(delete=False) as f:
        fp = Path(f.name)
        fp.write_bytes(uploaded_file.getvalue())
        
        acdata = dv.AcConv(f'{f.name}')
        acdata.convert()
        
    # ファイルを削除  
    fp.unlink()
    # st.write(acdata.estimate_value)
    xx = acdata.df["uvEnergy"].values
    yy = acdata.df["pyield"].values  
    fit_param2 = fit_m_relu(xx,np.power(yy,1/2))
    fit_param3 = fit_m_relu(xx,np.power(yy,2/5))
    fit_param4 = fit_m_relu(xx,np.power(yy,1/3))
    y_fit2 = m_relu(xx, *fit_param2)
    y_fit3 = m_relu(xx, *fit_param3)
    y_fit4 = m_relu(xx, *fit_param4)

    
    # 3つのグラフを縦に並べる
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12)) # (wide ,high)

    ax1.set_title(f'{acdata.metadata["sampleName"]}')
    ax1.plot(xx,acdata.df["npyield"],'ro-',label='Data')
    ax1.plot(xx,acdata.df["guideline"],'g-',label=f'Estimate\n {acdata.metadata["thresholdEnergy"]:.2f} eV')
    ax1.plot(xx,y_fit2,'b-',label=f' Auto Estimate\n {fit_param2[1]:.2f} eV')
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Energy [eV]')
    ax1.set_ylabel(f'Intensity^{acdata.metadata["powerNumber"]:.2f}')

    ax2.plot(xx, np.power(yy,2/5), 'ro-',label='Data')
    ax2.plot(xx, y_fit3,'b-',label=f'Auto Estimate\n {fit_param3[1]:.2f} eV')
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel('Energy [eV]')
    ax2.set_ylabel(f'Intensity^2/5')
    
    ax3.plot(xx, np.power(yy,1/3), 'ro-',label='Data')
    ax3.plot(xx, y_fit4,'b-',label=f'Auto Estimate\n {fit_param4[1]:.2f} eV')
    ax3.legend()
    ax3.grid()
    ax3.set_xlabel('Energy [eV]')
    ax3.set_ylabel(f'Intensity^1/3')

    # グラフのタイトルと余白の調整
    # fig.suptitle('Two Plots', fontsize=16)
    # fig.tight_layout(pad=2)
    fig.tight_layout()


    
    # メモリに保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    
    st.pyplot(fig)
    
    csv = acdata.df[["uvEnergy","pyield","npyield",	"nayield","guideline"]].to_csv(index=False)
    json = acdata.json

    # # ボタンを横に並べるため
    # col1, col2, col3 = st.columns([1,1,1])
    
    # with col1:
    #     st.download_button(label='Download csv data', 
    #                     data=csv, 
    #                     file_name=f'{save_name}.csv',
    #                     mime='text/csv',
    #                     )
    # with col2:
    #     st.download_button(label="Download image",
    #                     data=img,
    #                     file_name=f'{save_name}.png',
    #                     mime="image/png"
    #                     )
    # with col3:    
    #     st.download_button(label ="Download json",
    #                     data=json,
    #                     file_name=f'{save_name}.json',
    #                     mime="application/json",
    #                     )
