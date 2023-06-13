
from pathlib import Path
from tempfile import NamedTemporaryFile
import io
# import zipfile

import numpy as np

import matplotlib.pyplot as plt
import japanize_matplotlib

import streamlit as st
from scipy.optimize import minimize

from acdatconv import datconv as dv
from acdatconv import datlib as dlib

from pfit import power_fit as pof

# -----

st.title('AC Dat Auto Estimation')

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
    
    pysA2 = pof.PfAnalysis(xx,yy)
    pysA2.estimate(power_num=2,info=False,ini_para=None,retry_r2=0.9)
        
    pysA3 = pof.PfAnalysis(xx,yy)
    pysA3.estimate(power_num=3,info=False,ini_para=None,retry_r2=0.9)
        
    
    fig = plt.figure(figsize=(18,6), tight_layout=True)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    ax1 = pof.plot3_pw_ax(xx,yy,
                          m_xdata=None, m_ydata=None, 
                          n_xdata=None, n_ydata=None,
                          breakpoints=None,
                          title=f'{acdata.metadata["sampleName"]}', axi=ax1)

    
    ax2 = pof.plot3_pw_ax(pysA2.res_pof["rex"], pysA2.res_pof["rey"],
                            m_xdata=pysA2.res_pof["rex"],m_ydata=pysA2.res_pof["fit"],
                            n_xdata=pysA2.res_pof["rex"], n_ydata= acdata.df["guideline"], 
                            breakpoints=[pysA2.res_pof["popt"][1],acdata.metadata["thresholdEnergy"]], 
                            title=f'1/n=1/2 R2:{pysA2.res_pof["r2"]:.3f}, R2S:{pysA2.res_pof["r2_bp"]:.3f}',  
                            #  breakpoints=None, 
                            #  title='', 
                            axi=ax2)
    
    ax3 = pof.plot3_pw_ax(pysA3.res_pof["rex"], pysA3.res_pof["rey"],
                            pysA3.res_pof["rex"], pysA3.res_pof["fit"],
                            n_xdata=None, n_ydata=None,
                            breakpoints=[pysA3.res_pof["popt"][1],],
                            title=f'1/3 R2:{pysA3.res_pof["r2"]:.3f}, R2S:{pysA3.res_pof["r2_bp"]:.3f}',
                            #  breakpoints=None,
                            #  title='',
                            axi=ax3)
            
    
    if pysA2.res_pof["r2"] > pysA3.res_pof["r2"]:
        likeli_Ip = pysA2.res_pof["popt"][1]
        likeli_power = '1/2'
    else:
        likeli_Ip = pysA3.res_pof["popt"][1]
        likeli_power = '1/3'
        
    s_info=f'{file_name}\n\
    {acdata.metadata["sampleName"]}\n\
    1/2 R2:{pysA2.res_pof["r2"]:.3f}, R2S:{pysA2.res_pof["r2_bp"]:.3f}\n\
    Ip={pysA2.res_pof["popt"][1]:.2f}\n\
    1/3 R2:{pysA3.res_pof["r2"]:.3f}, R2S:{pysA3.res_pof["r2_bp"]:.3f}\n\
    Ip={pysA3.res_pof["popt"][1]:.2f}\n\
    User 1/n={acdata.metadata["powerNumber"]:.2f}, Ip={acdata.estimate_value["thresholdEnergy"]:.2f}\n\
    Auto likelihood: 1/n={likeli_power}, Ip={likeli_Ip:.2f}'
    
    # メモリに保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    
    st.text('*-'*20)
    st.text(s_info)
    st.text('-*'*20)
    
    st.pyplot(fig)
    
    st.download_button(label="Download image",
                        data=img,
                        file_name=f'{save_name}.png',
                        mime="image/png"
                        )
    
    # csv = acdata.df[["uvEnergy","pyield","npyield",	"nayield","guideline"]].to_csv(index=False)
    # json = acdata.json

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
