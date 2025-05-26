"""
Create: 2024/07/31

Shinjiro Yagyu

Licence: BSD 3-Clause License

1. upload dat files -> read dat file
2. fitting with 1/2, 1/3 power data
3. Zip download includes results image by png, all infomation by text and toml

Updata : 2025/05/26
Updates to Pfitlib and reader modules
Limits on maximum counts and maximum energy for ACdata

"""
import datetime
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
import toml
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from reader import datconv as dv
from pfitlib import loglog_fit as pf

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Flatten the dictionary
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ','.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

def plot_ac_inst_invanalysis(ac_inst,original_filename):
    xx = ac_inst.df["uvEnergy"].values
    yy = ac_inst.df["pyield"].values 
    
    pysi2 = pf.const_inv_power_fit(xx,yy,2)
    pysi3 = pf.const_inv_power_fit(xx,yy,3)
    
    # --- figure
    fig = plt.figure(figsize=(18,6), tight_layout=True)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    ax1.set_title(f'{ac_inst.metadata["sampleName"]}')    
    ax1.plot(xx, yy, color=cycle[0], marker="o",label='Data')
    ax1.set_xlabel('Energy [ev]')
    ax1.set_ylabel('Photoelectron yield [arb.unit]')
    ax1.legend() 
    ax1.grid()
     

    ax2.set_title('1/2 power') 
    ax2.plot(pysi2["rex"], pysi2["rey"], color=cycle[0], marker="o",label='Data')
    ax2.plot(pysi2["rex"], pysi2["fit"], color=cycle[1], linestyle = '-', 
             label=f'Fit\nR2: {pysi2["r2"]:.3f}\nthreshold: {pysi2["popt"][1]:.2f}\nslope: {pysi2["popt"][0]:.2f}')
    ax2.axvline(pysi2["popt"][1], color=cycle[1] )
    ax2.text(pysi2["popt"][1], np.max(pysi2["rey"])*0.2, f'{pysi2["popt"][1]:.2f}')
    
    if 0.49 < ac_inst.metadata["powerNumber"] < 0.51 and ~np.isnan(ac_inst.estimate_value['thresholdEnergy']):
        ax2.plot(pysi2["rex"], ac_inst.df["guideline"], color=cycle[2], linestyle = '-', 
                label=f'User\nthreshold: {ac_inst.estimate_value["thresholdEnergy"]:.2f}\nslope:{ac_inst.estimate_value["slope"]:.2f}')
        ax2.axvline(ac_inst.estimate_value["thresholdEnergy"], color=cycle[2] )
        ax2.text(ac_inst.estimate_value["thresholdEnergy"], np.max(pysi2["rey"])*0.1, f'{ac_inst.estimate_value["thresholdEnergy"]:.2f}')

    ax2.set_xlabel('Energy [eV]')
    ax2.set_ylabel('Photoelectron yield$^{1/2}$ [arb.unit]')
    ax2.legend()
    ax2.grid()
    
    ax3.set_title('1/3 power') 
    ax3.plot(pysi3["rex"], pysi3["rey"], color=cycle[0], marker="o",label='Data')
    ax3.plot(pysi3["rex"], pysi3["fit"], color=cycle[1], linestyle = '-', 
             label=f'Fit\nR2: {pysi3["r2"]:.3f}\nthreshold: {pysi3["popt"][1]:.2f}\nslope: {pysi3["popt"][0]:.2f}')
    ax3.axvline(pysi3["popt"][1], color=cycle[1] )
    ax3.text(pysi3["popt"][1], np.max(pysi3["rey"])*0.2, f'{pysi3["popt"][1]:.2f}')
    
    if 0.3 < ac_inst.metadata["powerNumber"] < 0.35 and ~np.isnan(ac_inst.estimate_value['thresholdEnergy']):
        ax3.plot(pysi3["rex"], ac_inst.df["guideline"], color=cycle[2], linestyle = '-',
                label=f'User\nthreshold: {ac_inst.estimate_value["thresholdEnergy"]:.2f}\nslope:{ac_inst.estimate_value["slope"]:.2f}')
        ax3.axvline(ac_inst.estimate_value["thresholdEnergy"], color=cycle[2] )
        ax3.text(ac_inst.estimate_value["thresholdEnergy"], np.max(pysi3["rey"])*0.1, f'{ac_inst.estimate_value["thresholdEnergy"]:.2f}')
    
    ax3.set_xlabel('Energy')
    ax3.set_ylabel('Photoelectron yield$^{1/3}$ [arb.unit]')
    ax3.legend() 
    ax3.grid()
    
   
    if pysi2["r2"] > pysi3["r2"]:
        likely_Ip =  pysi2["popt"][1]
        likely_slope =  pysi2["popt"][0]
        likely_power = '1/2'
    else:
        likely_Ip =  pysi3["popt"][1]
        likely_slope =  pysi3["popt"][0]
        likely_power = '1/3'
    
    # --- info    
    s_info=f'\
    {original_filename}\n\
    {ac_inst.metadata["sampleName"]}\n\
    1/2 R2:{pysi2["r2"]:.3f}, R2S:{pysi2["r2_bp"]:.3f}\n\
    Ip={pysi2["popt"][1]:.2f}, Slope={pysi2["popt"][0]:.2f}\n\
    1/3 R2:{pysi3["r2"]:.3f}, R2S:{pysi3["r2_bp"]:.3f}\n\
    Ip={pysi3["popt"][1]:.2f}, Slope={pysi3["popt"][0]:.2f}\n\
    User 1/n={ac_inst.metadata["powerNumber"]:.2f}\n\
    Ip={ac_inst.estimate_value["thresholdEnergy"]:.2f}, Slope={ac_inst.estimate_value["slope"]:.2f}\n\
    Auto likelihood: 1/n={likely_power}, Ip={likely_Ip:.2f}, Slope={likely_slope:.2f}'
    
    # --- info dict
    s_dict={'file':original_filename,
            'sample': ac_inst.metadata["sampleName"],
            'inv2':{
                'R2':pysi2["r2"],
                'R2bp':pysi2["r2_bp"],
                'Ip':pysi2["popt"][1],
                'Slope':pysi2["popt"][0],
                'Bg':pysi2["popt"][2]
                },
            'inv3':{
                'R2':pysi3["r2"],
                'R2bp':pysi3["r2_bp"],
                'Ip':pysi3["popt"][1],
                'Slope':pysi3["popt"][0],
                'Bg':pysi3["popt"][2]
                },
            'user':{
                'Ip':ac_inst.estimate_value["thresholdEnergy"],
                'Slope':ac_inst.estimate_value["slope"],
                'Bg':ac_inst.estimate_value["bg"]   
                },
            'likely':{
                'inv':str(likely_power),
                'Ip': likely_Ip, 
                'Slope': likely_slope 
                }
            }
    
    return fig, s_info, s_dict

def main():
    st.title("ACdat AutoFit")
    
    download_zip_file = st.empty()

    uploaded_files = st.file_uploader("dat files upload", accept_multiple_files=True, type=["dat"])

    infos = []
    dict_infos = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False) as f:
                fp = Path(f.name)
                fp.write_bytes(uploaded_file.getvalue())
                
                acdata = dv.AdvAcConv(f'{f.name}')
                acdata.convert()

            fp.unlink()
            
            fig, info, info_dict = plot_ac_inst_invanalysis(acdata, uploaded_file.name )   
            st.text('*-'*25)
            st.text(info)
            st.text('-*'*25)
    
            st.pyplot(fig)
            infos.append((fig, info, info_dict, uploaded_file.name))
            dict_infos.append(info_dict)

        def create_zip():
            in_memory = BytesIO()
            with zipfile.ZipFile(in_memory, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fig, info, dict_info, name in infos:
                    img_bytes = BytesIO()
                    toml_bytes = StringIO()
                    
                    fig.savefig(img_bytes, format='png')
                    img_bytes.seek(0)
                    zf.writestr(f"{Path(name).stem}.png", img_bytes.read())
                    zf.writestr(f"{Path(name).stem}.txt", info)
                    # convert dict to toml
                    toml.dump(dict_info,toml_bytes)
                    toml_bytes.seek(0)    
                    zf.writestr(f"{Path(name).stem}.toml", toml_bytes.read())
                
                # make Summary Data    
                csv_bytes = BytesIO()
                # nest dict to flat
                flat_list = [flatten_dict(d) for d in dict_infos]
                df_dict = pd.DataFrame(flat_list)
                df_dict.to_csv(csv_bytes, index=False, encoding='utf-8')
                csv_bytes.seek(0)
                t_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                zf.writestr(f'Summary_{t_now}.csv', csv_bytes.read())
                
            in_memory.seek(0)
            
            return in_memory

        zip_buffer = create_zip()
        download_zip_file.download_button(
            label="Download (zip)",
            data=zip_buffer,
            file_name='files.zip',
            mime='application/zip'
        )

if __name__ == "__main__":
    main()
    
    # streamlit run .\ACAutoFit.py
