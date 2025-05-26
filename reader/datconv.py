"""
AC Series Data Converter

"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getEncode(filepath):
    """Automatically determines the encoding of a file.
     If the file name contains Japanese, the encoding method may be Shift-jis.
     Function that returns the encoding method.

    Args:
        filepath (str): fil path and name

    Returns:
        encode type(str): encode type
    """
    encodings = ["iso-2022-jp", "euc-jp", "shift_jis", "utf-8"]
    for encoding in encodings:
        try:
            with open(filepath, encoding=encoding) as file:
                file.read()
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not determine the encoding of file {filepath}.")


class AcConv():
    """Extraction of measurement metadata data from the AC.dat file.
    
    Args:
        file_name (str): .dat filename

    Example : 
        >>> file_name = "./datafile.dat"
        >>> ac_converter = AcConverter(file_name)
        >>> ac_converter.convert()
        >>>
        >>> print("metadata")
        >>> print(ac_converter.metadata)
        >>> print(ac_converter.metadata.keys())
        >>> # Keys:
        >>> # - Original keys (From Dat File)
        >>> #     "fileType", "deadTime", "countingTime", "powerNumber",
        >>> #     "anodeVoltage", "step", "model", "yAxisMaximum", "startEnergy",
        >>> #     "finishEnergy", "flagDifDataGroundLevel", "bgCountingRate", "measureDate", "sampleName",
        >>> #     "uvIntensity59", "targetUv", "nameLightCorrection", "sensitivity1", "sensitivity2",
        >>> # - Calculated keys (list type)
        >>> #     "uvEnergy", "countingCorrection", "photonCorrection", "pyield", "npyield", "nayield", "guideline",
        >>> #     "countingRate", "flGrandLevel", "flRegLevel", "uvIntensity",
        >>> # - Manually estimated keys
        >>> #     "thresholdEnergy", "slope", "yslice", "bg",
        >>> # - Filename key
        >>> #     "file_name"
        >>>
        >>> print(ac_converter.metadata_wo_calc)
        >>> print(ac_converter.metadata_wo_calc.keys())
        >>> # Calculated keys are excluded.
        >>> # dict_keys([
        >>> #     'fileType', 'deadTime', 'countingTime', 'powerNumber',
        >>> #     'anodeVoltage', 'step', 'model', 'yAxisMaximum',
        >>> #     'startEnergy', 'finishEnergy', 'flagDifDataGroundLevel',
        >>> #     'bgCountingRate', 'measureDate', 'sampleName', 'uvIntensity59',
        >>> #     'targetUv', 'nameLightCorrection', 'sensitivity1', 'sensitivity2',
        >>> #     'thresholdEnergy', 'slope', 'yslice', 'bg', 'file_name'])
        >>>
        >>> print('pys data (DataFrame format):')
        >>> print(ac_converter.df)
        >>> print(ac_converter.df.columns)
        >>> # Index([
        >>> #     'uvEnergy', 'countingCorrection', 'photonCorrection',
        >>> #     'pyield', 'npyield', 'nayield', 'guideline',
        >>> #     'countingRate', 'flGrandLevel', 'flRegLevel', 'uvIntensity'],
        >>> #     dtype='object')
        >>>
        >>> print("User analysis values:")
        >>> print(ac_converter.estimated_value)
        >>> print(ac_converter.estimated_value.keys())
        >>> # dict_keys(['thresholdEnergy', 'slope', 'yslice', 'bg'])
 
    """
    def __init__(self,file_name):
        """Constructor.

        Args:
            file_name (str): .dat filename
        """
        self.file_name = Path(file_name)
        
    def convert(self):
        """Converts data and generates metadata, DataFrame, JSON, etc.

        """
        self._read_para()
        self.countingCorrection = self._count_calibration()
        self.photonCorrection = self._photon_calibration()
        self.ydata, self.npyield = self._pyield_intensity()
        self.user_estimation()
        self._make_metadata()
         
    def _make_metadata(self):
        # Original metadata keys
        meta_keys = ["fileType","deadTime","countingTime","powerNumber",
                      "anodeVoltage","step","model","yAxisMaximum","startEnergy",
                      "finishEnergy","flagDifDataGroundLevel","bgCountingRate","measureDate","sampleName",
                      "uvIntensity59","targetUv","nameLightCorrection","sensitivity1","sensitivity2"]
        
        meta_values = [self.fileType, self.deadTime, self.countingTime, self.powerNumber,
                      self.anodeVoltage, self.step,self.model, self.yAxisMaximum, self.startEnergy,
                      self.finishEnergy,self.flagDifDataGroundLevel, self.bgCountingRate,self.measureDate,self.sampleName,
                      self.uvIntensity59,self.targetUv, self.nameLightCorrection, self.sensitivity1,self.sensitivity2]
        
        # Keys for calculated data (values are lists)
        calc_data_keys = ["uvEnergy", "countingCorrection", "photonCorrection", "pyield", "npyield","nayield", "guideline",
                         "countingRate","flGrandLevel","flRegLevel","uvIntensity"]
        
        calc_data_values = [self.uvEnergy, self.countingCorrection, self.photonCorrection, self.ydata, self.npyield, self.nayield, self.guideline,
                            self.countingRate,self.flGrandLevel,self.flRegLevel,self.uvIntensity]
        calc_data_values_list = [d.tolist() for d in calc_data_values]
        
        # Filename key
        file_meta ={'file_name':self.file_name.name}
        
        self.metadata = dict(zip(meta_keys + calc_data_keys, meta_values + calc_data_values_list)) 
        self.metadata.update(self.estimate_value)
        self.metadata.update(file_meta)
        
        self.calcdata = dict(zip(calc_data_keys,calc_data_values))
        
        # Metadata excluding calculated data
        # self.metadata_wo_calc: the calculated data, which is array data, is not included.
        self.metadata_wo_calc = dict(zip(meta_keys,meta_values)) 
        self.metadata_wo_calc.update(self.estimate_value)
        self.metadata_wo_calc.update(file_meta)
        
        self.json = json.dumps(self.metadata)   
        self.df = pd.DataFrame(self.calcdata)
        
    def _read_para(self):
        # read parameters up to the third line
        enc = getEncode(str(self.file_name))
        with open(str(self.file_name),encoding=enc) as f:
            reader = csv.reader(f)
            meta = [row for row in reader]

        # Match the parameter list to the full parameter list.
        # this case is AC-5 old dat type
        if len(meta[0]) == 10:    
            meta[0].extend(['0','0.0'])
            meta[2].extend(['1','1'])
            
        else:
            pass
        
        self.fileType = meta[0][0]
        self.deadTime = float(meta[0][1])
        self.countingTime = float(meta[0][2])
        self.powerNumber = float(meta[0][3])
        self.anodeVoltage = float(meta[0][4])
        self.step = float(meta[0][5])
        self.model = meta[0][6]
        self.yAxisMaximum = float(meta[0][7])
        self.startEnergy = float(meta[0][8])
        self.finishEnergy = float(meta[0][9])
        self.flagDifDataGroundLevel = int(meta[0][10])
        self.bgCountingRate = float(meta[0][11])
        self.measureDate = meta[1][0]
        self.sampleName = meta[1][1]
        self.uvIntensity59 = float(meta[2][0])
        self.targetUv = float(meta[2][1])
        self.nameLightCorrection = meta[2][2]
        self.sensitivity1 = float(meta[2][3])
        self.sensitivity2 = float(meta[2][4])
        
        # Transpose row to col
        raw_data = [list(x) for x in zip(*meta[3:])]
        self.uvEnergy = np.array([float(v) for v in raw_data[0]])
        self.countingRate = np.array([float(v) for v in raw_data[1]])
        self.flGrandLevel = np.array([int(v) for v in raw_data[2]])
        self.flRegLevel = np.array([int(v) for v in raw_data[3]])
        self.uvIntensity = np.array([float(v) for v in raw_data[4]])
       
       
    def _count_calibration(self):
        """Performs counting rate calibration.

        "AC-3" and "AC-2" data do not require counting rate calibration.

        Returns:
            numpy.ndarray: Calibrated counting rate
        """
        # The "AC-3" and "AC-2" counts do not need to be calibrated.
        if self.model == 'AC-3' or self.model == 'AC-2':
            # print('AC-2,AC-3')
            self.countingCorrection = self.countingRate
            
        else:
            part1 = (self.countingRate)/(1-self.deadTime*(self.countingRate))
            part2 = np.exp(0.13571/(1-0.0028*(self.countingRate)))*self.sensitivity1
            part3 = (self.bgCountingRate)/(1-self.deadTime*(self.bgCountingRate))
            part4 = np.exp(0.13571/(1-0.0028*(self.bgCountingRate)))*self.sensitivity1
            self.countingCorrection = part1*part2-part3*part4
          
        return self.countingCorrection

    def _photon_calibration(self):
        """Performs photon number calibration.

        Returns:
            numpy.ndarray: Calibrated photon number
        """
        # number of Photons
        self.nPhoton = 0.625*(self.uvIntensity/self.uvEnergy)
        # number of photons per unit 
        self.unitPhoton = (self.uvIntensity59*0.625)/5.9
        self.photonCorrection = self.nPhoton/self.unitPhoton

        return self.photonCorrection

    def _pyield_intensity(self):
        """Calculates PYS intensity.

        Returns:
            tuple (numpy.ndarray, numpy.ndarray): ydata, npyield
        """
        self.ydata = self.countingCorrection/self.photonCorrection
        # Replace negative values with 0
        self.ydata = np.where(self.ydata < 0, 0, self.ydata)
        self.npyield = np.power(self.ydata, self.powerNumber)
        
        # replace Nan with 0
        self.npyield[np.isnan(self.npyield)] = 0

        return self.ydata, self.npyield
    
    
    @staticmethod
    def relu(xdata, a, b, bg):
        """ReLU function.

        Args:
            xdata (numpy.ndarray): Input data
            a (float): Slope
            b (float): y-intercept
            bg (float): Background value

        Returns:
            numpy.ndarray: Result of the ReLU function
        """
        ip = (bg - b)/a
        u = (xdata - ip)
        return a * u * (u > 0.0) + bg

    
    @staticmethod
    def user_fit(bg_ydata, reg_xdata, reg_ydata, printf=False):
        """liner fit function.

        Args:
            bg_ydata (numpy.ndarray): Background data
            reg_xdata (numpy.ndarray): Regression data (x)
            reg_ydata (numpy.ndarray): Regression data (y)
            printf (bool, optional): Whether to print the results. Defaults to False.

        Returns:
            dict: Fit parameters ('thresholdEnergy', 'slope', 'yslice', 'bg')
        """
        
        bg = np.nanmean(bg_ydata)
        popt = np.polyfit(reg_xdata, reg_ydata, 1)

        a = popt[0]
        b = popt[1]
        
        cross_point =  (bg - b) / a

        if printf:
            print(f"bg:{bg}, a(slope):{a}, b(yslice):{b}")
            print(f"thresholdEnergy -> {cross_point}" )
            
        return {'thresholdEnergy': cross_point, 'slope': a,'yslice': b, 'bg':bg}
        
    def user_estimation(self):
        """ Estimates the threshold value analyzed by the user.
        
        Returns:
            dict[float]: 'thresholdEnergy', 'slope','yslice','bg'
        """
    
        self.bg_flag_ind = np.where(self.flGrandLevel == -1)[0].tolist()
        self.reg_flag_ind = np.where(self.flRegLevel == -1)[0].tolist()
        # print( bg_flag_ind, reg_flag_ind)
        
        if  self.bg_flag_ind != [] and self.reg_flag_ind != []:
            # Calibration of background difference
            if self.flagDifDataGroundLevel == -1:
                # average
                bg_ave = np.nanmean(self.ydata[self.bg_flag_ind])
                c_pys = self.ydata - bg_ave
                self.cc_pys = np.where(c_pys < 0, 0, c_pys)
                self.cc_npys = np.power(self.cc_pys, self.powerNumber)
                
                # bg_xdata = self.uvEnergy[bg_flag_ind]
                bg_ydata = np.array([0.0]*len(self.bg_flag_ind))
                reg_xdata = self.uvEnergy[self.reg_flag_ind]
                reg_ydata = self.cc_npys[self.reg_flag_ind]
               
            #  elif self.flagDifDataGroundLevel == 0:  
            else:   
                # bg_xdata = self.uvEnergy[bg_flag_ind]
                
                self.cc_pys = self.ydata
                self.cc_npys = np.power(self.cc_pys, self.powerNumber)
                reg_xdata = self.uvEnergy[self.reg_flag_ind]
                reg_ydata = self.cc_npys[self.reg_flag_ind]
                bg_ydata = self.cc_npys[self.bg_flag_ind]
                

            self.estimate_value = AcConv.user_fit(bg_ydata, reg_xdata, reg_ydata, printf=False) 
            self.nayield = self.cc_npys
            self.guideline = AcConv.relu(xdata=self.uvEnergy, a=self.estimate_value['slope'], 
                                        b=self.estimate_value['yslice'], 
                                        bg=self.estimate_value['bg'])
        
        else:
            self.estimate_value =  {'thresholdEnergy': np.nan, 'slope': np.nan,'yslice':np. nan,'bg':np.nan}
            self.nayield = self.npyield
            self.guideline = np.array([np.nan]*len(self.uvEnergy.tolist()))

      
    def export_df2csv(self,df_out_file_name=None):
        """Exports the DataFrame data to a CSV file.

        Args:
            df_out_file_name (str, optional): Output filename. Defaults to None.
        """
        if df_out_file_name is None:
            df_out_file_name =self.file_name.with_suffix('.csv')
        
        self.df.to_csv(df_out_file_name, index=False)
    
    def export_json(self,json_out_file_name=None):
        """Exports the metadata to a JSON file.

        Args:
            json_out_file_name (str, optional): Output filename. Defaults to None.
        """
        if json_out_file_name is None:
            json_out_file_name =self.file_name.with_suffix('.json')

        dict_json = json.loads(self.json)
        with open(json_out_file_name, 'w') as f:
            json.dump(dict_json, f, indent=4)
    
    def plot_ax(self, axi=None):
        """Plots the data.

        Args:
            axis (matplotlib.axes._subplots.AxesSubplot, optional): 
            The axis to plot on. Defaults to None.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The plotted axis.
        """

        if axi is None:
            fig_ = plt.figure()
            ax_ = fig_.add_subplot(111)
        else:
            ax_ = axi
  
        ax_.set_title(f'{self.metadata["sampleName"]}')   
        ax_.plot(self.uvEnergy, self.npyield,'ro',label='Data')

        if  ~np.isnan(self.estimate_value['thresholdEnergy']):
            ax_.plot(self.uvEnergy,self.guideline, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyle = '-', 
                    label=f'User\nThreshold: {self.estimate_value["thresholdEnergy"]:.2f}eV\nSlope:{self.estimate_value["slope"]:.2f}')
            ax_.axvline(self.estimate_value["thresholdEnergy"], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2] )
            ax_.text(self.estimate_value["thresholdEnergy"], np.max(self.npyield)*0.3, f'{self.estimate_value["thresholdEnergy"]:.2f}')

            ax_.set_xlabel('Energy [eV]')
            ax_.legend(title=f"Power {self.metadata['uvIntensity59']:.2f}nW")
            ax_.grid()
            
            if 0.49 < self.metadata["powerNumber"] < 0.51:
                ax_.set_ylabel('Photoelectron yield$^{1/2}$ [arb.unit]')
        
            elif 0.3 < self.metadata["powerNumber"] < 0.35 :
                ax_.set_ylabel('Photoelectron yield$^{1/3}$ [arb.unit]')
            else:
                pass
        
        if axi is None:
            plt.show()
            return 
        
        return ax_   

class AdvAcConv(AcConv):
    """Advanced AC data converter.

    Removes count overflowed data and over 6.8eV data.
    
    Note:
    Light intensity correction above 6.8 eV may not be accurate
    Measures to prevent detector overflow
    Raw data: countingRate
    Calculated value after counting error correction: countingCorrection
    AC-2, AC-3 Max :2000cps
    AC-5, AC-2S Max :4000cps
    
    """
    def __init__(self, file_name):
        super().__init__(file_name)
    
    def convert(self):
        self._read_para()
        self.countingCorrection = self._count_calibration()
        self.photonCorrection = self._photon_calibration()
        self.ydata, self.npyield = self._pyield_intensity()
        self.user_estimation()
        self._trim_array_energy()
        self._trim_array_maxcount()
        self._make_metadata()
    
    def _trim_array_energy(self):
        """trim array data by max energy at 6.8.
        """
        limit_x_value = 6.81
        trim_max_index = np.argmax(self.uvEnergy >= limit_x_value)
        # If there are no values greater than limit_x_value,
        # trim_max_index will be 0. To avoid this,
        # if no values satisfy the condition, use the maximum index of the data.
        if np.all(self.uvEnergy < limit_x_value):
            trim_max_index = len(self.uvEnergy) + 1      
        self.uvEnergy = self.uvEnergy[:trim_max_index]
        self.countingCorrection = self.countingCorrection[:trim_max_index]
        self.countingRate = self.countingRate[:trim_max_index]
        self.photonCorrection = self.photonCorrection[:trim_max_index] 
        self.flGrandLevel = self.flGrandLevel[:trim_max_index]
        self.flRegLevel = self.flRegLevel[:trim_max_index]
        self.uvIntensity = self.uvIntensity[:trim_max_index] 
        self.ydata = self.ydata[:trim_max_index]
        self.npyield = self.npyield[:trim_max_index]
        self.nayield = self.nayield[:trim_max_index]
        self.guideline = self.guideline[:trim_max_index]
    
    def _trim_array_maxcount(self):
        """trim array data by max countingCorrection
        """
        if self.model == 'AC-3' or self.model == 'AC-2':
            limit_countingCorrection = 2000 
        else:
            limit_countingCorrection = 4000
            
        trim_first_index = np.argmax(self.countingCorrection >= limit_countingCorrection)

        if np.all(self.countingCorrection < limit_countingCorrection):
            trim_first_index = len(self.countingCorrection) + 1
        
        self.uvEnergy = self.uvEnergy[:trim_first_index]
        self.countingCorrection = self.countingCorrection[:trim_first_index]
        self.countingRate = self.countingRate[:trim_first_index]
        self.photonCorrection = self.photonCorrection[:trim_first_index]   
        self.flGrandLevel = self.flGrandLevel[:trim_first_index]
        self.flRegLevel = self.flRegLevel[:trim_first_index]
        self.uvIntensity = self.uvIntensity[:trim_first_index]
        self.ydata = self.ydata[:trim_first_index] 
        self.npyield = self.npyield[:trim_first_index]
        self.nayield = self.nayield[:trim_first_index]
        self.guideline = self.guideline[:trim_first_index]
            
     
if __name__ =='__main__':
    pass
    # fp = r'////.dat'
    # fpdata = AcConv(fp) 
    # fpdata.convert()
    # print(fpdata.df)
    # print(fpdata.json)
    # print(fpdata.metadata)

