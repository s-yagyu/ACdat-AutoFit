from pathlib import Path

import csv
import numpy as np

from acdatconv import datconv as dv
from acdatconv import datlib as dlib


def flag_overwrite(ref_file: str, target_file: str) -> None:
    # """
    # 概要:
    #     ref_fileに含まれるAcConvクラスのflGrandLevelとflRegLevelを取得し、
    #     target_fileのuvEnergy, countingRate, uvIntensityと結合して、新しいCSVファイルを作成する。

    # 引数:
    #     ref_file (str): 参照用のCSVファイルのパス。
    #     target_file (str): 新しいCSVファイルのパス。

    # 戻り値:
    #     None

    # 例外:
    #     ValueError: ファイルの読み込みに失敗した場合。

    # 注意:
    #     CSVファイルはUTF-8で保存されている必要がある。
    # """
    
    # 参照用CSVファイルの読み込み
    
    try:
        with open(ref_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            first_line = next(reader)
            second_line = next(reader)
            third_line = next(reader)
            ac_temp_data = dv.AcConv(ref_file)
            ac_temp_data.convert()
            au_temp_flG = ac_temp_data.flGrandLevel
            au_temp_flR = ac_temp_data.flRegLevel
    except Exception:
        raise ValueError("CSVファイルの読み込みに失敗しました。")

    # 新しいファイルの作成
    acdata = dv.AcConv(target_file)
    acdata.convert()
    a1 = acdata.uvEnergy.tolist()
    a2 = acdata.countingRate.tolist()
    a3 = au_temp_flG.astype(np.int8).tolist()
    a4 = au_temp_flR.astype(np.int8).tolist()
    a5 = acdata.uvIntensity.tolist()
    matrix = [[a1[i], a2[i], a3[i], a4[i], a5[i]] for i in range(len(a1))]
    file_path = Path(target_file)
    new_file_name = file_path.stem + "_new" + file_path.suffix
    with open(new_file_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
        writer.writerow(third_line)
        writer.writerows(matrix)
        
if __name__ == "__main__":
    # flag_overwrite(ref_file=r'Au_230403094512_1.dat',
    #            target_file=r'Au_230403094512_2.dat')
    pass