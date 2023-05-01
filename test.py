import matplotlib
import numpy as np  #NumPyライブラリ
import matplotlib.pyplot as plt  #データ可視化ライブラリ
import euler4
import least_squares

with open('/Users/matsuitakaya/Desktop/研究室/偏極陽子/作業/標的系/グラフ/pol_HIMAC/本測定/int_100/radiationloss/files/test!.txt', mode='a') as f:
            

            #実験データと合わせるために20分ごとにデータをファイルに書き込む
            #ここで生成するファイルは単位が[h]なので20min=0.3333...h→33行目
            
    f.write("aaa")