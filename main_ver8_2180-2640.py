#最小２乗関数を搭載
#a,b両方でループ
#偏極度を53%に合わせるために時定数を1.5kHz用に仮で書き直してみる
#偏極度の初期値は固定

import matplotlib
import numpy as np  #NumPyライブラリ
import matplotlib.pyplot as plt  #データ可視化ライブラリ
import euler4
import least_squares

#################
Pe = 0.9
TB = 4.07317
TL = 9.27057
TD = 1/(1/TB-1/TL)

#偏極度を合わせるために適当に置く
TD = 6.37

#################

#beam情報
#------------------------------------------
class beam:
    def __init__(self):
        #日程名称
        self.name = ""
        #平均ビーム強度
        self.I = 0
        #ビーム照射開始時間(そこまででいくらビームが照射されているか。積分効果計算のため。)
        self.t0 = 0
        #ビーム照射日時
        self.day = 0
        #初期偏極度
        self.P0 = 0
        #最小2乗法のχ
        self.chi = 0

beam_1 = beam()
beam_1.name = "beam_1"
beam_1.I = 22.32417886

beam_1.day = '2180-2640'
beam_1.P0 = 0.539676
beam_1.chi = 10000


#------------------------------------------
#積算効果計算用
#2180-2640
I_1 = 22.32417886
#3480-3620
I_2 = 62.21711034
#3640-3660
I_3 = 204.59
#3680-4020
I_4 = 528.959258
#4040-4080
I_5 = 671.25
#4920-4960
I_6 = 942.3
#4980-5160
I_7 = 713.5861279

t_1 = 16102/3600
t_2 = 6942/3600
t_3 = 1250/3600
t_4 = 17088/3600
t_5 = 2772/3600
t_6 = 2448/3600
#t_7はラストなので定義はしているが使用しない
t_7 = 23969/3600

S = 0
print("S={:.7f}".format(S))
#--------------------------------------------------------------------


P0 = beam_1.P0
#beta_s = [0.0,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.00010,0.00011,0.00012,0.00013,0.00014,0.00015,0.00016,0.00017,0.00018,0.00019,0.00020]
#alpha_s = [0.0]
#beta_s  = [0.0,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.00010,0.00011,0.00012,0.00013,0.00014,0.00015,0.00016,0.00017,0.00018,0.00019,0.00020]

#alpha_s = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.]
#beta_s  = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.]
alpha_s = []
beta_s = []

for i in range(100):
    alpha_s.append(float(i))
    beta_s.append(float(i))

#alpha_s = [0.00010]
#beta_s  = [0.00001]

#alpha_s = [0.0,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.00010,0.00011,0.00012,0.00013,0.00014,0.00014,0.00015]
#beta_s  = [0.0,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.00010,0.00011,0.00012,0.00013,0.00014,0.00014,0.00015]

#alpha_s = [0.0,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]
#beta_s  = [0.0,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]
#alpha_s = [-0.0005,-0.0004,-0.0003,-0.0002,-0.0001,-0.0000,0.0001,0.0002,0.0003,0.0004,0.0005]
#beta_s  = [-0.0005,-0.0004,-0.0003,-0.0002,-0.0001,-0.0000,0.0001,0.0002,0.0003,0.0004,0.0005]

#alpha_s = [0.000]
#beta_s = [0.]

alpha_opt = alpha_s[0]
beta_opt = beta_s[0]
h = open("chi_1227.txt", 'w')

for alpha in alpha_s:
    for beta in beta_s:
        print("alpha = {:.7f},  beta = {:.7f}".format(alpha, beta))
        euler4.main1(TD,TL,Pe,alpha,beta,beam_1.I,beam_1.t0,beam_1.day,P0,S)

        with open("2180-2640.txt") as f:
            lines = f.readlines()
        with open('/Users/matsuitakaya/Desktop/研究室/偏極陽子/作業/標的系/グラフ/pol_HIMAC/本測定/int_100/radiationloss/files/file_{0}_P0={1}_alpha={2}_beta={3}.txt'.format(beam_1.day,P0,alpha,beta)) as g:
            lines2 = g.readlines()

        #print(len(lines2))

        t_MAX = min(len(lines),len(lines2))
        print(len(lines))
        chi_1227 = least_squares.calc(lines,lines2,0,int(t_MAX))
        print(chi_1227)

        with open("chi_1227.txt", 'a') as h:
            h.write("{:.7f} {:.7f} {:.7f}\n".format(alpha,beta,chi_1227))
        if chi_1227 < beam_1.chi:
            beam_1.chi = chi_1227
            alpha_opt = alpha
            beta_opt = beta
    with open("chi_1227.txt", 'a') as h:
            h.write("\n")

    
print("alpha_last = {:.7f},  beta_last = {:.7f}".format(alpha_opt, beta_opt))
#a,bの最適化が終わったらその値を使ってプロット
euler4.main1(TD,TL,Pe,alpha_opt,beta_opt,beam_1.I,beam_1.t0,beam_1.day,P0,S)
print("chi_min={:.7f}".format(beam_1.chi))
#euler2.main1(TD,TL,Pe,alpha,beta,beam_2.I,beam_2.t0,beam_2.day)
    #for alpha in range(100):
    #   for beta in range(100):
    #      euler2.calc(TD.TL,Pe,alpha,beta,beam_1.I,beam_1.t0,beam_1.day)