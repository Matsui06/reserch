#ビーム強度をBCではなくcpsの形で3日目を与え、BC比を用いてbeam1.beam2のcpsを計算して用いる
#delta=1で固定

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import time




class memo():
    r_sum_memo = 100

f = np.loadtxt(R"C:\Users\mtsit\reserch\data\1540-4080.txt")
t = f[:,0]
P = f[:,1]

def theoreticalValue(TD,TL,P0,alpha_index,beta_index,gamma_index,delta=1,epsilon=1):
    Pe = 0.9
    #alpha,betaはオーダーから振る必要があるので、均等に振るにはこの形の方がいい
    alpha = 10**alpha_index
    #alpha = alpha_index
    beta = 10**beta_index

    #beam切り替えによる温度変化の時定数~60min
    gamma = gamma_index

    #TD = 10**TD_index
    #TL = 10**TL_index
    #時定数[min]

    #TD = param[0]
    #TL = param[1]
    """""
    TD = 10.5*60
    TL = 15*60
    TD = param[0]
    TL = param[1]
    alpha = param[2]
    beta = param[3]
    gamma = param[4]
    #gamma = 0.5
    delta = param[5]
    epsilon = param[6]
    #Pcal_2 = P[0] *gamma
    """
    Pcal_2 = P0
    """
    alpha = param[2]
    beta = param[3]
    gamma = param[4]
    delta = param[5]

    epsilon = param[6]
    """
    #epsilon = 1.
    

#固定したいパラメータはここで上書きする(LMとしてやっていい処理科は不明だが、現時点では大丈夫そう)
    #TD = 6.5*60.
    #TL = 10.0*60.
    #alpha = 9e-8
    #beta = 1e-6


    #計算数はlen(t)=116 * n
    #iからi+1の間をn分割
    n = 10


    Pcal=[0]*int(len(t))
    #print(Pcal)
    for i in range(int(len(t))):
        #print(i)
        t_cal = (1540+i*20)
        
        Pcal[i] = Pcal_2
        #print(t[i])
        for j in range(n):

            #使用するのは1540-4080###################################################
            #BIはcpm=cps*60で与えている
            #gamma_2はビームon,offの状態を表す指標

            #12/26
            t_cal_2 = t_cal+j*20/n
            if t_cal_2<2189:
                I = 0.
                S = 0.
                t_s = t_cal_2
                gamma_2 = -1.

            elif t_cal_2<2649:
                I = (3.41E6)*60./24.4
                S = 0.
                t_s = t_cal_2-2194
                gamma_2 = 1.
                t_start = 2189.

            elif t_cal_2<3478:
                I = 0.
                S = alpha * (3.41E6)*60./24.4 * (2649-2189)
                t_s = t_cal_2-2649
                gamma_2 = 0.
                t_start = 2649.

            #12/27
            
            elif t_cal_2<3620:
                I = (3.41E6)*60./24.4*4.46
                S = alpha * (3.41E6)*60./24.4 * (2649-2189)
                t_s = t_cal_2-3478
                gamma_2 = 1.
                t_start = 3478.
            
            elif t_cal_2<4083:
                I = (3.41E6)*60.
                S = alpha * ((3.41E6)*60./24.4 * (2649-2189) + (3.41E6)*60./24.4*4.46 * (3620-3478))
                t_s = t_cal_2-3620
                gamma_2 = 1.
                #強度を増やした場合は温度は変わっていない想定なのでベータの効果は変わらない→expは<3620の条件をそのまま用いる
                t_start = 3478
            #########################################################################
            #12/28
            #この先はBIなど何も修正していないためこのまま使用不可
            elif t_cal_2<4930:
                I = 0.
                S = alpha * (22.3 * (2649-2194) + 62.2 * (3620-3478) + 528 * (4083-3620))
                t_s = t_cal_2-4083
            elif t_cal_2<4971:
                I = 942.3
                S = alpha * (22.3 * (2649-2194) + 62.2 * (3620-3478) + 528 * (4083-3620))
                t_s = t_cal_2-4930
            elif t_cal_2<5401:
                I = 713.6
                S = alpha * (22.3 * (2649-2194) + 62.2 * (3620-3478) + 528 * (4083-3620) +  942.3 * (4971-4930))
                t_s = t_cal_2-4971
            

            
            
            
            #時間の単位はminで計算している(出力する時はh)
            #beam照射前
            if gamma_2 == -1:
                dif = (Pe-Pcal_2)/(TD*60.)-(1/(TL*60.))*Pcal_2

            #beam off
            if gamma_2 == 0:
                dif = (Pe-Pcal_2)/(TD*60.)-(1/(TL*60.)+S+alpha*I*t_s+beta*(np.exp(-(t_cal_2-t_start)/gamma)))*Pcal_2

            #beam on
            if gamma_2 == 1:
                dif = (Pe-Pcal_2)/(TD*60.)-(1/(TL*60.)+S+alpha*I*t_s+beta*(1-np.exp(-(t_cal_2-t_start)/gamma)))*Pcal_2
            #20/n min間の変化分を一つ前の偏極度に足す
            #20min * j/n としていたが、これだとjが大きくなるにしたがって足し上げるdifの時間幅も大きくなっていってしまう
            Pcal_2 = Pcal_2 + dif*20/float(n) 
            
            
        #if i == 0:
         #   Pcal[i] = P[i]
          #  continue
        #20 minごとにPcal_2の値をPcalに配列として入力し、データ形式をtxtファイルと合わせる
        #Pcal[i] = Pcal_2
    #パラメータが負になって欲しくないので、負の時を最適としないように全然違う値を入れておく
    if TD < 0 or TL < 0 or alpha < 0 or beta <0 or gamma < 0 or delta < 0 or epsilon <0:
        Pcal = [0]*int(len(t))
    #print(Pcal)
    return Pcal
#"""""
#残差
def objectiveFunction(TD,TL,P0,alpha,beta,gamma,delta=1,epsilon=1):
    #データ-計算値
    #最終的な残差を知りたいのでグローバル変数として定義
    global r
    r=(P-theoreticalValue(TD,TL,P0,alpha,beta,gamma,delta,epsilon))**2
    #データ点がない時刻のデータは0としてあるので、計算値もデータに合わせる
    r[9] = 0.
    r[10] =0.
    r[81] = 0.
    r[82] = 0.
    

    r_sum = sum(r)
    #print(r_sum)
    #print("r[9]={:},r[10]={:},r[10]={:}".format(r[9],r[10],r[11]))
    
    param_list = [TD,TL,alpha,beta,gamma]
    param_list[0] = param_list[0]/60.
    param_list[1] = param_list[1]/60.
    
    return r_sum
"""""
def objectiveFunction(beta):
    #データ-計算値
    r=(P-theoreticalValue(beta))
    r_sum = 0.
    for i in range(len(P)):
        r_sum = r_sum + (P[i]-theoreticalValue(beta)[i])
    beta_list = list(beta)
    beta_list[0] = beta_list[0]/60.
    beta_list[1] = beta_list[1]/60.
    #print('TD={:.3f}, TL={:.3f}, alpha={:e}, beta={:e}, r_sum={:.3f}'.format(beta_list[0],beta_list[1],beta_list[2],beta_list[3],r_sum))
    return r
"""""


