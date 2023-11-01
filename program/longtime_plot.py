#最適パラメータを用いてさらに長時間ビーム照射を行った時の偏極度推移をプロットする
import matplotlib.pyplot as plt
import numpy as np


def long_plot_calc(TD,TL,alpha,beta,gamma,t_end,I_cps):

    
    Pe = 0.9
    #十分偏極発展している状態からの減偏極をプロットする
    Pcal_2 = Pe/(1.+(TD/TL))
    I = I_cps * 60.
    #計算数はlen(t)=116 * n
    #iからi+1の間をn分割
    n = 100
#最適値でプロット用に計算する場合のみtxtファイルに書き込む
#1時間毎にプロット
    t = [0]*(int(t_end/20)+1)
    Pcal=[0]*(int(t_end/20)+1)
    test=[0]*(int(t_end/20)+1)
    #print(Pcal)
    for i in range(int(len(Pcal))):
        #print(i)
    #出力時間単位はday
        t[i] = i*20/60./24.
        #計算はminで行う
        t_cal = (i*20)
                #20 minごとにPcal_2の値をPcalに配列として入力し、データ形式をtxtファイルと合わせる
        Pcal[i] = Pcal_2
        
        for j in range(n):

            t_cal_2 = (t_cal+j*20/n)
            #使用するのは1540-4080###################################################
            #BIはcpm=cps*60で与えている
            
                
            #I = (3.41E6)*60.
            
            
            #########################################################################

            
            #時間の単位はminで計算している(出力する時はh)
            #dif = (Pe-Pcal_2)/(TD*60.)-(1/(TL*60.)+alpha*I*t_s+beta*gamma_2)*Pcal_2
            dif = (Pe-Pcal_2)/(TD*60.)-(1/(TL*60.) + alpha/60.*I*t_cal_2 + beta/60.*(1-np.exp(-(t_cal_2)/(gamma*60.))))*Pcal_2
            #20/n min間の変化分を一つ前の偏極度に足す
            #20min * j/n としていたが、これだとjが大きくなるにしたがって足し上げるdifの時間幅も大きくなっていってしまう
            Pcal_2 = Pcal_2 + dif*20/float(n)
            #print(alpha/60.*I*t_cal_2)
            
    
    return t,Pcal    
        #if i == 0:
            #Pcal[i] = P[i]
            #continue

            #データはminだが、時定数は1/hを使っているのでここでhに揃える

def long_plot(TD,TL,alpha,beta,gamma,t_end):
    plt.figure()
    plt.xlim([0,t_end/(60.*24.)])
    plt.ylim([0.,0.6])
    I_list = []
    for i in range(0,10,1):
        I_list.append(i)
    
    legend = []
    for i in range(len(I_list)):
        t,Pcal = long_plot_calc(TD,TL,alpha,beta,gamma,t_end,10**i)
        legend.append('10^{0} cps'.format(i))
        plt.plot(t, Pcal,'-')



    plt.legend(legend)
    
    
    #plt.plot(t, theoreticalValue(betaG),'-')

    plt.xlabel('t [day]')
    plt.ylabel('P')
    
    plt.show()

    return Pcal