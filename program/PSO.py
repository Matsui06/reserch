#参考サイト
#https://aimedchem.hatenablog.com/entry/2021/07/31/155602

import numpy as np
import pandas as pd
from IPython.display import Image
import array
import sys

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters import (formatters)
#Animating swarms
from pyswarms.utils.plotters.formatters import Mesher


import difcalc
import matplotlib.pyplot as plt
import longtime_plot
import time

start = time.time()


r_sum_memo = 100

#最初にログデータを初期化
#ファイル名などを変える場合はoptimizeの定義を開き、gbestと検索してcompute_gbestの定義を開き、65行目あたりのファイル名を書き換える
with open(R"C:\Users\mtsit\reserch\data\best_param_log.txt", 'r+') as f:
    f.truncate(0)
# 最適化したい関数を定義（10, -3, 3 で最小値 0 になる）
def object_function(x):
    #y = (x[:, 0] - 10) ** 2 + (x[:, 1] + 3) ** 2 + (x[:, 1] + x[:, 2]) ** 2
    y = np.empty(int(len(x[:,0])))
    """
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #print(x[:,0])
    #y = [0] * (int(len(x[:,0])))
    
    print('x type is {},length is {},dim is {}'.format((type(x)),len(x),x.ndim))
    print('y type is {},length is {},dim is {}'.format((type(y)),len(y),y.ndim))
    print('x  is {}'.format(x))
    print('y  is {}'.format(y))
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    """
    
    for i in range(len(x[:,0])):
        TD = float(x[i, 0])
        TL = float(x[i, 1])
        P0 = float(x[i, 2])
        alpha = float(x[i, 3])
        beta = float(x[i, 4])
        gamma = float(x[i, 5])
        #delta = float(x[i, 6])
        #epsilon = float(x[i, 7])
        #print('Alice is {} years old'.format(TD))
        #a = difcalc_2.objectiveFunction(TD,TL,alpha,beta,gamma,delta,epsilon)
        a = difcalc.objectiveFunction(TD,TL,P0,alpha,beta,gamma)
        #print('a type  is {}'.format(type(a)))
        #print(a)
        y[i] = a
        #if y[i] < y[i-1] and i > 0:
            #print('TD={:.3f}, TL={:.3f}, alpha={:e}, beta={:e}, r_sum={:.3f}'.format(TD,TL,alpha,beta,gamma,delta,epsilon,y[i]))
    #print(type(y))
    return y




# 乱数固定
#np.random.seed(0)

# 探索範囲を指定。一つ目のタプルは最小値を、二つ目のタプルは最大値を示す。
#bounds = ((5.*60,10.*60,1e-6,1e-6), (12.*60,20.*60,1e-4,1e-4))
#ビルドアップデータ用

#bounds = ((4.*60.,8.*60.,0.00000001,0.00001), (8.*60.,15.01*60., 0.000000301,0.000501))
#bounds = ((4.*60.,8.*60.,0.00000001,0.00001,0.3,1,1), (8.*60.,15.01*60., 0.000000301,0.000501,1.5,1.0001,1.0001))

#######################################パラメータ範囲指定############################################################
#偏極生成時定数
TD_min = 12.0
TD_max = 12.5
#緩和時定数
TL_min = 18.
TL_max = 20.5
#初期偏極度
P0_min = 0.49
P0_max = 0.505
#alpha,beta,gammaはオーダーをパラメータとしている
alpha_min = -16.
alpha_max = -14.
#強度効果をbeta(1-exp(-I/gamma))の形にしている
beta_min = - 4.
beta_max = -3.
gamma_min = 70.
gamma_max = 80.
#ビーム1の強度は22としている(by BC)が、この値に合わせてほかのパラメータが決まるのでこの数値に意味はない
#おそらく最終的にはここにどこかわかる日時のcpsが入る
#ビーム1に対するビーム2の倍数(BC say "4.457794134")
#delta_min = 1.0
#delta_max = 1.00001
#ビーム1に対するビーム3の倍数(BC say "24.4078291")
#epsilon_min = 24.
#epsilon_max = 24.5
#PSOパラメータ
n_particles = 100
iters = 30
# 最適化関数の重み(hyperparameter)を決める
#c1:ローカル項　c2:グローバル項　w:慣性項(0~1)
options = {"c1": 0.5, "c2": 0.9, "w":0.1}

##############################################################################################################



#bounds = ((TD_min,TL_min,P0_min,alpha_min,beta_min,gamma_min,delta_min,epsilon_min), (TD_max,TL_max,P0_max,alpha_max,beta_max,gamma_max,delta_max,epsilon_max))
bounds = ((TD_min,TL_min,P0_min,alpha_min,beta_min,gamma_min), (TD_max,TL_max,P0_max,alpha_max,beta_max,gamma_max))
#bounds = ((7.000000*60.,10.0000000*60.,float(1.4e-8), 0.9,float(9.1e-6),float(8e-8),float(5e-2)), (7000001*60.,10.0000001*60.,float(1.5e-8),1, float(9.2e-6),float(8.1e-8),float(6e-2)))
# init_pos で初期座標を設定
#print(type(bounds))
optimizer = ps.single.GlobalBestPSO(n_particles, dimensions = 6, options = options, bounds = bounds)
#print("opt!")
# Perform optimization
cost, pos = optimizer.optimize(object_function, iters)
#print("cast!")
# 最適化された値を表示
#print("TD_best={:.3f} h, TL_best={:.3f} h,P0={:.3f}, alpha_best={}, beta_best={}, gamma_best={}, delta_best={:.3f},epsilon_best={:.3f}".format(pos[0],pos[1],pos[2],'{:.3e}'.format(10**pos[3]),'{:.3e}'.format(10**pos[4]),pos[5],pos[6],pos[7]))
print("TD_best={:.3f} h, TL_best={:.3f} h,P0={:.3f}, alpha_best={}, beta_best={}, gamma_best={}".format(pos[0],pos[1],pos[2],'{:.3e}'.format(10**pos[3]),'{:.3e}'.format(10**pos[4]),pos[5]))
print(pos[0],pos[1],pos[2],'{:.3e}'.format(10**pos[3]),'{:.3e}'.format(10**pos[4]),pos[5])
# 最適化された座標を表示
print("cost_min={:.5f}".format(cost))

print('所要時間は{:.7f}です。'.format(time.time()-start))
print('TD_bestとTL_bestによる偏極度は{:.3f}です。'.format(0.9/(1.+(pos[0]/pos[1]))))
#sys.exit()
TD = pos[0]
TL = pos[1]
P0 = pos[2]
alpha = pos[3]
beta = pos[4]
gamma = pos[5]
#delta = pos[6]
#epsilon = pos[7]

with open(R'C:\Users\mtsit\reserch\data\calc_PSO.txt',mode = 'w') as f:
    for i in range(len(difcalc.theoreticalValue(TD,TL,P0,alpha,beta,gamma))):
        f.write('{:.7f} {:.7f}\n'.format(1540+i*20,difcalc.theoreticalValue(TD,TL,P0,alpha,beta,gamma)[i]))

##################################################


# FigureとAxesの設定
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

ax.set_xlabel("time [h]", fontsize=14)
ax.set_ylabel("Signal Intensity [a.u]", fontsize=14)
ax.set_xlim(1540/60,4080/60)
ax.set_ylim(-1.5, 1.5)

f = np.loadtxt(R'C:\Users\mtsit\reserch\data\1540-4080_to_plot.txt')



t = f[:,0]/60.
P = f[:,1]/0.533
cal = np.genfromtxt(R"C:\Users\mtsit\reserch\data\calc_PSO.txt")
plt.xlim([1540/60,4080/60])
plt.ylim([0.45/0.533,0.60/0.533])
# Axesにグラフをプロット
ax.plot(t, P, 'o')
ax.plot(cal[:,0]/60.,cal[:,1],'-')
#ax.plot(cal[:,0]/60.,cal[:,1],'-')

# y1とy1の間をライム色で塗り潰す
ax.axhspan(ymin=0, ymax=1.2, xmin=(2189-1540)/60/(4080/60-1540/60), xmax=(2649-1540)/60/(4080/60-1540/60), color="red", alpha=0.3)
ax.axhspan(ymin=0, ymax=1.2, xmin=(3478-1540)/60/(4080/60-1540/60), xmax=(4080-1540)/60/(4080/60-1540/60), color="red", alpha=0.3)

plt.show()
#######################################################################

##################################################


# FigureとAxesの設定
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

ax.set_xlabel("time [h]", fontsize=14)
ax.set_ylabel("Polarization", fontsize=14)
ax.set_xlim(1540/60,4080/60)
ax.set_ylim(-1.5, 1.5)

f = np.loadtxt(R'C:\Users\mtsit\reserch\data\1540-4080_to_plot.txt')



t = f[:,0]/60.
P = f[:,1]
cal = np.genfromtxt(R"C:\Users\mtsit\reserch\data\calc_PSO.txt")
plt.xlim([1540/60,4080/60])
plt.ylim([0.45,0.60])
# Axesにグラフをプロット
ax.plot(t, P, 'o')
ax.plot(cal[:,0]/60.,cal[:,1],'-')
#ax.plot(cal[:,0]/60.,cal[:,1],'-')

# y1とy1の間をライム色で塗り潰す
ax.axhspan(ymin=0, ymax=1.2, xmin=(2189-1540)/60/(4080/60-1540/60), xmax=(2649-1540)/60/(4080/60-1540/60), color="red", alpha=0.3)
ax.axhspan(ymin=0, ymax=1.2, xmin=(3478-1540)/60/(4080/60-1540/60), xmax=(4080-1540)/60/(4080/60-1540/60), color="red", alpha=0.3)

plt.show()
#######################################################################

#プロット用にNMR測定をしていない0を入れてある点を抜いたデータを使用
f = np.loadtxt(R'C:\Users\mtsit\reserch\data\1540-4080_to_plot.txt')

t = f[:,0]/60.
P = f[:,1]

g = np.loadtxt(R'C:\Users\mtsit\reserch\data\beamtime.txt')
t2 = g[:,0]/60.
y = g[:,1]*6/5




cal = np.genfromtxt(R"C:\Users\mtsit\reserch\data\calc_PSO.txt")

plt.figure(figsize=(15.,5.))
plt.subplot(1,2,1)

plt.get_current_fig_manager().window.wm_geometry("+0+100")

plt.xlim([1540/60,4080/60])
plt.ylim([0.45,0.60])
plt.plot(t, P,'o')
plt.plot(t2,y)
plt.plot(cal[:,0]/60.,cal[:,1],'-')



plt.xlabel('time [h]')
plt.ylabel('Polarization')
#plt.legend(['Row data','PSO'])
plt.legend(['Row data'])
#plt.legend(['Row data','Gauss-Newton','Levenberg-Marquardt'])

plt.subplot(1,2,2)

plt.plot(optimizer.cost_history,'-')
plt.xlabel('iterations')
plt.ylabel('cost')



"""
#plot_cost_history(cost_history=optimizer.cost_history)
plt.subplots(1, 2, 2)
iters = len(optimizer.cost_history)
designer = formatters.Designer(legend="Cost", label=["Iterations", "Cost"])
_, ax = plt.subplots(1, 1, figsize=designer.figsize)
ax.plot(np.arange(iters), optimizer.cost_history, "k", lw=2, label=designer.legend )
"""
#print(cost_fig)
plt.show()


#緩和項比較-------------------------------------------------------
t_list = []
TL_list = []
sekibun_list = []
beta_list = []
TL_beta_list = []
TL_beta_sekibun_list = []

#緩和項比較は[1/h]単位で行う
alpha = (10**alpha)*60.
beta = (10**beta)*60.
#gammaは単位が[min]だったので[h]にするために/60する
gamma = gamma/60.


for t in range(25,70):
    t_list.append(t)

    #beam3の強度をcpm(=cps * 60.)で与え、beam1,2はそこにBC比を掛けて計算
    #beam強度はcp(hour)で計算 = cps * 3600
    if t<2189./60.:
        I = 0.
        S = 0.
        t_s = t
        gamma_2 = -1.
    elif t<2649./60.:
        I = 3.41E6*60./24.4 *60.
        S = 0.
        t_s = t-2194/60.
        gamma_2 = 1.
        t_start = 2189./60.
    elif t<3478./60.:
        I = 0.
        S = alpha * 3.41E6*60./24.4 *60. * (2649.-2189.)/60.
        t_s = t-2649./60.
        gamma_2 = 0.
        t_start = 2649./60.
    #12/27
    
    elif t<3620./60.:
        I = 3.41E6*60./24.4*4.46 *60.
        S = alpha * 3.41E6*60./24.4 *60. * (2649.-2189.)/60.
        t_s = t-3478./60.
        gamma_2 = 1.
        t_start = 3478./60.
    
    elif t<4083/60.:
        I = 3.41E6*60. *60.
        S = alpha * (3.41E6*60./24.4 *60. * (2649.-2189.)/60. + 3.41E6*60./24.4*4.46*60. * (3620-3478)/60.)
        t_s = t-3620. /60.
        gamma_2 = 1.
        t_start = 3478./60.

    sekibun_list.append(S+alpha*I*t_s)
    if gamma_2 == -1:
        beta_list.append(0)
    if gamma_2 == 0:
        beta_list.append(beta*(np.exp(-(t-t_start)/gamma)))
    if gamma_2 == 1:
        beta_list.append(beta*(1-np.exp(-(t-t_start)/gamma)))
    #TLは[h]で与えているため、minに合わせる
    TL_list.append(1/(TL*60. /60.))

for i in range(len(TL_list)):
    TL_beta_list.append(TL_list[i]+beta_list[i])
    TL_beta_sekibun_list.append(TL_list[i]+beta_list[i]+sekibun_list[i])


    
plt.xlim([1540/60.,4080/60.])
plt.ylim([0,0.1])
plt.plot(t_list, TL_list,'-')
plt.plot(t_list,TL_beta_list,'-')
plt.plot(t_list,TL_beta_sekibun_list,'-')
plt.xlabel('t [h]')
plt.ylabel('gamma [/h]')
plt.legend(['1/TL','beta','alpha'])
plt.show()


#----------------------------------------------------------------------------
#長時間照射したときの減偏極をプロット
t_end = 50 * 24 *60
Pcal = longtime_plot.long_plot(TD,TL,alpha,beta,gamma,t_end)





"""""
# Initialize mesher with sphere function
m = Mesher(func=fx.sphere)
# capture
# Make animation
animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))

# Enables us to view it in a Jupyter notebook
animation.save('plot0.gif', writer='pillow', fps=100)
Image(url='plot0.gif')
print(optimizer.pos_history[1])
pos_history_2dim = []
print(len(optimizer.pos_history))
"""

#optimizer.pos_historyは4×particleの2次元リストがiteration(計算回数)個格納されたリストになっている
#プロットするにはパラメータを4→2にしなければいけないので、列を2に直す必要がある
pos_hist_2dim = []



"""""
#まずは各iter毎に分解
for i in range(len(optimizer.pos_history)):
    pos_hist = optimizer.pos_history[i]
    #2次元リストを定義
    pos_past_array = [[0] * 2 for i in range(len(pos_hist))]
    pos_past = [[0] * 2 for j in range(len(pos_hist))]
    for k in range(len(pos_hist)):
        
        TD = pos_hist[k][0]/60.
        TL = pos_hist[k][1]/60.
        alpha = pos_hist[k][2]
        beta = pos_hist[k][3]
        gamma = pos_hist[k][4]
        delta = pos_hist[k][5]
        epsilon = pos_hist[k][6]

        pos_past[k][0] = alpha
        pos_past[k][1] = beta
        #2次元リストをarrayの中に入れる
        pos_past_array = list(pos_past_array)
        pos_past_array[k] = pos_past[k]
        
        #print(pos_past_array)
    pos_past_array = np.array(pos_past_array)
    #print(pos_past_array)
    pos_hist_2dim.append(pos_past_array)
#print(pos_hist_2dim)


#print("##################################################################")
#print(optimizer.pos_history)

#
# Make animation





#plot範囲を変えたい場合は、plot_contourを右クリックして宣言へ移動し、designerから変更する
#引数としてここから入力できるはずなので、そのうちできるようにしたい
animation = plot_contour(pos_history=pos_hist_2dim,title = "alpha - beta")

# Enables us to view it in a Jupyter notebook
animation.save('alpha-beta.gif', writer='pillow', fps=10)
Image(url='plot0.gif')
"""

