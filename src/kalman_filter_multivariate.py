#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math

#https://jp.mathworks.com/help/control/ug/state-estimation-using-time-varying-kalman-filter.html
#https://ocwx.ocw.u-tokyo.ac.jp/course_11416/

# %%
#状態モデル x = Fx + Gv, v ~ N(0, Q)
#観測モデル y = Hx + w, w ~ N(0, R)

#状態ベクトルx 位置(2D)と速度(2D)の計4次元
DIM_STATE = 4   #状態ベクトルの次元
#観測ベクトルy 位置(2D)の計2次元
DIM_OBS = 2     #観測ベクトルの次元

#状態遷移行列
F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
#観測行列
H = np.array([[1,0,0,0],[0,1,0,0]])

G = np.array([[1/2,0],[0,1/2],[1,0],[0,1]])
I = np.identity((DIM_STATE))#単位行列

#%%
# 実験データ作成
N = 100
mean = np.zeros([DIM_OBS])
covQ = np.array([[1,0],[0,1]])
v = np.random.multivariate_normal(mean, covQ, size = N)
mean = np.zeros([DIM_OBS])
covR = np.array([[100,0],[0,100]])
w = np.random.multivariate_normal(mean, covR, size = N)

x0 = [[0],[0],[0],[0]]

x = np.zeros([N, DIM_STATE, 1])
y = np.zeros([N, DIM_OBS, 1])

x[0] = F @ x0 + G @ v[0].reshape([DIM_OBS, 1])
y[0] = H @ x[0] + w[0].reshape([DIM_OBS, 1])
for n in range(1, N):
    x[n] = F @ x[n-1] + G @ v[n].reshape([DIM_OBS, 1])
    y[n] = H @ x[n] + w[n].reshape([DIM_OBS, 1])

plt.plot(range(N), y[:,0])

#観測値を格納しておく
df = pd.DataFrame(range(N))
df[1] = y[:, 0]
df[2] = y[:, 1]

observed_value = df[[1,2]].values

#%%
def kalman_filter(param, mx_nn, vx_nn, optimize=False):
    Q = np.array([[param[0], 0],[0,param[1]]])
    R = np.array([[param[2], 0],[0,param[3]]])

    ld1 = []
    ld2=[]
    val = []
    for n in range(N):
        mx_nbnb = mx_nn
        vx_nbnb = vx_nn

        #一期先予測
        mx_nnb = F @ mx_nbnb
 
        vx_nnb = F @ vx_nbnb @ F.T + G @ Q @ G.T
        
        y = np.array([observed_value[n]])

        #カルマンゲイン
        tmp = H @ vx_nnb @ H.T + R
        K = vx_nnb @ H.T @ np.linalg.pinv(tmp)
        #K = (np.linalg.solve(tmp.T, H @ vx_nnb.T)).T

        #フィルタ
        mx_nn = mx_nnb + K @ (y.T - H @ mx_nnb)
        vx_nn = (I - K @ H) @ vx_nnb

        #epsi
        epsi = y.T - H @ mx_nnb

        #d_nnb
        d_nnb = H @ vx_nnb @ H.T + R

        ld1.append(epsi.tolist())
        ld2.append(d_nnb.tolist())
        val.append(mx_nn.tolist())

    if optimize == True:
        nld1 = np.array(ld1)
        nld2 = np.array(ld2)

        lik = -1/2*(N * math.log(2*math.pi) + np.log(nld2[:,0,0]).sum()\
            + (nld1[:,0,0]*nld1[:,0,0]/nld2[:,0,0]).sum())\
                -1/2*(N * math.log(2*math.pi) + np.log(nld2[:,1,1]).sum()\
            + (nld1[:,1,0]*nld1[:,1,0]/nld2[:,1,1]).sum())\
        
        return (-1) * lik
    else:
        return val
#%%
#フィルタリング分布の初期値
mx_nn_0 = np.array([[0],[0],[0],[0]])
vx_nn_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

param = np.array([10,10,10,10])
param_pred = fmin(kalman_filter, param, args=(mx_nn_0,vx_nn_0,True,))
# %%
res_val = kalman_filter(param_pred, mx_nn_0,vx_nn_0)

# %%
plt.plot(range(N)[:N-1], np.array(observed_value)[:,0][:N-1],label="obs")
plt.plot(range(N)[:N-1], np.array(res_val)[:,0:2,0][:,0][:N-1],label="fiter")


# %%
plt.plot(range(N)[:N-1], np.array(observed_value)[:,1][:N-1],label="obs")
plt.plot(range(N)[:N-1], np.array(res_val)[:,0:2,0][:,1][:N-1],label="fiter")


# %%
param_pred
# %%
