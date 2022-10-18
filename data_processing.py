import numpy as np
import pandas as pd
import seaborn as sns
import time
import os

min_vx = 0.02
e = 1e-9
steps = 5

# read_csv
directory = "8.9-2"
# os.mkdir("../Picture/" + directory)

rawdata = pd.read_csv("../Data/"+directory+"/rawdata.csv")

N1 = rawdata.shape[0]
print(f"Number before processing: {N1}")


columns1 = []
columns2 = []
columns3 = []
for i in range(steps):
    columns1 = np.concatenate((columns1, ["vx", "vy", "r", "throttle", "steering"]))
    columns2 = np.concatenate((columns2, ["vx", "vy", "r"]))
    columns3 = np.concatenate((columns3, ["throttle", "steering"]))

columns1 = np.concatenate((columns1, ["dvx", "dvy", "dr"]))
columns3 = np.concatenate((columns3, ["dvx", "dvy", "dr"]))
columns2 = np.concatenate((columns2, columns3))

random_data = np.random.rand(N1, 5*steps+3)
df_read = pd.DataFrame(data=random_data, columns=columns1)
df_use = pd.DataFrame(data=random_data, columns=columns2)

time_begin = time.time()
cnt = 0
for i in range(2, N1-steps-2):
    error = 0
    for j in range(steps):
        if np.abs(rawdata.iloc[i+j*2, 2] - rawdata.iloc[i+j*2-2, 2]) <= e:       # x not  change
            error = 1
            break
        elif rawdata.iloc[i+j*2, 5] < min_vx:                                    # vx < min_vx
            error = 1
            break

    if error == 0:
        for j in range(steps):
            df_read.iloc[cnt, j*5: j*5+5] = rawdata.iloc[i+j*2, 5:10]                 # vx vy r th st
            df_use.iloc[cnt, j*3: j*3+3] = rawdata.iloc[i+j*2, 5:8]                   # vx vy r
            df_use.iloc[cnt, 3*steps+j*2: 3*steps+j*2+2] = rawdata.iloc[i+j*2, 9:10]  # th st

        df_read.iloc[cnt, 5*steps:5*steps+3] \
            = (rawdata.iloc[i+steps*2, 5:8] - rawdata.iloc[i+steps*2-2, 5:8]) * 50   # dvx dvy dr
        df_use.iloc[cnt, 5*steps:5*steps+3] \
            = (rawdata.iloc[i + steps * 2, 5:8] - rawdata.iloc[i + steps * 2 - 2, 5:8]) * 50  # dvx dvy dr
        cnt += 1

    # print progress
    time_now = time.time()
    time_pass = time.time() - time_begin
    if i % 500 == 0:
        print(f"time: {time_pass:.4} s -- progress: {i} / {N1}")

df_read = df_read.iloc[:cnt, :]
df_use = df_use.iloc[:cnt, :]

print(f"Number after processing: {cnt}")

df_read.to_csv("../Data/" + directory + "/data_read_" + str(steps) + ".csv")
df_use.to_csv("../Data/" + directory + "/data_use_" + str(steps) + ".csv")

# X = ekf.iloc[begin, 5]
# Y = ekf.iloc[begin, 6]
# [x, y, z, w] = ekf.iloc[begin, 8:12]
# [x2, y2, z2, w2] = ekf.iloc[begin+4, 8:12]
# Psi = quat2euler(x, y, z, w)
# Psi2 = quat2euler(x2, y2, z2, w2)
#
# df.iloc[i, 1] = X                                          # X
# df.iloc[i, 2] = Y                                          # Y
# df.iloc[i, 3] = Psi                                        # Psi
#
# dt = ekf.iloc[begin+4, 0] - ekf.iloc[begin, 0]
# print(dt)
#
# vx = ekf.iloc[begin, 48]
# vy = ekf.iloc[begin, 49]
#
# df.iloc[i, 4] = max(min_vx, vx * np.cos(Psi) + vy * np.sin(Psi))    # vx
# df.iloc[i, 5] = -vx * np.sin(Psi) + vy * np.cos(Psi)                # vy
#
# dPsi = Psi2 - Psi
# if dPsi > np.pi:
#     dPsi = -2 * np.pi + dPsi
# elif dPsi < -np.pi:
#     dPsi = 2 * np.pi + dPsi
#
# df.iloc[i, 6] = dPsi / dt * 1e9                                           # r
#
# df.iloc[i, 9] = (df.iloc[i, 4] - df.iloc[i-1, 4]) / df.iloc[i, 0] * 1e9   # dvx
# df.iloc[i, 10] = (df.iloc[i, 5] - df.iloc[i-1, 5]) / df.iloc[i, 0] * 1e9  # dvy
# df.iloc[i, 11] = (df.iloc[i, 6] - df.iloc[i-1, 6]) / df.iloc[i, 0] * 1e9  # dr

# print progress
# time_now = time.time()
# time_pass = time.time() - time_begin
# if i % 500 == 0:
#     print(f"time: {time_pass:.4} s -- progress: {i} / {N1}")

# **************************************************************************8
# delete LOW SPEED period
# move = []
# linkoff = []
# for i in range(N1):
#     if df.iloc[i, 4] >= min_vx:
#         if i == 0 or np.abs(df.iloc[i, 1] - df.iloc[i-1, 1]) > e:
#             move.append(i)
#         else:
#             linkoff.append(i)

# print(linkoff)
#
# df = df.iloc[move, :]
# df = df.iloc[2:-2, :]
# N3 = df.shape[0]
# print(f"Number after processing: {N3}")
#
# df.to_csv("../Data/" + directory + "/data_with_time.csv")
# df1 = df.iloc[:, 1:]
# df1.to_csv("../Data/" + directory + "/data.csv")

# sns.set_theme(style="ticks")
# vx_vy = sns.JointGrid(data=df, x='vx', y='vy', marginal_ticks=True)
# vx_vy.plot(sns.scatterplot, sns.histplot)
# vx_vy.savefig("../Picture/" + directory + "/vx-vy-jointgrid .png")
#
# th_st = sns.JointGrid(data=df, x='throttle', y='steering', marginal_ticks=True)
# th_st.plot(sns.scatterplot, sns.histplot)
# th_st.savefig("../Picture/" + directory + "/th-st-jointgrid.png")
#
# dvx_dvy = sns.JointGrid(data=df, x='dvx', y='dvy', marginal_ticks=True)
# dvx_dvy.plot(sns.scatterplot, sns.histplot)
# dvx_dvy.savefig("../Picture/" + directory + "/dvx-dvy-jointgrid.png")
#
# st_r = sns.JointGrid(data=df, x='steering', y='r', marginal_ticks=True)
# st_r.plot(sns.scatterplot, sns.histplot)
# st_r.savefig("../Picture/" + directory + "/st-r-jointgrid.png")


