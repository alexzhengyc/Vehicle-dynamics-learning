import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

directory = '4.29-3/'
os.mkdir("Data/" + directory)
os.mkdir("Picture/" + directory)

# df = pd.read_csv('/home/alexzheng/Documents/GitHub/AutonomousRacing-simulator/' + directory + 'raw.csv')
# df.columns = ['x', 'y', 'psi', 'throttle', 'steering', 'throttle_num', 'steering_num']
# print(df.dtypes)

# # 去掉开头无控制
# for i in range(df.shape[0]):
#     if df.iloc[i, 5] != 0:
#         break
# df = df[i:]
#
# # 去掉结尾无控制
# for i in range(df.shape[0]):
#     if df.iloc[i, 5] == -1 and df.iloc[i, 6] == -1:
#         break
#
# df = df[:i]

# df.to_csv('/home/alexzheng/Documents/GitHub/AutonomousRacing-simulator/' + directory + 'pose.csv')
# df.to_csv('Data/' + directory + 'pose-ctrl.csv')
#
# # 差分计算速度
# df = df.iloc[:, :5]
# df1 = df.copy()
# df1.columns = ['vx', 'vy', 'omega', 'throttle', 'steering']
#
# for i in range(1, df.shape[0] - 1):
#     dotX = (df.iloc[i, 0] - df.iloc[i - 1, 0]) * 50
#     dotY = (df.iloc[i, 1] - df.iloc[i - 1, 1]) * 50
#     if df.iloc[i, 2] - df.iloc[i - 1, 2] > 3.14:
#         omega = (df.iloc[i, 2] - df.iloc[i - 1, 2] - 2 * np.pi) * 50
#     elif df.iloc[i, 2] - df.iloc[i - 1, 2] < -3.14:
#         omega = (df.iloc[i, 2] - df.iloc[i - 1, 2] + 2 * np.pi) * 50
#     else:
#         omega = (df.iloc[i, 2] - df.iloc[i - 1, 2]) * 50
#
#     df1.iloc[i, 0] = dotX * np.cos(df.iloc[i, 2]) + dotY * np.sin(df.iloc[i, 2])
#     df1.iloc[i, 1] = -dotX * np.sin(df.iloc[i, 2]) + dotY * np.cos(df.iloc[i, 2])
#     df1.iloc[i, 2] = omega
#
# df1 = df1.iloc[1:-1]
# df1.to_csv('/home/alexzheng/Documents/GitHub/AutonomousRacing-simulator/' + directory + 'velocity.csv')
# df1.to_csv('Data/' + directory + 'vel-ctrl.csv')

df = pd.read_csv('../Files/data.csv')
df.to_csv('Data/' + directory + 'vel-ctrl.csv')
print(df.dtypes)

sns.set_theme(style="ticks")

vx_vy = sns.JointGrid(data=df, x='vx', y='vy', marginal_ticks=True)
vx_vy.plot(sns.scatterplot, sns.histplot)
vx_vy.savefig("Picture/" + directory + "vx-vy.png")

th_st = sns.JointGrid(data=df, x='throttle', y='steering', marginal_ticks=True)
th_st.plot(sns.scatterplot, sns.histplot)
th_st.savefig("Picture/" + directory + "th-st.png")
