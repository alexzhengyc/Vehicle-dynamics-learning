import numpy as np
import csv
from car_model import TinyCar_model
from car_model import FormulaCar_model

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

directory = 'Formula-3/'
os.mkdir("Data/" + directory)
os.mkdir("Picture/" + directory)

NUM = 1500

min_steering = -1.0
max_steering = 1.0
min_throttle = -1
max_throttle = 1.0

if __name__ == '__main__':

    vx = 5
    vy = 0
    omega = 0
    dvx = 0
    dvy = 0
    dr = 0

    throttle = 0
    steering = 0

    model = FormulaCar_model()
    # model = TinyCar_model()


with open("../Files/data.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["vx", "vy", "omega", "throttle", "steering", "dvx", "dvy", "dr"])

    for i in range(NUM):
        for j in range(25):

            rand = np.random.rand()
            if steering > 0.8:
                rand = np.random.rand() - 0.2
            elif steering < -0.8:
                rand = np.random.rand() + 0.2

            ang_range = max_steering / 2.
            rand_ang = rand * ang_range - ang_range / 2.
            steering = min(max(steering + rand_ang, min_steering), max_steering)

            writer.writerow([vx, vy, omega, throttle, steering, dvx, dvy, dr])
            vx, vy, omega, dvx, dvy, dr = model.update(vx, vy, omega, throttle, steering)

        if vx > 30:
            rand = np.random.rand() - vx / 50
        elif vx < 5:
            rand = np.random.rand() + (5-vx)/5

        if throttle > 0.8:
            rand = rand - 0.2
        elif throttle < -0.8:
            rand = rand + 0.2
        rand = min(max(rand, 0), 1)
        # rand = np.random.rand()
        thr_range = max_throttle / 2
        rand_thr = rand * thr_range - thr_range / 2.0
        throttle = min(max(throttle + rand_thr, min_throttle), max_throttle)

    for i in range(NUM):
        for j in range(25):

            if vx > 30:
                rand = np.random.rand() - vx / 50
            elif vx < 5:
                rand = np.random.rand() + (5 - vx) / 5

            if throttle > 0.8:
                rand = rand - 0.2
            elif throttle < -0.8:
                rand = rand + 0.2
            rand = min(max(rand, 0), 1)
            # rand = np.random.rand()
            thr_range = max_throttle / 2.
            rand_thr = rand * thr_range - thr_range / 2.
            throttle = min(max(throttle + rand_thr, min_throttle), max_throttle)

            writer.writerow([vx, vy, omega, throttle, steering, dvx, dvy, dr])
            vx, vy, omega, dvx, dvy, dr = model.update(vx, vy, omega, throttle, steering)

        rand = np.random.rand()
        if steering > 0.8:
            rand = np.random.rand() - 0.2
        elif steering < -0.8:
            rand = np.random.rand() + 0.2

        ang_range = max_steering / 2.
        rand_ang = rand * ang_range - ang_range / 2.0
        steering = min(max(steering + rand_ang, min_steering), max_steering)

    for i in range(NUM * 5):
        for j in range(5):
            if vx > 30:
                rand = np.random.rand() - vx / 50
            elif vx < 5:
                rand = np.random.rand() + (5 - vx) / 5

            if throttle > 0.8:
                rand = rand - 0.2
            elif throttle < -0.8:
                rand = rand + 0.2
            rand = min(max(rand, 0), 1)
            # rand = np.random.rand()
            thr_range = max_throttle / 2.
            rand_thr = rand * thr_range - thr_range / 2.0
            throttle = min(max(throttle + rand_thr, min_throttle), max_throttle)

            writer.writerow([vx, vy, omega, throttle, steering, dvx, dvy, dr])
            vx, vy, omega, dvx, dvy, dr = model.update(vx, vy, omega, throttle, steering)

        for j in range(5):

            rand = np.random.rand()
            if steering > 0.8:
                rand = np.random.rand() - 0.2
            elif steering < -0.8:
                rand = np.random.rand() + 0.2

            ang_range = max_steering / 2.0
            rand_ang = rand * ang_range - ang_range / 2.0
            steering = min(max(steering + rand_ang, min_steering), max_steering)

            writer.writerow([vx, vy, omega, throttle, steering, dvx, dvy, dr])
            vx, vy, omega, dvx, dvy, dr = model.update(vx, vy, omega, throttle, steering)

    df = pd.read_csv('../Files/data.csv')
    df.to_csv('Data/' + directory + 'data_true.csv')
    print(df.dtypes)

    max_value = df.values.max(0)
    min_value = df.values.min(0)
    df1 = pd.DataFrame(np.vstack((max_value, min_value)), columns=["vx", "vy", "r", "throttle", "steering", "dvx", "dvy", "dr"])
    df1.to_csv('Data/' + directory + 'normalize.csv')

    sns.set_theme(style="ticks")

    vx_vy = sns.JointGrid(data=df, x='vx', y='vy', marginal_ticks=True)
    vx_vy.plot(sns.scatterplot, sns.histplot)
    vx_vy.savefig("Picture/" + directory + "vx-vy.png")

    th_st = sns.JointGrid(data=df, x='throttle', y='steering', marginal_ticks=True)
    th_st.plot(sns.scatterplot, sns.histplot)
    th_st.savefig("Picture/" + directory + "th-st.png")

    th_st = sns.JointGrid(data=df, x='dvx', y='dvy', marginal_ticks=True)
    th_st.plot(sns.scatterplot, sns.histplot)
    th_st.savefig("Picture/" + directory + "dvx-dvy.png")




