import numpy as np
from car_model import TinyCar_model
from car_model import zonda_model
import torch
import pandas as pd
import seaborn as sns

# car = TinyCar_model()
# car = zonda_model()
car_para = zonda_model()

model = torch.load("Model/Tire1-data1/5/MLP2plus_16.pth")
# model = torch.jit.load("Model/TinyCar-1.pt")
norm = pd.read_csv("Data/Tire1-data1/normalize.csv")
data = np.array([[0, 0, 0]])


def normalize(data):
    max_value = norm.iloc[0, :5]
    min_value = norm.iloc[1, :5]
    for i in range(data.shape[0]):
        data[i] = (data[i] - min_value) / (max_value - min_value)
    return data


def denormalize(data):
    max_value = norm.iloc[0, -3:]
    min_value = norm.iloc[1, -3:]
    data = torch.detach(data).numpy()
    for i in range(data.shape[0]):
        data[i] = np.dot(data[i], max_value - min_value) + min_value
    return data


for rear_slip in range(0, 90, 2):
    for vx in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for vy in [0, 0.5, 1.0, -0.5, -1.0]:
            r = (np.tan(rear_slip/100) * vx + vy) / car_para.lr
            input_np = np.array([[vx, vy, r, 0.2, 0]])
            input_np = normalize(input_np)
            input_tensor = torch.tensor(input_np)
            input_tensor = input_tensor.unsqueeze(0).float()

            # print(input_tensor.shape)
            output_tensor = model(input_tensor)
            output_tensor = output_tensor.squeeze(0)
            # print(output_tensor.shape)

            output = denormalize(output_tensor)
            dvy = output[0, 1]
            dr = output[0, 2]

            Fry = (car_para.lf * (car_para.m * dvy + car_para.m * vx * r) - car_para.Iz * dr) \
                    (car_para.lf+car_para.lr)
            data.append(rear_slip, Fry, vx)

print(data.shape)
df = pd.DataFrame(data, columns=["slip_angle", "Fry", "vx"])
fig = sns.lineplot(x="slip_angle", y="Fry", hue="vx", data=df).get_figure()
fig.savefig("F_ry.jpg")

