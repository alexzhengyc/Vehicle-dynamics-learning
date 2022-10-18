import numpy as np
import pandas as pd

dir1 = "Zonda-2"
dir2 = "Zonda-3"
dir3 = "Zonda-drifting"

df1 = pd.read_csv("../Data/" + dir1 + "/data.csv")
df2 = pd.read_csv("../Data/" + dir2 + "/data.csv")
df3 = pd.read_csv("../Data/" + dir3 + "/data.csv")

df = pd.concat([df1, df2, df3])
df = df.iloc[:, 4:]
df = df[1:]

df.to_csv("../Data/Tire1-data1/data.csv")

