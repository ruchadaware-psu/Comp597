import matplotlib.pyplot as plt
import pandas as pd

#read the csv files
df = pd.read_csv('shows3.csv')


x=list(df['frame'])
print(list(x))
b=["right_elbow_angle","right_shoulder_angle","right_wrist_angle","left_elbow_angle","left_shoulder_angle","left_wrist_angle"]
for z in b:
    y=list(df[z])
    plt.plot(x,y)
    file_name=f"{z}_graph"
    plt.savefig(f'{file_name}.png')
    plt.clf()