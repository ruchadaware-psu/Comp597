import os 
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def create_data():

    folder_dir_1 = "E:/GRE Jamboree/PENN STATE/Study/Neural Network and Deep Learning/project/data_preprocessing/data_generated_rgb/"
    training_data=[]
    x_train=[]
    y_train=[1,1,0,0,0,0,0,0,0,0]
    x_test=[]
    y_test=[1,0]

    for img in os.listdir(folder_dir_1):
        pic=cv2.imread(os.path.join(folder_dir_1,img))
        training_data.append(pic)

    for train in range(len(training_data)):
        if train==2 or train==9:
            x_test.append(training_data[train])
        else:
            
            x_train.append(training_data[train])

    

    return x_train,y_train,x_test,y_test

create_data()


# np_data=np.array(training_data)
# df=pd.DataFrame(training_data)
# print(np_data.shape)



#plt.imshow(np.array(training_data[0]).reshape(1440, 1920, 3))
#plt.show()

