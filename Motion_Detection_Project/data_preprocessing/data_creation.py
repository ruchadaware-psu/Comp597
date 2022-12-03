import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.color import rgb2gray,rgba2rgb
from skimage.transform import rescale
from skimage import filters,exposure
import os
from os import listdir
from skimage import color
from skimage import img_as_ubyte

folder_dir = "E:/GRE Jamboree/PENN STATE/Study/Neural Network and Deep Learning/project/data/data_1"
folder_dir_1 = "E:/GRE Jamboree/PENN STATE/Study/Neural Network and Deep Learning/project/data/data_generated/"
folder_dir_2 = "E:/GRE Jamboree/PENN STATE/Study/Neural Network and Deep Learning/project/data/data_stored/"
name_of_file=""




# design a function that gives the longest min to max slope

def slope(angle_list):
    slopes=[]
    x=0
    
    while x<(len(angle_list)-2):
        #print("first while ",x)
        start=x
        while angle_list[x]<angle_list[x+1] and x<(len(angle_list)-2):
            x+=1
            #print("second while",x)
        end=x
        if start==end:
            x+=1
        else:
            slopes.append(list(angle_list[start:end+1]))
            x+=1

    max=0
    z=0
    req_index=0

    for p in slopes:
        diff=p[-1]-p[0]
        if diff>max:
            max=diff
            req_index=z
        z+=1




    return slopes[req_index]

def normalize(joint_slope,og_list):
    normalized_slope=[]
    a=min(og_list)
    b=max(og_list)
    y=min(joint_slope)
    z=max(joint_slope)

    for x in joint_slope:
        ans=(b-a)*((x-y)/(z-y))+a
        normalized_slope.append(ans)
    return normalized_slope

def graph_create(final_slope):
    for y in final_slope:
        plt.plot([x for x in range(len(y))], y, linewidth=1)
        plt.xlim([0, 50])
        plt.ylim([-360, 360])
    text_name=name_of_file[:-4]+".png"

    #removes axis and borders
    plt.axis('off')

    plt.savefig(f"{folder_dir_2}{text_name}",dpi=300)
    plt.clf()
    print(f"{folder_dir_2}{text_name}")
    feature_extraction(f"{folder_dir_2}{text_name}")
    

def feature_extraction(data):
    image=imread(data)
    image=rgba2rgb(image)
    image_g=rgb2gray(image)
    image_l= exposure.adjust_gamma(image_g, 1)
    print(data)
    #edge_sobel = filters.sobel(image_g)
    #edge_sobel_r = rescale(edge_sobel, 0.25, anti_aliasing=True)
    newtext=data.replace("E:/GRE Jamboree/PENN STATE/Study/Neural Network and Deep Learning/project/data/data_stored/","")
    imsave(f"{folder_dir_1}{newtext}",img_as_ubyte(image_l))

def data_generation(name_of_files,dir):
    global name_of_file
    name_of_file=name_of_files
    df=pd.read_csv(f"{dir}/{name_of_files}")
    right_elbow_angle=df['right_elbow_angle']
    right_shoulder_angle=df['right_shoulder_angle']
    right_wrist_angle=df['right_wrist_angle']
    left_elbow_angle=df['left_elbow_angle']
    left_shoulder_angle=df['left_shoulder_angle']
    left_wrist_angle=df['left_wrist_angle']

    # getting the required slope
    right_elbow_angle_slope=slope(right_elbow_angle)
    right_shoulder_angle_slope=slope(right_shoulder_angle)
    right_wrist_angle_slope=slope(right_wrist_angle)
    left_elbow_angle_slope=slope(left_elbow_angle)
    left_shoulder_angle_slope=slope(left_shoulder_angle)
    left_wrist_angle_slope=slope(left_wrist_angle)

    #normalizing the slope

    right_elbow_angle_slope_normalized=normalize(right_elbow_angle_slope,right_elbow_angle)
    right_shoulder_angle_slope_normalized=normalize(right_shoulder_angle_slope,right_shoulder_angle)
    right_wrist_angle_slope_normalized=normalize(right_wrist_angle_slope,right_wrist_angle)
    left_elbow_angle_slope_normalized=normalize(left_elbow_angle_slope,left_elbow_angle)
    left_shoulder_angle_slope_normalized=normalize(left_shoulder_angle_slope,left_shoulder_angle)
    left_wrist_angle_slope_normalized=normalize(left_wrist_angle_slope,left_wrist_angle)


    final_slope=[right_elbow_angle_slope_normalized,
    right_shoulder_angle_slope_normalized,
    right_wrist_angle_slope_normalized,
    left_elbow_angle_slope_normalized,
    left_shoulder_angle_slope_normalized,
    left_wrist_angle_slope_normalized]

    graph_create(final_slope)


for images in os.listdir(folder_dir):
    data_generation(images,folder_dir)



#creating the graphs
# graph_create(right_elbow_angle_slope_normalized,"right_elbow_angle")
# graph_create(right_shoulder_angle_slope_normalized,"right_shoulder_angle")
# graph_create(right_wrist_angle_slope_normalized,"right_wrist_angle")
# graph_create(left_elbow_angle_slope_normalized,"left_elbow_angle")
# graph_create(left_shoulder_angle_slope_normalized,"left_shoulder_angle")
# graph_create(left_wrist_angle_slope_normalized,"left_wrist_angle")






# design a function to neutralize the slope
# convert it into images