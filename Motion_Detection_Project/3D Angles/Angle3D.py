from cmath import acos
import cv2
import mediapipe as mp
import numpy as np
import math 
import time
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

data=[]
data2=[]
def capture():
    global data
    frame_1=0
    cap = cv2.VideoCapture(0)
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            
            ret, frame = cap.read()
            frame_1+=1
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            temp=[]
            temp.append(frame_1)
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                temp.append('RIGHT_WRIST')
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].z)

                temp.append('RIGHT_ELBOW')
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].z)

                temp.append('RIGHT_SHOULDER')
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

                temp.append('RIGHT_HIP')
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z)

                temp.append('LEFT_WRIST')
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].z)

                temp.append('LEFT_ELBOW')
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].z)

                temp.append('LEFT_SHOULDER')
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z)

                temp.append('LEFT_HIP')
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_HIP].z)

                temp.append('RIGHT_PINKY')
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].z)

                temp.append('RIGHT_INDEX')
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y)
                temp.append(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].z)

                temp.append('LEFT_PINKY')
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_PINKY].y)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_PINKY].z)

                temp.append('LEFT_INDEX')
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y)
                temp.append(landmarks[mp_pose.PoseLandmark.LEFT_INDEX].z)

                temp.append("\n")
                data.append(temp)
                
                
            except:
                pass
            
            
            #Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)
            
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    #print(data)
    #df = pd.DataFrame(data)
    #df.to_csv('shows.csv')


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
    ab_vector  = [b[0]-a[0], b[1]-a[1], b[2]-a[2]]
    bc_vector = [c[0]-b[0], c[1]-b[1], b[2]-c[2]]

    dot_product = (ab_vector[0] * bc_vector[0],
                    ab_vector[1] * bc_vector[1],
                    ab_vector[2] * bc_vector[2])

    ab_magnitude = math.sqrt(ab_vector[0]*ab_vector[0],
                                ab_vector[1]*ab_vector[1],
                                ab_vector[2]*ab_vector[2])

    bc_magnitude = math.sqrt(bc_vector[0]*bc_vector[0],
                                bc_vector[1]*bc_vector[1],
                                bc_vector[2]*bc_vector[2])

    angle = acos(dot_product/(ab_magnitude * bc_magnitude))
    return angle

def finalfile():
    header=[]
    header.append('frame')
    global data
    global data2
    for x in data:
        temp=[]
        print(x[0])
        temp.append(x[0])
        
        if 'right_elbow_angle' not in header:
            header.append('right_elbow_angle')
        r_ang_elbow=getAngle([x[10],x[11],x[12]],[x[6],x[7],x[8]],[x[2],x[3],x[4]])
        temp.append(r_ang_elbow)

        
        
        r_shoulder_ang=getAngle([x[14],x[15],x[16]],[x[10],x[11],x[12]],[x[6],x[7],x[8]])
        if 'right_shoulder_angle' not in header:
            header.append('right_shoulder_angle')
        temp.append(r_shoulder_ang)
        
        
        
        avg_x = (x[34]+x[38])/2
        avg_y = (x[35]+x[39])/2
        avg_z = (x[36]+x[40])/2
        r_wrist_ang=getAngle([x[6],x[7],x[8]],[x[2],x[3],x[4]],[avg_x,avg_y,avg_z])
        if 'right_wrist_angle' not in header:
            header.append('right_wrist_angle')
        temp.append(r_wrist_ang)
        

        
        l_ang_elbow=getAngle([x[26],x[27],x[28]],[x[22],x[23],x[24]],[x[18],x[19],x[20]])
        if 'left_elbow_angle' not in header:
            header.append('left_elbow_angle')
        temp.append(l_ang_elbow)
        
        
        
        l_shoulder_ang=getAngle([x[30],x[31],x[32]],[x[26],x[27],x[28]],[x[22],x[23],x[24]])
        if 'left_shoulder_angle' not in header:
            header.append('left_shoulder_angle')
        if l_shoulder_ang > 200:
            l_shoulder_ang = 360 - l_shoulder_ang
        temp.append(l_shoulder_ang)
      
        
        
        avg_x = (x[42]+x[46])/2
        avg_y = (x[43]+x[47])/2
        avg_z = (x[44]+x[48])/2
        l_wrist_ang=getAngle([x[22],x[23],x[24]],[x[18],x[19],x[20]],[avg_x,avg_y,avg_z])
        if 'left_wrist_angle' not in header:
            header.append('left_wrist_angle')
        temp.append(l_wrist_ang)
        if header not in data2:
            data2.append(header)
        data2.append(temp)
        



capture()
finalfile()
df = pd.DataFrame(data2)
df.to_csv('shows3.csv')


