import cv2
import numpy as np
from PIL import Image
from filterpy.kalman import KalmanFilter

# function to segment object based on color 
def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]  #Get the hue value
    # Handle red hue wrap-around
    if hue >= 165:  #Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 1:  #Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

# Initialize Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 dimensions (x, y, dx, dy), 2 observations (x, y)
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])  # State Transition Matrix

kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])  # Measurement Function

kf.P *= 10  # Covariance matrix
kf.R = np.array([[0.8, 0],
                 [0, 0.8]])  # Measurement Noise

yellow = [0, 255, 255]
vid = cv2.VideoCapture(0)
center=[]
predicted_center= []
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
i = 0
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    ret, frame = vid.read()
    if ret:
        frame=cv2.flip(frame,1)
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lowerLimit, upperLimit = get_limits(color=yellow)
        # creating a mas of the segmented object
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        mask_ = Image.fromarray(mask)     
        bbox = mask_.getbbox()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # coordinates of center of bounding box
            x = int((x2 + x1) / 2)
            y = int((y2 + y1) / 2)
            # creating a bounding box around the mask   
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            center.append((x,y))
            if i==0:
                cv2.line(frame,center[i],center[i],(0, 0, 255),2)
            else:
                cv2.line(frame,center[i],center[i-1],(0, 0, 255),2)

            # making predictions using the kalaman filter
            kf.predict()
            kf.update([x, y])
            predicted = kf.x.astype(int)
            predicted_center.append((int(predicted[0]), int(predicted[1])))

            if i==0:
                cv2.line(frame,predicted_center[i],predicted_center[i],(255, 0, 0),2)
            else:
                cv2.line(frame,predicted_center[i],predicted_center[i-1],(255, 0, 0),2)
            i+=1

        cv2.imshow('frame', frame)
        out.write(frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

vid.release()
out.release()

# creating a final output of the actual and predicted paths
final = np.zeros([frame_width,frame_height,3],dtype=np.uint8)
final.fill(255)
for i,itr in enumerate(center):
    if i==0:
        cv2.line(final,center[i],center[i],(0, 0, 255),4)
    else:
        cv2.line(final,center[i],center[i-1],(0, 0, 255),4)
for i, itr in enumerate(predicted_center):
    if i==0:
        cv2.line(final,predicted_center[i],predicted_center[i],(255, 0, 0),2)
    else:
        cv2.line(final,predicted_center[i],predicted_center[i-1],(255, 0, 0),2)
cv2.imwrite("path_ex2.jpg",final)
cv2.destroyAllWindows()
