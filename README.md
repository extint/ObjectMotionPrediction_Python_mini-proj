# ObjectMotionPrediction
## Description
Object detection using OpenCV and predicting trajectory of object using KalmanFilter Algorithm.
Currently it simply detects objects in respective hue range (e.g. yellow) and applies the motion prediction algorithm on it.

## Kalman Filter
The Kalman filter is a recursive method that estimates the next state of a system. It's used for dynamic systems whose parameters are time-dependent, like motion equations. 
The Kalman filter is a corrective predictor filter. It takes information on the state of an object at a precise moment, and then uses this information to predict where the object is in the next frame. 
[READ](https://www.sciencedirect.com/science/article/abs/pii/S0923596519302395#:~:text=Kalman%20filter%20is%20a%20recursive,prediction%20in%20a%20frame%20sequence.)

## Output
![](Assets/motionPrediction.gif)
