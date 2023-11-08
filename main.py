import numpy as np
from matplotlib import pyplot as plt
import os

# plot gps and odom data
def plotData(mydata):
    x, y, _ = mydata.T
    plt.plot(x, y)

# plot the KF data 
def plotKFTRajectory(X):
    x, y = np.array(X).T
    plt.plot(x,y)

class Filter:

    '''
    Kalman filter class
    Reference: https://github.com/amolloma/Sensor_Fusion_Exercises/blob/main/lesson-3-EKF/exercises/solution/2_filter.py
    '''
    def __init__(self):
        self.dim_state = 4 # process model dimension
        self.dt = 1 # time increment
        self.q= 0.1 # process noise variable for Kalman filter Q

    def F(self):
        # system matrix
        dt = self.dt
        return np.matrix([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def Q(self):
        # process noise covariance Q
        q = self.q
        dt = self.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q 
        return np.matrix([[q1, 0, q2, 0],
                        [0, q1, 0, q2],
                        [q2, 0, q3, 0],
                        [0, q2, 0,  q3]])
        
    def H(self):
        # measurement matrix H
        return np.matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0]]) 
    
    def predict(self, x, P):
        # predict state and estimation error covariance to next timestep
        F = self.F()
        x = F*x # state prediction
        P = F*P*F.transpose() + self.Q() # covariance prediction
        return x, P

    def update(self, x, P, z, R):
        # update state and covariance with associated measurement
        H = self.H() # measurement matrix
        gamma = z - H*x # residual
        S = H*P*H.transpose() + R # covariance of residual
        K = P*H.transpose()*np.linalg.inv(S) # Kalman gain
        x = x + K*gamma # state update
        I = np.identity(self.dim_state)
        P = (I - K*H) * P # covariance update
        return x, P     
    

def run_filter(gpsdata, odomdata):
    # Initial data of x, y
    # Since I know that both the sensors reading starts with 0, 0 
    # I can choose 0,0 as my initial value 
    ekf_data = [[0, 0]] 

    # initial state (4x1)
    x = np.array([[0],[0],[0],[0]])

    # initial covariance
    P = 100000 * np.eye(4)

    # Q - process noise (4x4)
    Q_gps = np.eye(4)
    Q_odom = np.eye(4)

    # R - measurement noise (2x2)
    R_gps = 0.2 * np.eye(2)
    R_odom = 0.05 * np.eye(2)

    # filter instance
    filter = Filter()

    for kk in range(gpsdata.shape[0]):
        
        # GPS Measurement 
        gps_k = np.array([ [gpsdata[kk][0]], [gpsdata[kk][1]]  ])
        odom_k = np.array([ [odomdata[kk][0]], [odomdata[kk][1]]  ])
        
        # Measurement update
        x, P = filter.update(x, P, gps_k, R_gps)
        x, P = filter.update(x, P, odom_k, R_odom)

        ekf_data.append([np.array(x).T[0][0], np.array(x).T[0][1]])

        # predict to next time step
        x, P = filter.predict(x, P)
        
    return ekf_data

if __name__=='__main__':
    # Get the current path
    cwd = os.getcwd()

    file_name_gps = "gps_data.npy"
    file_name_odom = "odom_data.npy"
    
    gps_npy_file = cwd + "/odom and gps data/" + file_name_gps
    odom_npy_file = cwd + "/odom and gps data/" + file_name_odom
    
    gpsdata = np.load(gps_npy_file)
    gpsdata = np.array(gpsdata)

    odomdata = np.load(odom_npy_file)
    odomdata = np.array(odomdata)

    # Run the filter 
    ekf_data = run_filter(gpsdata, odomdata)
    
    # Plot data
    plotData(gpsdata)
    plotData(odomdata)
    plotKFTRajectory(ekf_data)

    plt.legend(["gps", "odom", "ekf"], loc ="lower left")
    plt.show()