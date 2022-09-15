"""
MIT License

© Crown Copyright 2017-2022 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2022 Defence Research and Development Canada / Recherche et développement pour la défense Canada
© Copyright 2018-2022 University of Liverpool UK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Imports for plotting
from matplotlib import pyplot as plt
# Other general imports
import numpy as np
from datetime import datetime, timedelta
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial.transform import Rotation as R
import cv2

font_size = 20

deg2rad = 3.14/180.0
USE_CONST_ACC_MODEL = False

PIXELS_X, PIXELS_Y = 100, 100
x_min, x_max, y_min, y_max = -100, 100, -100, 100

number_steps = 240
death_probability = 1e-4
birth_probability = 1e-4
probability_detection = 0.9
merge_threshold = 5
prune_threshold = 1E-8
state_threshold = 0.9
gaussian_plot_threshold=0.1

# target_sn = 0.2

HFOV = 90*deg2rad # degrees
VFOV = 90*deg2rad # degrees

# FOVsize = 30.0*np.sqrt(2)
# clutter_rate = 0.05*(1.0/100.0)*(FOVsize/np.sqrt(2))**2
# print(clutter_rate)

clutter_rate = 0.45

to_img_bias = [ 0.5*PIXELS_X, 0.5*PIXELS_Y ]
to_img_scale = [ PIXELS_X/(x_max-x_min) , PIXELS_Y/(y_max-y_min) ]
NEG_MEAS_VALUE = -0.5*9
# p_s = 0.5; # probability of staying at the same location
SS_Like = 0.01 # Steady State Likelihood Value
Decay = 0.01; # decay rate
Absent_Like = 1.0/(1.0+SS_Like); # probability of target not present in arena
Birth_Like = Decay*(1.0-Absent_Like);
Death_Like = Decay*Absent_Like;
blur_kernel = (1.0-Death_Like)*cv2.getGaussianKernel(3,0)
Like_Ratio = np.ones((PIXELS_X, PIXELS_Y)) * SS_Like
Like_Ratio_by_time = []
Search_Prob_by_time = []

start_time = datetime.now()
truths_by_time = []

entropy_by_time = np.zeros((PIXELS_X, PIXELS_Y, number_steps))

gaussian_guidance_threshold = 7.0

magnitude_threshold = 0.1
X_GRAD_IMG = np.zeros((2*PIXELS_X, 2*PIXELS_Y))
Y_GRAD_IMG = np.zeros((2*PIXELS_X, 2*PIXELS_Y))
for i in range(2*PIXELS_Y):
    for j in range(2*PIXELS_X):
        x = (j-PIXELS_X)/to_img_scale[0]
        y = (i-PIXELS_Y)/to_img_scale[1]
        magnitude = 100*(1.0/(x**2+y**2+0.01)**1.5)
        if (magnitude>magnitude_threshold):
            magnitude = 0.0
            # magnitude = 1.0
        X_GRAD_IMG[i,j] = magnitude*x
        Y_GRAD_IMG[i,j] = magnitude*y
    
cv2.imshow('title',X_GRAD_IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()

ray1 = np.array([np.tan(0.5*VFOV), -np.tan(0.5*HFOV),-1])
ray2 = np.array([np.tan(0.5*VFOV), np.tan(0.5*HFOV),-1])
ray3 = np.array([-np.tan(0.5*VFOV), np.tan(0.5*HFOV),-1])
ray4 = np.array([-np.tan(0.5*VFOV), -np.tan(0.5*HFOV),-1])
plane_normal = np.array([0,0,1])

def getFOVCorners(UAV_pos,orient):
    # corner1 = (UAV_pos[0]+size*np.cos(orient+45*deg2rad),UAV_pos[1]+size*np.sin(orient+45*deg2rad))
    # corner2 = (UAV_pos[0]+size*np.cos(orient+135*deg2rad),UAV_pos[1]+size*np.sin(orient+135*deg2rad))
    # corner3 = (UAV_pos[0]+size*np.cos(orient+225*deg2rad),UAV_pos[1]+size*np.sin(orient+225*deg2rad))
    # corner4 = (UAV_pos[0]+size*np.cos(orient+315*deg2rad),UAV_pos[1]+size*np.sin(orient+315*deg2rad))

    ray1_rot = orient.apply(ray1)
    ray2_rot = orient.apply(ray2)
    ray3_rot = orient.apply(ray3)
    ray4_rot = orient.apply(ray4)

    ray_orig = UAV_pos
    plane_orig = np.array([UAV_pos[0],UAV_pos[1],0])

    numerator = np.dot((plane_orig-ray_orig),plane_normal)
    t1 = numerator/np.dot(ray1_rot,plane_normal)
    t2 = numerator/np.dot(ray2_rot,plane_normal)
    t3 = numerator/np.dot(ray3_rot,plane_normal)
    t4 = numerator/np.dot(ray4_rot,plane_normal)

    corner1 = ray_orig+ray1_rot*t1
    corner2 = ray_orig+ray2_rot*t2
    corner3 = ray_orig+ray3_rot*t3
    corner4 = ray_orig+ray4_rot*t4

    corner1 = (corner1[0],corner1[1])
    corner2 = (corner2[0],corner2[1])
    corner3 = (corner3[0],corner3[1])
    corner4 = (corner4[0],corner4[1])

    return np.array([corner1,corner2,corner3,corner4])

def getFOVPolygon(UAV_pos,orient):
    return Polygon(getFOVCorners(UAV_pos,orient))

if (USE_CONST_ACC_MODEL):
    def inFOV(target_state,UAV_state,UAV_orient):
        point = Point(target_state[0], target_state[3])
        polygon = getFOVPolygon(UAV_state,UAV_orient)
        return polygon.contains(point)
else:
    def inFOV(target_state,UAV_state,UAV_orient):
        point = Point(target_state[0], target_state[2])
        polygon = getFOVPolygon(UAV_state,UAV_orient)
        return polygon.contains(point)    

def get_Rotation_from_acc(acc,yaw):
    xc = np.array([[np.cos(yaw), np.sin(yaw), 0]])
    zb = acc/np.linalg.norm(acc)
    yb = np.cross(zb,xc)
    yb = yb/np.linalg.norm(yb)
    xb = np.cross(yb,zb)
    return R.from_matrix(np.concatenate((xb.T,yb.T,zb.T),axis=1))

## Generate UAV positions and orientations
UAV1 = []
UAV2 = []
UAV3 = []
UAV1_Rot = []
UAV2_Rot = []
UAV3_Rot = []
# for i in range(number_steps):
#     UAV1.append([-30+30*np.cos(i/10),   -30+10*np.sin(-i/10),    20])
#     UAV2.append([0+50*np.cos(i/5),     0+50*np.sin(i/5),      30])
#     UAV3.append([30+10*np.cos(i/10),   30+30*np.sin(i/10),   20])

#     ## Simulate UAV orientation
#     acc1 = np.array([[-30*(1/10)*(1/10)*np.cos(i/10) , -10*(-1/10)*(-1/10)*np.sin(-i/10) ,  9.81]])
#     acc2 = np.array([[-50*(1/5)*(1/5)*np.cos(i/5) , -50*(1/5)*(1/5)*np.sin(i/5) ,     9.81]])
#     acc3 = np.array([[-10*(1/10)*(1/10)*np.cos(i/10) , -30*(1/10)*(1/10)*np.sin(i/10) ,  9.81]])
#     yaw1 = -i/10 + 3.14/2
#     yaw2 = i/5 + 3.14/2
#     yaw3 = i/10 + 3.14/2
#     UAV1_Rot.append(get_Rotation_from_acc(acc1,yaw1))
#     UAV2_Rot.append(get_Rotation_from_acc(acc2,yaw2))
#     UAV3_Rot.append(get_Rotation_from_acc(acc3,yaw3))

# UAV1 = np.array(UAV1)
# UAV2 = np.array(UAV2)
# UAV3 = np.array(UAV3)

## Generate FOVPolygons for plotting
UAV1_Polygon = []
UAV2_Polygon = []
UAV3_Polygon = []
# for i in range(number_steps):
#     UAV1_Polygon.append(getFOVPolygon(UAV1[i],UAV1_Rot[i]))
#     UAV2_Polygon.append(getFOVPolygon(UAV2[i],UAV2_Rot[i]))
#     UAV3_Polygon.append(getFOVPolygon(UAV3[i],UAV3_Rot[i]))

# Create transition model
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, ConstantAcceleration
if (USE_CONST_ACC_MODEL):
    transition_model = CombinedLinearGaussianTransitionModel(
        (ConstantAcceleration(0.1), ConstantAcceleration(0.1)))
else:
    transition_model = CombinedLinearGaussianTransitionModel(
        (ConstantVelocity(0.1), ConstantVelocity(0.1)))

# Make the measurement model
from stonesoup.models.measurement.linear import LinearGaussian
if (USE_CONST_ACC_MODEL):
    measurement_model = LinearGaussian(
        ndim_state=6,
        mapping=(0, 3),
        noise_covar=np.array([[0.75, 0],
                              [0, 0.75]])
        )
else:
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0],
                              [0, 0.75]])
        )

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
start_time = datetime.now()
truths = set()  # Truths across all time
current_truths = set()  # Truths alive at current time
start_truths = set()

# Initialize 3 truths. This can be changed to any number of truths you wish.
truths_by_time.append([])
for i in range(3):
    # x, y = initial_position = np.random.uniform(-30, 30, 2)  # Range [-30, 30] for x and y
    # x_vel, y_vel = (np.random.rand(2))*2 - 1  # Range [-1, 1] for x and y velocity
    # state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time)
    
    x, y = initial_position = [i*30-30,i*30-30]  # Range [-30, 30] for x and y
    x_vel, y_vel = [0,0] # Range [-1, 1] for x and y velocity
    x_acc, y_acc = [0,0] # Range [-1, 1] for x and y acceleration

    if (USE_CONST_ACC_MODEL):
        state = GroundTruthState([x, x_vel, x_acc, y, y_vel, y_acc], timestamp=start_time)
    else:
        state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time)

    truth = GroundTruthPath([state])
    current_truths.add(truth)
    truths.add(truth)
    start_truths.add(truth)
    truths_by_time[0].append(state)

# Simulate the ground truth over time
for k in range(number_steps):
    truths_by_time.append([])
    # # Death
    # for truth in current_truths.copy():
    #     if np.random.rand() <= death_probability:
    #         current_truths.remove(truth)
    # Update truths
    iter = 1
    for truth in current_truths:
        # updated_state = GroundTruthState(
        #     transition_model.function(truth[-1], noise=True, time_interval=timedelta(seconds=1)),
        #     timestamp=start_time + timedelta(seconds=k))
        x_acc = 0.0
        y_acc = 0.0
        x_vel = -iter*1*np.sign(np.sin(k/10))
        y_vel = iter*1*np.sign(np.cos(k/10))

        if (iter==3):
            x_vel, y_vel = 0, 0

        if (USE_CONST_ACC_MODEL):
            x = truth.state_vector[0] + x_vel # dt is 1 second
            y = truth.state_vector[3] + y_vel
            updated_state = GroundTruthState([x, x_vel, x_acc, y, y_vel, y_acc], timestamp=start_time + timedelta(seconds=k))
        else:
            x = truth.state_vector[0] + x_vel # dt is 1 second
            y = truth.state_vector[2] + y_vel
            updated_state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time + timedelta(seconds=k))
        truth.append(updated_state)
        truths_by_time[k].append(updated_state)
        iter = iter+1
    # # Birth
    # for _ in range(np.random.poisson(birth_probability)):
    #     x, y = initial_position = np.random.rand(2) * [120, 120]  # Range [0, 20] for x and y
    #     x_vel, y_vel = (np.random.rand(2))*2 - 1  # Range [-1, 1] for x and y velocity
    #     state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time + timedelta(seconds=k))

    #     # Add to truth set for current and for all timestamps
    #     truth = GroundTruthPath([state])
    #     current_truths.add(truth)
    #     truths.add(truth)
    #     truths_by_time[k].append(state)


# Generate detections and clutter

from scipy.stats import uniform

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter

all_measurements = []

# for k in range(number_steps):
#     measurement_set = set()
#     timestamp = start_time + timedelta(seconds=k)

#     for truth in truths:
#         # try:
#         #     truth_state = truth[timestamp]
#         # except IndexError:
#         #     # This truth not alive at this time. Skip this iteration of the for loop.
#         #     continue

#         truth_state = truth[timestamp]

#         if (inFOV(truth_state.state_vector,UAV1[k],UAV1_Rot[k])
#             or inFOV(truth_state.state_vector,UAV2[k],UAV2_Rot[k])
#             or inFOV(truth_state.state_vector,UAV3[k],UAV3_Rot[k])):

#             # Generate actual detection from the state with a 10% chance that no detection is received.
#             if np.random.rand() <= probability_detection:
#                 # Generate actual detection from the state
#                 measurement = measurement_model.function(truth_state, noise=True)
#                 measurement_set.add(TrueDetection(state_vector=measurement,
#                                                   groundtruth_path=truth,
#                                                   timestamp=truth_state.timestamp,
#                                                   measurement_model=measurement_model))

#     # Generate clutter at this time-step
#     for _ in range(np.random.poisson(clutter_rate)):
#             while True:
#                 x = uniform.rvs(-200, 400)
#                 y = uniform.rvs(-200, 400)
#                 if (USE_CONST_ACC_MODEL):
#                     clutter_state = [x,0,0,y,0,0]
#                 else:
#                     clutter_state = [x,0,y,0]
#                 if(inFOV(clutter_state,UAV1[k],UAV1_Rot[k]) 
#                     or inFOV(clutter_state,UAV2[k],UAV2_Rot[k]) 
#                     or inFOV(clutter_state,UAV3[k],UAV3_Rot[k])):
#                     measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=timestamp,
#                                         measurement_model=measurement_model))
#                     break

#     all_measurements.append(measurement_set)

time0 = time.time()

from stonesoup.updater.kalman import KalmanUpdater
kalman_updater = KalmanUpdater(measurement_model, force_symmetric_covariance=True)

# Area in which we look for target. Note that if a target appears outside of this area the
# filter will not pick up on it.
meas_range = np.array([[-1, 1], [-1, 1]])*100
clutter_spatial_density = clutter_rate/np.prod(np.diff(meas_range))

from stonesoup.updater.pointprocess import PHDUpdater
updater = PHDUpdater(
    kalman_updater,
    clutter_spatial_density=clutter_spatial_density,
    prob_detection=0.1*probability_detection,
    prob_survival=1.0-death_probability)

from stonesoup.predictor.kalman import KalmanPredictor
kalman_predictor = KalmanPredictor(transition_model)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
base_hypothesiser = DistanceHypothesiser(kalman_predictor, kalman_updater, Mahalanobis(), missed_distance=30)

from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
hypothesiser = GaussianMixtureHypothesiser(base_hypothesiser, order_by_detection=True)

from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
# Initialise a Gaussian Mixture reducer

reducer = GaussianMixtureReducer(
    prune_threshold=prune_threshold,
    pruning=True,
    merge_threshold=merge_threshold,
    merging=True)

from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import CovarianceMatrix
if (USE_CONST_ACC_MODEL):
    covar = CovarianceMatrix(np.diag([10, 5, 1, 10, 5, 1]))
else:
    covar = CovarianceMatrix(np.diag([10, 5, 10, 5]))

tracks = set()
for truth in start_truths:
    new_track = TaggedWeightedGaussianState(
            state_vector=truth.state_vector,
            covar=covar**2,
            weight=0.25,
            tag=TaggedWeightedGaussianState.BIRTH,
            timestamp=start_time)
    tracks.add(Track(new_track))

reduced_states = set([track[-1] for track in tracks])

if (USE_CONST_ACC_MODEL):
    birth_covar = CovarianceMatrix(np.diag([1000, 2, 1, 1000, 2, 1]))
    birth_component = TaggedWeightedGaussianState(
        state_vector=[0, 0, 0, 0, 0, 0],
        covar=birth_covar**2,
        weight=0.25,
        tag='birth',
        timestamp=start_time
    )
else:
    birth_covar = CovarianceMatrix(np.diag([1000, 2, 1000, 2]))
    birth_component = TaggedWeightedGaussianState(
        state_vector=[0, 0, 0, 0],
        covar=birth_covar**2,
        weight=0.25,
        tag='birth',
        timestamp=start_time
    )

all_gaussians = []
tracks_by_time = []

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import vorbin
# from vorbin.voronoi_2d_binning import voronoi_2d_binning

def get_entropy(p_array):
    # print(p_array)
    entropy = -p_array*np.log2(p_array+1e-10) - (1-p_array)*np.log2(1-p_array+1e-10)
    # print(entropy)
    return entropy

def get_mixture_density(x, y, weights, means, sigmas):
    # We use the quantiles as a parameter in the multivariate_normal function. We don't need to pass in any quantiles,
    # but the last axis must have the components x and y
    quantiles = np.empty(x.shape + (2,))  # if  x.shape is (m,n) then quantiles.shape is (m,n,2)
    quantiles[:, :, 0] = x
    quantiles[:, :, 1] = y

    # Go through each gaussian in the list and add its PDF to the mixture
    z = np.zeros(x.shape)
    for gaussian in range(len(weights)):
        z += weights[gaussian]*multivariate_normal.pdf(x=quantiles, mean=means[gaussian, :], cov=sigmas[gaussian])
    return z

def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def get_force_from_entropy(pos, field, mask):
    # field = field
    # mask = 1-mask
    x_img = X_GRAD_IMG[PIXELS_Y-pos[1] : 2*PIXELS_Y-pos[1], PIXELS_X-pos[0] : 2*PIXELS_X-pos[0]]
    y_img = Y_GRAD_IMG[PIXELS_Y-pos[1] : 2*PIXELS_Y-pos[1], PIXELS_X-pos[0] : 2*PIXELS_X-pos[0]]
    # x_img = mask * x_img
    # y_img = mask * y_img
    fx_img = cv2.multiply(x_img,field)
    fy_img = cv2.multiply(y_img,field)
    fx = np.sum(fx_img)
    fy = np.sum(fy_img)
    # print(fx)
    # cv2.imshow('title',x_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    return np.array([fx,fy,0])

print("--- Took %s seconds to initilize ---" % (time.time() - time0))
time0 = datetime.now()

xx = np.linspace(x_min, x_max, PIXELS_X)
yy = np.linspace(y_min, y_max, PIXELS_Y)
x, y = np.meshgrid(xx, yy)

tasks_x_by_time=[]
tasks_y_by_time=[]
tasks_by_time=[]

UAV1_pos = np.array([10.0, 10.0, 20.0])
UAV2_pos = np.array([-10.0, 10.0, 20.0])
UAV3_pos = np.array([0.0, -10.0, 20.0])
UAV1_vel_lp = np.array([0.0, 0.0, 0.0])
UAV2_vel_lp = np.array([0.0, 0.0, 0.0])
UAV3_vel_lp = np.array([0.0, 0.0, 0.0])
# UAV1.append(UAV1_pos)
# UAV2.append(UAV2_pos)
# UAV3.append(UAV3_pos)
v_min , v_max = -5, 5
a_min , a_max = -5, 5
control_gain = 10.0
lp_gain = 0.0

for n in range(number_steps):

    UAV1.append(UAV1_pos)
    UAV2.append(UAV2_pos)
    UAV3.append(UAV3_pos)

    ## Simulate UAV orientation
    acc1 = np.array([[0, 0, 9.81]])
    acc2 = np.array([[0, 0, 9.81]])
    acc3 = np.array([[0, 0, 9.81]])
    yaw1 = 0 
    yaw2 = 0
    yaw3 = 0
    UAV1_Rot.append(get_Rotation_from_acc(acc1,yaw1))
    UAV2_Rot.append(get_Rotation_from_acc(acc2,yaw2))
    UAV3_Rot.append(get_Rotation_from_acc(acc3,yaw3))

    measurement_set = set()
    timestamp = start_time + timedelta(seconds=n)

    for truth in truths:
        # try:
        #     truth_state = truth[timestamp]
        # except IndexError:
        #     # This truth not alive at this time. Skip this iteration of the for loop.
        #     continue

        truth_state = truth[timestamp]

        # print(UAV1[n])

        if (inFOV(truth_state.state_vector,UAV1[n],UAV1_Rot[n])
            or inFOV(truth_state.state_vector,UAV2[n],UAV2_Rot[n])
            or inFOV(truth_state.state_vector,UAV3[n],UAV3_Rot[n])):

            # Generate actual detection from the state with a 10% chance that no detection is received.
            if np.random.rand() <= probability_detection:
                # Generate actual detection from the state
                measurement = measurement_model.function(truth_state, noise=True)
                measurement_set.add(TrueDetection(state_vector=measurement,
                                                  groundtruth_path=truth,
                                                  timestamp=truth_state.timestamp,
                                                  measurement_model=measurement_model))

    # Generate clutter at this time-step
    for _ in range(np.random.poisson(clutter_rate)):
            while True:
                x_clutter = uniform.rvs(-200, 400)
                y_clutter = uniform.rvs(-200, 400)
                if (USE_CONST_ACC_MODEL):
                    clutter_state = [x_clutter,0,0,y_clutter,0,0]
                else:
                    clutter_state = [x_clutter,0,y_clutter,0]
                if(inFOV(clutter_state,UAV1[n],UAV1_Rot[n]) 
                    or inFOV(clutter_state,UAV2[n],UAV2_Rot[n]) 
                    or inFOV(clutter_state,UAV3[n],UAV3_Rot[n])):
                    measurement_set.add(Clutter(np.array([[x_clutter], [y_clutter]]), timestamp=timestamp,
                                        measurement_model=measurement_model))
                    break

    all_measurements.append(measurement_set)
    measurements = measurement_set

    mask1 = np.zeros((PIXELS_Y,PIXELS_X))
    mask2 = np.zeros((PIXELS_Y,PIXELS_X))
    mask3 = np.zeros((PIXELS_Y,PIXELS_X))
    UAV1_FOV = getFOVCorners(UAV1[n],UAV1_Rot[n]) * to_img_scale + to_img_bias
    UAV2_FOV = getFOVCorners(UAV2[n],UAV2_Rot[n]) * to_img_scale + to_img_bias
    UAV3_FOV = getFOVCorners(UAV3[n],UAV3_Rot[n]) * to_img_scale + to_img_bias
    cv2.fillPoly(mask1, np.int32([UAV1_FOV]), color=1)
    cv2.fillPoly(mask2, np.int32([UAV2_FOV]), color=1)
    cv2.fillPoly(mask3, np.int32([UAV3_FOV]), color=1)
    # mask1 = mask1.astype(bool)
    # mask2 = mask2.astype(bool)
    # mask3 = mask3.astype(bool)
    mask = mask1
    mask = cv2.bitwise_or(mask, mask2)
    mask = cv2.bitwise_or(mask, mask3)
    # cv2.imshow('title',mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    Like_Ratio_Blurred = cv2.filter2D(Like_Ratio,-1,blur_kernel)
    Like_Ratio_Propagated = Birth_Like + Like_Ratio_Blurred
    Log_Like_Ratio_Propagated = np.log2(Like_Ratio_Propagated)

    Log_Like_Ratio_Update_Value = NEG_MEAS_VALUE * mask
    Log_Like_Ratio = Log_Like_Ratio_Propagated + Log_Like_Ratio_Update_Value

    # Like_Ratio_Quant = 1L << Log_Like_Ratio_Quant
    Like_Ratio = np.power(2,Log_Like_Ratio)
    Like_Ratio_by_time.append(Like_Ratio)
    Search_Prob_by_time.append(Like_Ratio / ( 1 + Like_Ratio))

    # print(Like_Ratio)
    # cv2.imshow('title',1000*Like_Ratio)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    tracks_by_time.append([])
    all_gaussians.append([])

    # The hypothesiser takes in the current state of the Gaussian mixture. This is equal to the list of
    # reduced states from the previous iteration. If this is the first iteration, then we use the priors
    # defined above.
    current_state = reduced_states

    # At every time step we must add the birth component to the current state
    if measurements:
        time = list(measurements)[0].timestamp
    else:
        time = start_time + timedelta(seconds=n)
    birth_component.timestamp = time
    current_state.add(birth_component)

    # Generate the set of hypotheses
    hypotheses = hypothesiser.hypothesise(current_state,
                                          measurements,
                                          timestamp=time,
                                          # keep our hypotheses ordered by detection, not by track
                                          order_by_detection=True)

    # Turn the hypotheses into a GaussianMixture object holding a list of states
    updated_states = updater.update(hypotheses)

    # Prune and merge the updated states into a list of reduced states
    reduced_states = set(reducer.reduce(updated_states))

    # Add the reduced states to the track list. Each reduced state has a unique tag. If this tag matches the tag of a
    # state from a live track, we add the state to that track. Otherwise, we generate a new track if the reduced
    # state's weight is high enough (i.e. we are sufficiently certain that it is a new track).
    for reduced_state in reduced_states:
        # Add the reduced state to the list of Gaussians that we will plot later. Have a low threshold to eliminate some
        # clutter that would make the graph busy and hard to understand
        if reduced_state.weight > gaussian_plot_threshold: all_gaussians[n].append(reduced_state)
        # all_gaussians[n].append(reduced_state)

        tag = reduced_state.tag
        # Here we check to see if the state has a sufficiently high weight to consider being added.
        if reduced_state.weight > state_threshold:
            # Check if the reduced state belongs to a live track
            for track in tracks:
                track_tags = [state.tag for state in track.states]

                if tag in track_tags:
                    track.append(reduced_state)
                    tracks_by_time[n].append(reduced_state)
                    break
            else:  # Execute if no "break" is hit; i.e. no track with matching tag
                # Make a new track out of the reduced state
                new_track = Track(reduced_state)
                tracks.add(new_track)
                tracks_by_time[n].append(reduced_state)

    # Initialize the variables
    weights = []  # weights of each Gaussian. This is analogous to the probability of its existence
    means = []    # means of each Gaussian. This is equal to the x and y of its state vector
    sigmas = []   # standard deviation of each Gaussian.

    # Fill the lists of weights, means, and standard deviations
    if (USE_CONST_ACC_MODEL):
        for state in all_gaussians[n]:
            weights.append(state.weight)
            means.append([state.state_vector[0], state.state_vector[3]])
            sigmas.append([state.covar[0][0], state.covar[3][3]])
    else:
        for state in all_gaussians[n]:
            weights.append(state.weight)
            means.append([state.state_vector[0], state.state_vector[2]])
            sigmas.append([state.covar[0][0], state.covar[2][2]])
    
    means = np.array(means)
    sigmas = np.array(sigmas)
    search_entropy = get_entropy(Search_Prob_by_time[n])
    track_entropy = get_entropy(get_mixture_density(x, y, weights, means, sigmas))
    total_entropy = search_entropy + track_entropy

    means_guidance = []
    sigmas_guidance = []
    weights_guidance = []

    for idx, mean in enumerate(means):
        # print(sigmas[idx][0])
        trace = sigmas[idx][0] + sigmas[idx][1]
        if (trace > gaussian_guidance_threshold):
            means_guidance.append(mean)
            sigmas_guidance.append(sigmas[idx])
            weights_guidance.append(weights[idx])

    means_guidance = np.array(means_guidance)
    sigmas_guidance = np.array(sigmas_guidance)
    track_guidance_entropy = get_entropy(get_mixture_density(x, y, weights_guidance, means_guidance, sigmas_guidance))
    total_guidance_entropy = search_entropy + track_guidance_entropy

    entropy_by_time[:, :, n] = total_guidance_entropy
    # entropy_by_time[:, :, n] = total_entropy

    # signal = entropy_by_time[:, :, n]
    # noise = np.ones((PIXELS_X, PIXELS_Y))
    # binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
        # x.flatten(), y.flatten(), signal.flatten(), noise.flatten(), target_sn, plot=0, quiet=1)
    # bin_img = binNum.reshape(PIXELS_X,PIXELS_Y)
    # bin_img = 1+255*(bin_img/np.amax(bin_img))
    # print(bin_img)
    # cv2.imshow('title',bin_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    # tasks_x_by_time.append(x_bar)
    # tasks_y_by_time.append(y_bar)
    # tasks_by_time.append([x_gen,y_gen])

    # plt.scatter(x_gen, y_gen)
    # plt.tight_layout()
    # plt.pause(0.1)
    # plt.clf()

    # print(UAV1_pos[0:2])
    UAV1_pos_px = np.int32(UAV1_pos[0:2] * to_img_scale + to_img_bias)
    UAV2_pos_px = np.int32(UAV2_pos[0:2] * to_img_scale + to_img_bias)
    UAV3_pos_px = np.int32(UAV3_pos[0:2] * to_img_scale + to_img_bias)

    # if (np.random.random()<0.7):
    #     UAV1_vel = control_gain*get_force_from_entropy(UAV1_pos_px, search_entropy)
    # else:
    #     UAV1_vel = control_gain*get_force_from_entropy(UAV1_pos_px, total_entropy)

    # if (np.random.random()<0.7):
    #     UAV2_vel = control_gain*get_force_from_entropy(UAV2_pos_px, search_entropy)
    # else:
    #     UAV2_vel = control_gain*get_force_from_entropy(UAV2_pos_px, total_entropy)

    # if (np.random.random()<0.7):
    #     UAV3_vel = control_gain*get_force_from_entropy(UAV3_pos_px, search_entropy)
    # else:
    #     UAV3_vel = control_gain*get_force_from_entropy(UAV3_pos_px, total_entropy)

    UAV1_vel = control_gain*get_force_from_entropy(UAV1_pos_px, total_guidance_entropy, ~mask1.astype(bool))
    UAV2_vel = control_gain*get_force_from_entropy(UAV2_pos_px, total_guidance_entropy, ~mask2.astype(bool))
    UAV3_vel = control_gain*get_force_from_entropy(UAV3_pos_px, total_guidance_entropy, ~mask3.astype(bool))

    # UAV1_acc[0] = constrain(UAV1_acc[0], a_min, a_max)
    # UAV1_acc[1] = constrain(UAV1_acc[1], a_min, a_max)
    # UAV2_acc[0] = constrain(UAV2_acc[0], a_min, a_max)
    # UAV2_acc[1] = constrain(UAV2_acc[1], a_min, a_max)
    # UAV3_acc[0] = constrain(UAV3_acc[0], a_min, a_max)
    # UAV3_acc[1] = constrain(UAV3_acc[1], a_min, a_max)

    # UAV1_vel = UAV1_vel + UAV1_acc
    # UAV2_vel = UAV2_vel + UAV2_acc
    # UAV3_vel = UAV3_vel + UAV3_acc

    UAV1_vel[0] = constrain(UAV1_vel[0], v_min, v_max)
    UAV1_vel[1] = constrain(UAV1_vel[1], v_min, v_max)
    UAV2_vel[0] = constrain(UAV2_vel[0], v_min, v_max)
    UAV2_vel[1] = constrain(UAV2_vel[1], v_min, v_max)
    UAV3_vel[0] = constrain(UAV3_vel[0], v_min, v_max)
    UAV3_vel[1] = constrain(UAV3_vel[1], v_min, v_max)

    UAV1_vel_lp = lp_gain*UAV1_vel_lp + (1-lp_gain)*UAV1_vel
    UAV2_vel_lp = lp_gain*UAV2_vel_lp + (1-lp_gain)*UAV2_vel
    UAV3_vel_lp = lp_gain*UAV3_vel_lp + (1-lp_gain)*UAV3_vel

    UAV1_pos = UAV1_pos + UAV1_vel_lp
    UAV2_pos = UAV2_pos + UAV2_vel_lp
    UAV3_pos = UAV3_pos + UAV3_vel_lp

    UAV1_pos[0] = constrain(UAV1_pos[0], x_min, x_max)
    UAV1_pos[1] = constrain(UAV1_pos[1], x_min, x_max)
    UAV2_pos[0] = constrain(UAV2_pos[0], x_min, x_max)
    UAV2_pos[1] = constrain(UAV2_pos[1], x_min, x_max)
    UAV3_pos[0] = constrain(UAV3_pos[0], x_min, x_max)
    UAV3_pos[1] = constrain(UAV3_pos[1], x_min, x_max)

    UAV1_Polygon.append(getFOVPolygon(UAV1[n],UAV1_Rot[n]))
    UAV2_Polygon.append(getFOVPolygon(UAV2[n],UAV2_Rot[n]))
    UAV3_Polygon.append(getFOVPolygon(UAV3[n],UAV3_Rot[n]))

# print(tasks_by_time)

# print(UAV1)
UAV1 = np.array(UAV1)
UAV2 = np.array(UAV2)
UAV3 = np.array(UAV3)
# print(UAV1)

time_elaspsed = datetime.now()-time0
time_per_iteration = (time_elaspsed.seconds+1e-6*time_elaspsed.microseconds)/float(number_steps)
print("--- Took %s seconds on average per iteration ---" % time_per_iteration)
print("--- Estimated Rate %s Hz" % float(1/time_per_iteration))

# # Get bounds from the tracks
# for track in tracks:
#     for state in track:
#         x_min = min([state.state_vector[0], x_min])
#         x_max = max([state.state_vector[0], x_max])
#         y_min = min([state.state_vector[2], y_min])
#         y_max = max([state.state_vector[2], y_max])

# # Get bounds from measurements
# for measurement_set in all_measurements:
#     for measurement in measurement_set:
#         if type(measurement) == TrueDetection:
#             x_min = min([measurement.state_vector[0], x_min])
#             x_max = max([measurement.state_vector[0], x_max])
#             y_min = min([measurement.state_vector[1], y_min])
#             y_max = max([measurement.state_vector[1], y_max])

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D  # Will be used when making the legend

# This is the function that updates the figure we will be animating. As parameters we must
# pass in the elements that will be changed, as well as the index i
def animate(i, img_plot, truths, tracks, measurements, clutter):
    # Set up the axes
    # axL.clear()
    axR.set_title('Tracking Space at k='+str(i), fontsize=font_size)
    # axL.set_xlabel("x")
    # axL.set_ylabel("y")
    # axL.set_title('PDF of the Gaussian Mixture')
    # axL.view_init(elev=30, azim=-80)
    # axL.set_zlim(0, 0.3)

    # # Initialize the variables
    # weights = []  # weights of each Gaussian. This is analogous to the probability of its existence
    # means = []    # means of each Gaussian. This is equal to the x and y of its state vector
    # sigmas = []   # standard deviation of each Gaussian.

    # # Fill the lists of weights, means, and standard deviations
    # if (USE_CONST_ACC_MODEL):
    #     for state in all_gaussians[i]:
    #         weights.append(state.weight)
    #         means.append([state.state_vector[0], state.state_vector[3]])
    #         sigmas.append([state.covar[0][0], state.covar[3][3]])
    # else:
    #     for state in all_gaussians[i]:
    #         weights.append(state.weight)
    #         means.append([state.state_vector[0], state.state_vector[2]])
    #         sigmas.append([state.covar[0][0], state.covar[2][2]])
    
    # means = np.array(means)
    # sigmas = np.array(sigmas)

    # Generate the z values over the space and plot on the left axis
    # zarray[:, :, i] = get_mixture_density(x, y, weights, means, sigmas)
    # zarray[:, :, i] = get_entropy(get_mixture_density(x, y, weights, means, sigmas))
    # zarray[:, :, i] += get_entropy(Search_Prob_by_time[i])

    # sf = axL.plot_surface(x, y, zarray[:, :, i], cmap=cm.RdBu, linewidth=0, antialiased=False)
    img_plot = axR.imshow(cv2.flip(entropy_by_time[:, :, i],0),
                            extent=[x_min,x_max,y_min,y_max],
                            norm=cm.colors.Normalize(vmin=0, vmax=0.1))


    # Make lists to hold the new ground truths, tracks, detections, and clutter
    new_truths, new_tracks, new_measurements, new_clutter, new_tasks = [], [], [], [], []

    if (USE_CONST_ACC_MODEL):
        for truth in truths_by_time[i]:
            new_truths.append([truth.state_vector[0], truth.state_vector[3]])
        for state in tracks_by_time[i]:
            new_tracks.append([state.state_vector[0], state.state_vector[3]])
    else:
        for truth in truths_by_time[i]:
            new_truths.append([truth.state_vector[0], truth.state_vector[2]])
        for state in tracks_by_time[i]:
            new_tracks.append([state.state_vector[0], state.state_vector[2]])

    for measurement in all_measurements[i]:
        if isinstance(measurement, TrueDetection):
            new_measurements.append([measurement.state_vector[0], measurement.state_vector[1]])
        elif isinstance(measurement, Clutter):
            new_clutter.append([measurement.state_vector[0], measurement.state_vector[1]])

    # Plot the contents of these lists on the right axis
    if new_truths:
        truths.set_offsets(new_truths)
    if new_tracks:
        tracks.set_offsets(new_tracks)
    if new_measurements:
        measurements.set_offsets(new_measurements)
    if new_clutter:
        clutter.set_offsets(new_clutter)

    UAV1_p.set_offsets([UAV1[i,0],UAV1[i,1]])
    UAV2_p.set_offsets([UAV2[i,0],UAV2[i,1]])
    UAV3_p.set_offsets([UAV3[i,0],UAV3[i,1]])

    UAV1_FOV_Plot.set_data(*UAV1_Polygon[i].exterior.xy)
    UAV2_FOV_Plot.set_data(*UAV2_Polygon[i].exterior.xy)
    UAV3_FOV_Plot.set_data(*UAV3_Polygon[i].exterior.xy)

    # for task in tasks_by_time[i]:
    #     print(task)
    #     new_tasks.append([task[0], task[1]])
    # if new_tasks:
    #     tasks_p.set_offsets(new_tasks)

    # print("printing tasks")
    # tasks_array = np.array([tasks_x_by_time[i],tasks_y_by_time[i]])
    # tasks_p.set_offsets(tasks_array.T)

    # Create a legend. The use of Line2D is purely for the visual in the legend
    # data_types = [Line2D([0], [0], color='blue', marker='+', markerfacecolor='blue', markersize=15,
    #                      label='Ground Truth'),
    #              Line2D([0], [0], color='green', marker='1', markerfacecolor='green', markersize=15,
    #                      label='Clutter'),
    #              Line2D([0], [0], color='orange', marker='1', markerfacecolor='orange', markersize=15,
    #                      label='Detection'),
    #              Line2D([0], [0], color='red', marker='x', markerfacecolor='red', markersize=15,
    #                      label='Track'),
    #              Line2D([0], [0], color='white', marker='*', markerfacecolor='white', markersize=15,
    #                      label='UAV')]
    # axR.legend(handles=data_types, bbox_to_anchor=(1.0, 1), loc='upper left')
    # axR.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    return img_plot, truths, tracks, measurements, clutter

# Set up the x, y, and z space for the 3D axis
xx = np.linspace(x_min, x_max, PIXELS_X)
yy = np.linspace(y_min, y_max, PIXELS_Y)
x, y = np.meshgrid(xx, yy)
zarray = np.zeros((PIXELS_X, PIXELS_Y, number_steps))

# Create the matplotlib figure and axes. Here we will have two axes being animated in sync.
# `axL` will be the a 3D axis showing the Gaussian mixture
# `axR` will be be a 2D axis showing the ground truth, detections, and updated tracks at
# each time step.
# plt.style.use('dark_background')
# fig = plt.figure(figsize=(16, 8))
# axL = fig.add_subplot(121, projection='3d')
# axR = fig.add_subplot(122)
fig, axR = plt.subplots(figsize=(12, 10))
plt.setp(axR.get_xticklabels(), fontsize=font_size)
plt.setp(axR.get_yticklabels(), fontsize=font_size)
axR.set_xlim(x_min, x_max)
axR.set_ylim(y_min, y_max)

# Add an initial surface to the left axis and scattered points on the right axis. Doing
# this now means that in the animate() function we only have to update these variables
# sf = axL.plot_surface(x, y, zarray[:, :, 0], cmap=cm.RdBu, linewidth=0, antialiased=False)
img_plot = axR.imshow(cv2.flip(zarray[:, :, 0],0),extent=[x_min,x_max,y_min,y_max],norm=cm.colors.Normalize(vmin=0, vmax=0.1))
cbar = fig.colorbar(img_plot)
cbar.set_label('Entropy',size=1.4*font_size)
cbar.ax.tick_params(labelsize=font_size)

truths = axR.scatter(x_min, y_min, c='blue', s=200, marker="+", linewidth=2, zorder=0.5, label="Ground Truth")
tracks = axR.scatter(x_min, y_min, c='red', s=200, marker="x" , linewidth=2, zorder=1, label="Track")
measurements = axR.scatter(x_min, y_min, c='red', s=200, marker=".", linewidth=2, zorder=0.5, label="Detection")
clutter = axR.scatter(x_min, y_min, c='green', s=200, marker="1", linewidth=2, zorder=0.5, label="Clutter")
UAV1_p = axR.scatter(x_min, y_min, s=200, c='white', marker="*", linewidth=1, zorder=2, label="UAV")
UAV2_p = axR.scatter(x_min, y_min, s=200, c='white', marker="*", linewidth=1, zorder=2)
UAV3_p = axR.scatter(x_min, y_min, s=200, c='white', marker="*", linewidth=1, zorder=2)

# tasks_p = axR.scatter(x_min, y_min, c='white', s=50, marker="+", linewidth=1, zorder=3, label="Tasks")

axR.legend(loc='upper right',prop={'size': 16})
# axR.legend(bbox_to_anchor=(1.0, 1), loc='upper left')

UAV1_FOV_Plot, = axR.plot([],[],'-w')
UAV2_FOV_Plot, = axR.plot([],[],'-w')
UAV3_FOV_Plot, = axR.plot([],[],'-w')

# Create and display the animation
from matplotlib import rc
anim = animation.FuncAnimation(fig, animate, frames=number_steps, interval=200,
                               fargs=(img_plot, truths, tracks, measurements, clutter), blit=False)
rc('animation', html='jshtml')
anim.save("output.mp4")
# anim.save("output.gif",writer="fisjiofjs")


# plt.style.use('dark_background')
# from stonesoup.plotter import Plotter
# plotter = Plotter()
# plotter.plot_ground_truths(truths, [0, 2])
# plotter.plot_measurements(all_measurements, [0, 2], color='g')
# plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
# plotter.ax.plot(UAV1[:,0], UAV1[:,1], '--')
# plotter.ax.plot(UAV2[:,0], UAV2[:,1], '--')
# plotter.ax.plot(UAV3[:,0], UAV3[:,1], '--')
# plotter.ax.set_xlim(-100, 100)
# plotter.ax.set_ylim(-100, 100)
# plt.show()

print("Completed")
