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
plt.rcParams['figure.figsize'] = (14, 12)
plt.style.use('seaborn-colorblind')
# Other general imports
import numpy as np
from datetime import datetime, timedelta

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

deg2rad = 3.14/180.0

number_steps = 150
death_probability = 0.000000005
birth_probability = 0.00002
probability_detection = 0.9
merge_threshold = 5
prune_threshold = 1E-8
state_threshold = 0.1

FOVsize=30.0*np.sqrt(2)
clutter_rate = 0.1*(1.0/100.0)*(FOVsize/np.sqrt(2))**2

start_time = datetime.now()
truths_by_time = []

def getFOVPolygon(UAV_pos,orient,size):
    corner1 = (UAV_pos[0]+size*np.cos(orient+45*deg2rad),UAV_pos[1]+size*np.sin(orient+45*deg2rad))
    corner2 = (UAV_pos[0]+size*np.cos(orient+135*deg2rad),UAV_pos[1]+size*np.sin(orient+135*deg2rad))
    corner3 = (UAV_pos[0]+size*np.cos(orient+225*deg2rad),UAV_pos[1]+size*np.sin(orient+225*deg2rad))
    corner4 = (UAV_pos[0]+size*np.cos(orient+315*deg2rad),UAV_pos[1]+size*np.sin(orient+315*deg2rad))
    return Polygon([corner1, corner2, corner3, corner4])

def inFOV(target_state,UAV_state):
    point = Point(target_state[0], target_state[2])
    polygon = getFOVPolygon([UAV_state[0],UAV_state[1]],UAV_state[2],FOVsize)

    return polygon.contains(point)

## Generate UAV positions and orientations
UAV1 = []
UAV2 = []
UAV3 = []
for i in range(number_steps):
    UAV1.append([-30+30*np.cos(i/10),30+10*np.sin(-i/10),-i/10])
    UAV2.append([0+30*np.cos(i/10),0+30*np.sin(i/10),i/10])
    UAV3.append([-30+10*np.cos(i/10),-30+30*np.sin(-i/10),-i/10])

UAV1 = np.array(UAV1)
UAV2 = np.array(UAV2)
UAV3 = np.array(UAV3)
# print(UAV1[:,0])

## Generate FOVPolygons for plotting
UAV1_Polygon = []
UAV2_Polygon = []
UAV3_Polygon = []
for i in range(number_steps):
    UAV1_Polygon.append(getFOVPolygon([UAV1[i][0],UAV1[i][1]],UAV1[i][2],FOVsize))
    UAV2_Polygon.append(getFOVPolygon([UAV2[i][0],UAV2[i][1]],UAV2[i][2],FOVsize))
    UAV3_Polygon.append(getFOVPolygon([UAV3[i][0],UAV3[i][1]],UAV3[i][2],FOVsize))

# Create transition model
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(0.0001), ConstantVelocity(0.0001)))

# Make the measurement model
from stonesoup.models.measurement.linear import LinearGaussian
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
        x_vel = -iter*1*np.sign(np.sin(k/10))
        y_vel = iter*1*np.sign(np.cos(k/10))
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

for k in range(number_steps):
    measurement_set = set()
    timestamp = start_time + timedelta(seconds=k)

    for truth in truths:
        # try:
        #     truth_state = truth[timestamp]
        # except IndexError:
        #     # This truth not alive at this time. Skip this iteration of the for loop.
        #     continue

        truth_state = truth[timestamp]

        if (inFOV(truth_state.state_vector,UAV1[k]) 
            or inFOV(truth_state.state_vector,UAV2[k]) 
            or inFOV(truth_state.state_vector,UAV3[k])):

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
                x = uniform.rvs(-200, 400)
                y = uniform.rvs(-200, 400)
                if(inFOV([x,0,y,0],UAV1[k]) 
                    or inFOV([x,0,y,0],UAV2[k]) 
                    or inFOV([x,0,y,0],UAV3[k])):
                    measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=timestamp,
                                        measurement_model=measurement_model))
                    break

    all_measurements.append(measurement_set)


from stonesoup.updater.kalman import KalmanUpdater
kalman_updater = KalmanUpdater(measurement_model)

# Area in which we look for target. Note that if a target appears outside of this area the
# filter will not pick up on it.
meas_range = np.array([[-1, 1], [-1, 1]])*400
clutter_spatial_density = clutter_rate/np.prod(np.diff(meas_range))

from stonesoup.updater.pointprocess import PHDUpdater
updater = PHDUpdater(
    kalman_updater,
    clutter_spatial_density=clutter_spatial_density,
    prob_detection=probability_detection,
    prob_survival=1-death_probability)

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

for n, measurements in enumerate(all_measurements):
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
        # if reduced_state.weight > 0.001: all_gaussians[n].append(reduced_state)

        all_gaussians[n].append(reduced_state)
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

x_min, x_max, y_min, y_max = -100, 100, -100, 100

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


from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D  # Will be used when making the legend

# # This is the function that updates the figure we will be animating. As parameters we must
# # pass in the elements that will be changed, as well as the index i
# def animate(i, sf, truths, tracks, measurements, clutter):
#     # Set up the axes
#     axL.clear()
#     axR.set_title('Tracking Space at k='+str(i))
#     axL.set_xlabel("x")
#     axL.set_ylabel("y")
#     axL.set_title('PDF of the Gaussian Mixture')
#     axL.view_init(elev=30, azim=-80)
#     axL.set_zlim(0, 0.3)

#     # Initialize the variables
#     weights = []  # weights of each Gaussian. This is analogous to the probability of its existence
#     means = []    # means of each Gaussian. This is equal to the x and y of its state vector
#     sigmas = []   # standard deviation of each Gaussian.

#     # Fill the lists of weights, means, and standard deviations
#     for state in all_gaussians[i]:
#         weights.append(state.weight)
#         means.append([state.state_vector[0], state.state_vector[2]])
#         sigmas.append([state.covar[0][0], state.covar[1][1]])
#     means = np.array(means)
#     sigmas = np.array(sigmas)

#     # Generate the z values over the space and plot on the left axis
#     zarray[:, :, i] = get_mixture_density(x, y, weights, means, sigmas)
#     sf = axL.plot_surface(x, y, zarray[:, :, i], cmap=cm.RdBu, linewidth=0, antialiased=False)

#     # Make lists to hold the new ground truths, tracks, detections, and clutter
#     new_truths, new_tracks, new_measurements, new_clutter, new_UAV1_p = [], [], [], [], []
#     for truth in truths_by_time[i]:
#         new_truths.append([truth.state_vector[0], truth.state_vector[2]])
#     for state in tracks_by_time[i]:
#         new_tracks.append([state.state_vector[0], state.state_vector[2]])
#     for measurement in all_measurements[i]:
#         if isinstance(measurement, TrueDetection):
#             new_measurements.append([measurement.state_vector[0], measurement.state_vector[1]])
#         elif isinstance(measurement, Clutter):
#             new_clutter.append([measurement.state_vector[0], measurement.state_vector[1]])

#     # Plot the contents of these lists on the right axis
#     if new_truths:
#         truths.set_offsets(new_truths)
#     if new_tracks:
#         tracks.set_offsets(new_tracks)
#     if new_measurements:
#         measurements.set_offsets(new_measurements)
#     if new_clutter:
#         clutter.set_offsets(new_clutter)

#     UAV1_p.set_offsets([UAV1[i][0],UAV1[i][1]])
#     UAV2_p.set_offsets([UAV2[i][0],UAV2[i][1]])
#     UAV3_p.set_offsets([UAV3[i][0],UAV3[i][1]])

#     UAV1_FOV_Plot.set_data(*UAV1_Polygon[i].exterior.xy)
#     UAV2_FOV_Plot.set_data(*UAV2_Polygon[i].exterior.xy)
#     UAV3_FOV_Plot.set_data(*UAV3_Polygon[i].exterior.xy)

#     # Create a legend. The use of Line2D is purely for the visual in the legend
#     data_types = [Line2D([0], [0], color='white', marker='o', markerfacecolor='blue', markersize=15,
#                          label='Ground Truth'),
#                  Line2D([0], [0], color='white', marker='1', markerfacecolor='green', markersize=15,
#                          label='Clutter'),
#                  Line2D([0], [0], color='white', marker='1', markerfacecolor='orange', markersize=15,
#                          label='Detection'),
#                  Line2D([0], [0], color='white', marker='x', markerfacecolor='red', markersize=15,
#                          label='Track'),
#                  Line2D([0], [0], color='white', marker='*', markerfacecolor='white', markersize=15,
#                          label='UAV')]
#     axR.legend(handles=data_types, bbox_to_anchor=(1.0, 1), loc='upper left')

#     return sf, truths, tracks, measurements, clutter

# # Set up the x, y, and z space for the 3D axis
# xx = np.linspace(x_min-5, x_max+5, 100)
# yy = np.linspace(y_min-5, y_max+5, 100)
# x, y = np.meshgrid(xx, yy)
# zarray = np.zeros((100, 100, number_steps))

# # Create the matplotlib figure and axes. Here we will have two axes being animated in sync.
# # `axL` will be the a 3D axis showing the Gaussian mixture
# # `axR` will be be a 2D axis showing the ground truth, detections, and updated tracks at
# # each time step.
# plt.style.use('dark_background')
# fig = plt.figure(figsize=(16, 8))
# axL = fig.add_subplot(121, projection='3d')
# axR = fig.add_subplot(122)
# axR.set_xlim(x_min-5, x_max+5)
# axR.set_ylim(y_min-5, y_max+5)

# # Add an initial surface to the left axis and scattered points on the right axis. Doing
# # this now means that in the animate() function we only have to update these variables
# sf = axL.plot_surface(x, y, zarray[:, :, 0], cmap=cm.RdBu, linewidth=0, antialiased=False)
# truths = axR.scatter(x_min-10, y_min-10, c='blue', linewidth=6, zorder=0.5)
# tracks = axR.scatter(x_min-10, y_min-10, c='red', s=200, marker="x" , linewidth=2, zorder=1)
# measurements = axR.scatter(x_min-10, y_min-10, c='orange', s=200, marker="1", linewidth=3, zorder=0.5)
# clutter = axR.scatter(x_min-10, y_min-10, c='green', s=200, marker="1", linewidth=2, zorder=0.5)
# UAV1_p = axR.scatter(x_min-10, y_min-10, s=200, c='white', marker="*", linewidth=1, zorder=2)
# UAV2_p = axR.scatter(x_min-10, y_min-10, s=200, c='white', marker="*", linewidth=1, zorder=2)
# UAV3_p = axR.scatter(x_min-10, y_min-10, s=200, c='white', marker="*", linewidth=1, zorder=2)

# UAV1_FOV_Plot, = axR.plot([],[],'-w')
# UAV2_FOV_Plot, = axR.plot([],[],'-w')
# UAV3_FOV_Plot, = axR.plot([],[],'-w')

# # Create and display the animation
# from matplotlib import rc
# anim = animation.FuncAnimation(fig, animate, frames=number_steps, interval=100,
#                                fargs=(sf, truths, tracks, measurements, clutter), blit=False)
# rc('animation', html='jshtml')
# anim.save("output.gif")



plt.style.use('dark_background')
from stonesoup.plotter import Plotter
plotter = Plotter()
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(all_measurements, [0, 2], color='g')
plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
plotter.ax.plot(UAV1[:,0], UAV1[:,1], '--')
plotter.ax.plot(UAV2[:,0], UAV2[:,1], '--')
plotter.ax.plot(UAV3[:,0], UAV3[:,1], '--')
plotter.ax.set_xlim(-100, 100)
plotter.ax.set_ylim(-100, 100)
plt.show()

print("Completed")