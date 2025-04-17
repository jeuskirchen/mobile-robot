import numpy as np
from scipy.optimize import least_squares
from math import pi 


def compute_residuals(pos, beacon_positions, distances):
    """
    Returns list of errors between a given position (pos) and all beacon positions
    where the error is simply the estimated distance minus actual distance 
    """
    residuals = []
    pos = np.array(pos)
    for b, d in zip(beacon_positions, distances):
        residuals.append(np.linalg.norm(pos-b)-d)
    return residuals 

def trilaterate(beacon_positions, beacon_distances, beacon_bearings):
    """
    Find point such that distances from it to known beacons are the same as the observed distances
    """
    if len(beacon_positions) < 3:
        # TODO: perhaps also check if it the constellation is "degenerate" 
        #  Because right now, this happens sometimes and the robot gets tricked into believing 
        #  in a very wrong trilaterated position! 
        return 
    
    # Position
    beacon_positions = np.array(beacon_positions)
    beacon_distances = np.array(beacon_distances)
    # Initial guess: midpoints between beacons
    initial_guess = np.mean(beacon_positions, axis=0)
    # Least squares optimization
    result = least_squares(compute_residuals, initial_guess, args=(beacon_positions, beacon_distances))
    est_x, est_y = result.x

    # Orientation
    #  Calculate orientation using bearing, estimated robot position and actual beacon position
    #  using atan2. Take average of this calculation over all beacons (using cos, sin)
    angle_x, angle_y = 0, 0
    for (beacon_x, beacon_y), beacon_bearing in zip(beacon_positions, beacon_bearings):
        angle = np.atan2(beacon_y-est_y, beacon_x-est_x)-beacon_bearing
        # Wrap into interval [0, 2π]:
        angle %= 2*pi 
        angle_x += np.cos(angle)
        angle_y += np.sin(angle)
    angle_x = angle_x/len(beacon_positions)
    angle_y = angle_y/len(beacon_positions)
    est_angle = np.atan2(angle_y, angle_x)
    est_angle %= 2*pi 

    return est_x, est_y, est_angle 
