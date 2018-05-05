import numpy as np
from pydrake.math import sin, cos

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

def ground_distances(state):
    # array of [0,1,2,3]
    # add .5 for the offset

    z = state[1]
    theta = state[2]

    offset = .5*np.sqrt(2)*sin(np.pi/4.0+theta)
    val = sin(theta)

    dist_0 = z - offset
    dist_1 = dist_0 + val
    dist_2 = z + offset
    dist_3 = dist_2 - val

    dist_0 += .5
    dist_1 += .5
    dist_2 += .5
    dist_3 += .5

    return np.asarray([dist_0, dist_1, dist_2, dist_3])

def cube_dynamics(state, u, force):

    # Need to grab important parameters
    M_c = 1.0 # self.M_c
    M_w = 1.0 # self.M_w
    M_t = M_c + M_w

    I_c = 1.0 #self.I_c
    I_w = 1.0 #self.I_w
    I_t = I_c + I_w

    # Distance from edge to center of cube
    L_t = np.sqrt(2) #np.sqrt.(2*self.L)

    # Assuming friction is 0 right now
    F_c = 0.5
    F_w = 0.5

    g = 9.81 # self.g

    # Relevant states are x,z,thetay, phi
    x = state[0]
    z = state[1]
    thetay = state[2]
    phi = state[3]

    # Velocity States
    xdot = state[4]
    zdot = state[5]
    thetaydot = state[6]
    phidot = state[7]

    # Setup the derivative of the state vector
    derivs = np.zeros_like(state)
    derivs[0:4] = state[4:]

    # Ballistic Dynamics
    derivs[4] = (force[1] - force[2] + force[6] - force[5])*cos(thetay) - (force[0] + force[3] - force[4] - force[7])*sin(thetay) # forces along x
    derivs[5] = (force[1] - force[2] + force[6] - force[5])*sin(thetay) + (force[0] + force[3] - force[4] - force[7])*cos(thetay) - g  # forces in y direction

    # Back torque due to wheel
    derivs[6] = (-u[0] + F_w*phidot - F_c*thetaydot)/I_c + (-force[0]+force[1]-force[2]+force[3]-force[4]+force[5]-force[6]+force[7])*.5

    # Wheel accel
    derivs[7] = (u[0]*I_t + F_c*thetaydot*I_w - F_w*phidot*I_t)/(I_w*I_c)

    return derivs

class CubeVisualizer:
    def __init__(self):
        self.vis = meshcat.Visualizer()

        self.cube = self.vis["cube"]
        self.pivot = self.cube["pivot"]
        self.wheel = self.pivot["wheel"]

        # create and draw the cube
        self.cube_dim = [1.0,1.0,1.0] # x,y,z
        self.cube.set_object(g.Box(self.cube_dim))

        # pivot and wheel
        self.pivot.set_transform(tf.translation_matrix([0,0,0])) # set location of pole
        wheel_dim = [1.5,.5,.5] # x,y,z
        self.wheel.set_object(g.Box(wheel_dim))

        self.initialize()

    def draw_transformation(self, state):
        state = list(state)
        origin = state[0:3]
        origin[0] = 0.0
        origin[1] = state[0]
        origin[2] = state[1] + self.cube_dim[2]/2.0
        theta = state[2]
        wheel_angle = state[3]
        temp = tf.rotation_matrix(theta,[1,0,0]) # assume rotate about y
        temp[0:3, -1] = tf.translation_from_matrix(tf.translation_matrix(origin))
        self.cube.set_transform(temp)
        self.wheel.set_transform(tf.rotation_matrix(-wheel_angle,[1,0,0])) # rotate the pole

    def initialize(self):
        # set the initial state
        x = 0.0
        z = 0.0
        x_dot = 0.0
        z_dot = 0.0
        thetay = 0.0
        # state of the flywheel
        phi = 0.0
        phi_dot = 0.0

        state_initial = (x,z,thetay,phi,x_dot,z_dot,0.,phi_dot)
        self.draw_transformation(state_initial)
