import numpy as np
from pydrake.math import sin, cos

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

def get_nd_state(state, dim):
    x,y,theta,alpha,x_dot,y_dot,theta_dot,alpha_dot = state
    if dim > 2:
        nd_state = [x] + [y] + [0]*(dim-2)
        nd_state += [theta] + [0]*(dim-1)
        nd_state += [alpha]

        nd_state += [x_dot] + [y_dot] + [0]*(dim-2)
        nd_state += [theta_dot] + [0]*(dim-1)
        nd_state += [alpha_dot]
        return nd_state
    return state

def nd_cube_dynamics(state, dim, u, force):

    # print(state)

    # half state length
    hl = len(state) / 2

    # Need to grab important parameters
    M_c = 1.0 # self.M_c
    M_w = 1.0 # self.M_w
    M_t = M_c + M_w

    I_c = 1.0 #self.I_c
    I_w = 1.0 #self.I_w
    I_t = I_c + I_w

    # friction
    F_c = 0.5
    F_w = 0.5

    g = 9.81 # self.g

    # relevant state
    x = state[0]
    y = state[1]
    theta = state[dim]
    alpha = state[hl-1]

    # Velocity States
    xdot = state[0+hl]
    ydot = state[1+hl]
    theta_dot = state[dim+hl]
    alpha_dot = state[-1]

    # Setup the derivative of the state vector
    derivs = np.zeros_like(state)
    # print(derivs)
    # print(hl)
    derivs[0:hl] = state[hl:]

    # Ballistic Dynamics
    derivs[0+hl] = (force[1] - force[2] + force[6] - force[5])*cos(theta) - (force[0] + force[3] - force[4] - force[7])*sin(theta) # forces along x
    derivs[1+hl] = (force[1] - force[2] + force[6] - force[5])*sin(theta) + (force[0] + force[3] - force[4] - force[7])*cos(theta) - g  # forces in y direction

    # Back torque due to wheel
    derivs[dim + hl] = (-u[0] + F_w*alpha_dot - F_c*theta_dot)/I_c + (-force[0]+force[1]-force[2]+force[3]-force[4]+force[5]-force[6]+force[7])*.5

    # Wheel accel
    derivs[-1] = (u[0]*I_t + F_c*theta_dot*I_w - F_w*alpha_dot*I_t)/(I_w*I_c)

    return derivs


def ground_distances(state, dim):
    # array of [0,1,2,3]
    # add .5 for the offset

    y = state[1]
    theta = state[dim]

    offset = .5*np.sqrt(2)*sin(np.pi/4.0+theta)
    val = sin(theta)

    dist_0 = y - offset
    dist_1 = dist_0 + val
    dist_2 = y + offset
    dist_3 = dist_2 - val

    dist_0 += .5
    dist_1 += .5
    dist_2 += .5
    dist_3 += .5

    return np.asarray([dist_0, dist_1, dist_2, dist_3])

def corner_x_pos(state, dim):
    # array of [0,1,2,3]
    # add .5 for the offset

    x = state[0]
    theta = state[dim]

    offset = .5*np.sqrt(2)*cos(np.pi/4.0+theta)
    val = cos(theta)

    dist_0 = x - offset
    dist_1 = dist_0 + val
    dist_2 = x + offset
    dist_3 = dist_2 - val

    return np.asarray([dist_0, dist_1, dist_2, dist_3])

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

    def draw_transformation(self, state, dim):
        nd = len(state)
        state = list(state)
        origin = state[0:3] # so you can edit
        origin[0] = 0.0
        origin[1] = state[0]
        origin[2] = state[1] + self.cube_dim[2]/2.0
        theta = state[dim]
        wheel_angle = state[len(state)/2 - 1]
        temp = tf.rotation_matrix(theta,[1,0,0]) # assume rotate about y
        temp[0:3, -1] = tf.translation_from_matrix(tf.translation_matrix(origin))
        self.cube.set_transform(temp)
        self.wheel.set_transform(tf.rotation_matrix(-wheel_angle,[1,0,0])) # rotate the pole

    def initialize(self):
        # set the initial state in 2d
        x = 0.0
        y = 0.0
        x_dot = 0.0
        y_dot = 0.0
        theta = 0.0
        theta_dot = 0.0
        # state of the flywheel
        alpha = 0.0
        alpha_dot = 0.0

        state_initial = (x,y,theta,alpha,x_dot,y_dot,theta_dot,alpha_dot)
        self.draw_transformation(state_initial, 2)
