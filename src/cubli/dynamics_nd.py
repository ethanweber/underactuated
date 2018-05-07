import numpy as np
from pydrake.math import sin, cos
from pydrake.all import (SignalLogger, CompliantMaterial, ConstantVectorSource, DirectCollocation, DiagramBuilder, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult, AddModelInstancesFromSdfString,
                         MathematicalProgram, Simulator, BasicVector, AddFlatTerrainToWorld)

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

def get_nd_state(state, dim=2):
    """Return higher dimensional state given a state in 2D.

    Keyword arguments:
    state -- the state of the cube in 2D
    dim -- the desired dimension (default 2)
    """
    assert len(state) == 8 # make sure input is 2D
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

def get_nd_dynamics(state, u, force, dim=2):
    """Return state space dyanmics in n dimensions.

    Keyword arguments:
    state -- the state of the cube
    u -- the input torque on the wheel
    force -- the 8 forces acting on the wheel corners
    dim -- the dimension of the state space (default 2)
    """
    # half state length
    hl = len(state) / 2

    # mass
    m_c = 1.0 # cube
    m_w = 1.0 # wheel
    m_t = m_c + m_w # total

    # moments of inertia
    I_c = 1.0
    I_w = 1.0
    I_t = I_c + I_w

    # friction
    F_c = 0.5
    F_w = 0.5

    # gravity
    g = 9.81

    # unpack the states
    # x = state[0]
    # y = state[1]
    theta = state[dim]
    # alpha = state[hl-1]
    # xdot = state[0+hl]
    # ydot = state[1+hl]
    theta_dot = state[dim+hl]
    alpha_dot = state[-1]

    # derivative vector
    derivs = np.zeros_like(state)
    derivs[0:hl] = state[hl:] # set velocities

    # ballistic dynamics
    derivs[0+hl] = (force[1] - force[2] + force[6] - force[5])*cos(theta) - (force[0] + force[3] - force[4] - force[7])*sin(theta) # forces along x
    derivs[1+hl] = (force[1] - force[2] + force[6] - force[5])*sin(theta) + (force[0] + force[3] - force[4] - force[7])*cos(theta) - g  # forces in y direction

    # cube angle acceleration
    derivs[dim + hl] = (-u[0] + F_w*alpha_dot - F_c*theta_dot)/I_c + (-force[0]+force[1]-force[2]+force[3]-force[4]+force[5]-force[6]+force[7])*.5

    # wheel acceleration
    derivs[-1] = (u[0]*I_t + F_c*theta_dot*I_w - F_w*alpha_dot*I_t)/(I_w*I_c)

    return derivs

def get_corner_distances(state, dim=2):
    """Return distances to ground in the corner order [0,1,2,3] as a numpy array.

    Keyword arguments:
    state -- the state of the cube
    dim -- the dimension of the state space (default 2)
    """

    y = state[1]
    theta = state[dim]

    offset = .5*np.sqrt(2)*sin(np.pi/4.0+theta)
    val = sin(theta)

    dist_0 = y - offset
    dist_1 = dist_0 + val
    dist_2 = y + offset
    dist_3 = dist_2 - val

    # add .5 for the offset
    dist_0 += .5
    dist_1 += .5
    dist_2 += .5
    dist_3 += .5

    return np.asarray([dist_0, dist_1, dist_2, dist_3])

def get_corner_x_positions(state, dim=2):
    """Return x position of the corners in the order [0,1,2,3] as a numpy array.

    Keyword arguments:
    state -- the state of the cube
    dim -- the dimension of the state space (default 2)
    """

    x = state[0]
    theta = state[dim]

    offset = .5*np.sqrt(2)*cos(np.pi/4.0+theta)
    val = cos(theta)

    pos_0 = x - offset
    pos_1 = pos_0 + val
    pos_2 = x + offset
    pos_3 = pos_2 - val

    return np.asarray([pos_0, pos_1, pos_2, pos_3])

def compute_optimal_control(initial_state, final_state, min_time, max_time, max_torque, dim=2):

    print("Initial State: {}".format(initial_state))
    print("Final State: {}".format(final_state))

    # a few checks
    assert(len(initial_state) == len(final_state))
    assert(min_time <= max_time)

    # some values that can be changed if desired
    N = 50 # number knot points
    dynamics_error_thresh = 0.01 # error thresh on xdot = f(x,u,f)
    floor_offset = -.01 # used to allow a little penitration
    final_state_error_thresh = 0.01 # final state error thresh
    max_ground_force = 100
    # impose contraint to stay on the ground
    stay_on_ground = True
    stay_on_ground_tolerance = 0.1 # tolerance for leaving the ground if contstraint is used
    complimentarity_constraint_thresh = 0.0001

    fix_corner_on_ground = True
    corner_fix = [0, -0.5] # corner index, location

    mu = 0.1 # friction force

    # use SNOPT in Drake
    mp = MathematicalProgram()

    # state length
    state_len = len(initial_state)

    # total time used (assuming equal time steps)
    time_used = mp.NewContinuousVariables(1, "time_used")
    dt = time_used/(N+1)

    # input torque decision variables
    u = mp.NewContinuousVariables(1, "u_%d" % 0) # only one input for the cube
    u_over_time = u
    for k in range(1,N):
        u = mp.NewContinuousVariables(1, "u_%d" % k)
        u_over_time = np.vstack((u_over_time, u))
    total_u = u_over_time

    # contact force decision variables
    f = mp.NewContinuousVariables(8, "f_%d" % 0) # only one input for the cube
    f_over_time = f
    for k in range(1,N):
        f = mp.NewContinuousVariables(8, "f_%d" % k)
        f_over_time = np.vstack((f_over_time, f))
    total_f = f_over_time

    # state decision variables
    x = mp.NewContinuousVariables(state_len, "x_%d" % 0) # for both input thrusters
    x_over_time = x
    for k in range(1,N+1):
        x = mp.NewContinuousVariables(state_len, "x_%d" % k)
        x_over_time = np.vstack((x_over_time, x))
    total_x = x_over_time

    # impose dynamic constraints
    for n in range(N):
        state_next = total_x[n+1]
        dynamic_state_next = total_x[n,:] + get_nd_dynamics(total_x[n,:], total_u[n,:], total_f[n,:], dim)*dt
        # make sure the actual and predicted align to follow dynamics
        for j in range(state_len):
            state_error = state_next[j] - dynamic_state_next[j]
            mp.AddConstraint(state_error <= dynamics_error_thresh)
            mp.AddConstraint(state_error >= -dynamics_error_thresh)

    # can't penitrate the floor and can't leave the floor
    for n in range(N):
        distances = get_corner_distances(total_x[n,:], dim)

        mp.AddConstraint(distances[0] >= floor_offset)
        mp.AddConstraint(distances[1] >= floor_offset)
        mp.AddConstraint(distances[2] >= floor_offset)
        mp.AddConstraint(distances[3] >= floor_offset)

        # don't leave the ground if specified
        if stay_on_ground == True:
            mp.AddConstraint(distances[0] <= np.sqrt(2)+stay_on_ground_tolerance)
            mp.AddConstraint(distances[1] <= np.sqrt(2)+stay_on_ground_tolerance)
            mp.AddConstraint(distances[2] <= np.sqrt(2)+stay_on_ground_tolerance)
            mp.AddConstraint(distances[3] <= np.sqrt(2)+stay_on_ground_tolerance)

        if fix_corner_on_ground == True:
            x_pos = get_corner_x_positions(total_x[n,:], dim)
            # make left corner on the ground
            mp.AddConstraint(distances[0] == 0.0)
            num, loc = corner_fix
            mp.AddConstraint(x_pos[num] == loc)

    # ground forces can't pull on the ground
    for n in range(N):
        force = total_f[n]
        for j in range(8):
            mp.AddConstraint(force[j] <= max_ground_force)
            mp.AddConstraint(force[j] >= 0)

    # add complimentary constraint
    for n in range(N):
        force = total_f[n]
        state = total_x[n]
        theta = state[dim]

        distances = get_corner_distances(total_x[n+1,:], dim)

        s = sin(theta)
        c = cos(theta)

        z_0 = force[0]*c + force[1]*s
        z_1 = - force[2]*s + force[3]*c
        z_2 = - force[4]*c - force[5]*s
        z_3 = force[6]*s - force[7]*c

        xy_0 = - force[0]*s + force[1]*c
        xy_1 = - force[2]*c - force[3]*s
        xy_2 = force[4]*s - force[5]*c
        xy_3 = force[6]*c + force[7]*s

        mp.AddConstraint(xy_0 <= z_0*mu)
        mp.AddConstraint(xy_0 >= -z_0*mu)
        mp.AddConstraint(xy_1 <= z_1*mu)
        mp.AddConstraint(xy_1 >= -z_1*mu)
        mp.AddConstraint(xy_2 <= z_2*mu)
        mp.AddConstraint(xy_2 >= -z_2*mu)
        mp.AddConstraint(xy_3 <= z_3*mu)
        mp.AddConstraint(xy_3 >= -z_3*mu)

        # vector_0 = force[0] * force[1]
        # vector_1 = force[2] * force[3]
        # vector_2 = force[4] * force[5]
        # vector_3 = force[6] * force[7]
        # val = np.asarray([vector_0, vector_1, vector_2, vector_3])
        # mp.AddConstraint(val.dot(distances) <= complimentarity_constraint_thresh)
        # mp.AddConstraint(val.dot(distances) >= -complimentarity_constraint_thresh)

        val_0 = np.asarray([force[0], force[2], force[4], force[6]])
        val_1 = np.asarray([force[1], force[3], force[5], force[7]])
        mp.AddConstraint(val_0.dot(distances) <= complimentarity_constraint_thresh)
        mp.AddConstraint(val_0.dot(distances) >= -complimentarity_constraint_thresh)
        mp.AddConstraint(val_1.dot(distances) <= complimentarity_constraint_thresh)
        mp.AddConstraint(val_1.dot(distances) >= -complimentarity_constraint_thresh)

    # initial state, no state error allowed
    for i in range(state_len):
        initial_state_error = x_over_time[0,i] - initial_state[i]
        mp.AddConstraint(initial_state_error == 0.0)

    # don't care about final wheel angle, so skip restriction in it
    state_indices = [i for i in range(0, state_len)]
    a = state_indices[0:state_len/2-1] + state_indices[state_len/2:]
    for i in a:
        # final
        final_state_error = x_over_time[-1,i] - final_state[i]
        mp.AddConstraint(final_state_error <= final_state_error_thresh)
        mp.AddConstraint(final_state_error >= -final_state_error_thresh)

    # add time constraint
    mp.AddConstraint(time_used[0] >= min_time)
    mp.AddConstraint(time_used[0] <= max_time)

    # add torque constraints
    for n in range(N):
        mp.AddConstraint(u_over_time[n,0] <= max_torque)
        mp.AddConstraint(u_over_time[n,0] >= -max_torque)

    # try to keep the velocity of the wheel in the correct direction
    # mp.AddLinearCost(x_over_time[:,-1].sum())

    # mp.AddLinearCost(-x_over_time[:,1].sum())
    # mp.AddLinearCost(-x_over_time[N//2,1])

    print "Number of decision vars", mp.num_vars()
    print(mp.Solve())

    trajectory = mp.GetSolution(x_over_time)
    input_trajectory = mp.GetSolution(u_over_time)
    force_trajectory = mp.GetSolution(f_over_time)
    t = mp.GetSolution(time_used)
    time_array = np.arange(0.0, t, t/(N+1))

    return trajectory, input_trajectory, force_trajectory, time_array

class MeshcatCubeVisualizer:
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

    def draw_transformation(self, state, dim=2.0):
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
        self.wheel.set_transform(tf.rotation_matrix(wheel_angle,[1,0,0])) # rotate the pole

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


def contact_next_state(state, u, dt, dim=2):
    """Return the next state after computing the force for the given timestep.
    Note that this is not guaranteed to work realtime or even every time.

    Keyword arguments:
    state -- the state of the cube
    u -- the input torque on the wheel
    dt -- the timestep size
    dim -- the dimension of the state space (default 2)
    """

    mu = 0.1 # friction force
    max_ground_force = 200
    complimentarity_constraint_thresh = 0.01

    # use SNOPT in Drake
    mp = MathematicalProgram()

    floor_offset = -0.01 # used to allow a little penitration

    x = mp.NewContinuousVariables(len(state), "x_%d" % 0)
    u_decision = mp.NewContinuousVariables(len(u), "u_%d" % 0)
    f = mp.NewContinuousVariables(8, "f_%d" % 0)

    # starting values
    for i in range(len(x)):
        mp.AddConstraint(x[i] == state[i])
    for i in range(len(u)):
        mp.AddConstraint(u_decision[i] == u[i])

    dynamic_state_next = x[:] + get_nd_dynamics(x[:], u_decision[:], f[:], dim)*dt

    # can't penitrate the floor
    distances = get_corner_distances(dynamic_state_next, dim)
    mp.AddConstraint(distances[0] >= floor_offset)
    mp.AddConstraint(distances[1] >= floor_offset)
    mp.AddConstraint(distances[2] >= floor_offset)
    mp.AddConstraint(distances[3] >= floor_offset)

    # ground forces can't pull on the ground
    for j in range(8):
        # mp.AddConstraint(f[j] <= max_ground_force)
        mp.AddConstraint(f[j] >= 0)

    # add complimentary constraint
    theta = state[dim]

    distances = get_corner_distances(state, dim)

    s = sin(theta)
    c = cos(theta)

    z_0 = f[0]*c + f[1]*s
    z_1 = - f[2]*s + f[3]*c
    z_2 = - f[4]*c - f[5]*s
    z_3 = f[6]*s - f[7]*c

    xy_0 = - f[0]*s + f[1]*c
    xy_1 = - f[2]*c - f[3]*s
    xy_2 = f[4]*s - f[5]*c
    xy_3 = f[6]*c + f[7]*s

    mp.AddConstraint(xy_0 <= z_0*mu)
    mp.AddConstraint(xy_0 >= -z_0*mu)
    mp.AddConstraint(xy_1 <= z_1*mu)
    mp.AddConstraint(xy_1 >= -z_1*mu)
    mp.AddConstraint(xy_2 <= z_2*mu)
    mp.AddConstraint(xy_2 >= -z_2*mu)
    mp.AddConstraint(xy_3 <= z_3*mu)
    mp.AddConstraint(xy_3 >= -z_3*mu)

    vector_0 = f[0] * f[1]
    vector_1 = f[2] * f[3]
    vector_2 = f[4] * f[5]
    vector_3 = f[6] * f[7]

    val = np.asarray([vector_0, vector_1, vector_2, vector_3])

    mp.AddConstraint(val.dot(distances) <= complimentarity_constraint_thresh)
    mp.AddConstraint(val.dot(distances) >= -complimentarity_constraint_thresh)

    # print "Number of decision vars", mp.num_vars()
    # print(mp.Solve())
    mp.Solve()

    f_comp = mp.GetSolution(f)
    return state + get_nd_dynamics(state, u, f_comp, dim)*dt
