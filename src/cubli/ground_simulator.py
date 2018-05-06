import numpy as np
from pydrake.math import sin, cos
from pydrake.all import (SignalLogger, CompliantMaterial, ConstantVectorSource, DirectCollocation, DiagramBuilder, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult, AddModelInstancesFromSdfString,
                         MathematicalProgram, Simulator, BasicVector, AddFlatTerrainToWorld)

def contact_next_state(state, u, dt, dim=2):

    mu = 0.1 # friction force

    # use SNOPT in Drake
    mp = MathematicalProgram()

    # state length
    state_len = len(initial_state)
    floor_offset = -.01 # used to allow a little penitration

    # total time used (assuming equal time steps)
    time_used = mp.NewContinuousVariables(1, "time_used")
    dt = time_used/(N+1)

    # contact force decision variables
    f = mp.NewContinuousVariables(8, "f_%d" % 0) # only one input for the cube

    # next state decision variable
    x = mp.NewContinuousVariables(state_len, "x_%d" % 0) # for both input thrusters

    dynamic_state_next = state + get_nd_dynamics(state, u, f, dim)*dt

    # can't penitrate the floor
    distances = get_corner_distances(dynamic_state_next, dim)
    mp.AddConstraint(distances[0] >= floor_offset)
    mp.AddConstraint(distances[1] >= floor_offset)
    mp.AddConstraint(distances[2] >= floor_offset)
    mp.AddConstraint(distances[3] >= floor_offset)

    # ground forces can't pull on the ground
    for j in range(8):
        mp.AddConstraint(f[j] <= max_ground_force)
        mp.AddConstraint(f[j] >= 0)

    # add complimentary constraint
    force = f
    theta = state[dim]

    distances = get_corner_distances(state, dim)

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

    vector_0 = force[0] * force[1]
    vector_1 = force[2] * force[3]
    vector_2 = force[4] * force[5]
    vector_3 = force[6] * force[7]

    val = np.asarray([vector_0, vector_1, vector_2, vector_3])

    mp.AddConstraint(val.dot(distances) <= complimentarity_constraint_thresh)
    mp.AddConstraint(val.dot(distances) >= -complimentarity_constraint_thresh)

    print "Number of decision vars", mp.num_vars()
    print(mp.Solve())

    forces_computed = mp.GetSolution(f)

    return dynamic_state_next
