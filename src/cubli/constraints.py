# define thresholds throughout

from dynamics_nd import *

floor_offset = -.01 # used to allow a little penitration
def add_floor_constraint(mp, state, dim=2):
    distances = get_corner_distances(state[:], dim)
    mp.AddConstraint(distances[0] >= floor_offset)
    mp.AddConstraint(distances[1] >= floor_offset)
    mp.AddConstraint(distances[2] >= floor_offset)
    mp.AddConstraint(distances[3] >= floor_offset)


stay_on_ground_tolerance = 0.1 # tolerance for leaving the ground if contstraint is used
def stay_on_ground(mp, state, dim=2):
    # don't leave the ground if specified
    distances = get_corner_distances(state[:], dim)
    mp.AddConstraint(distances[0] <= np.sqrt(2)+stay_on_ground_tolerance)
    mp.AddConstraint(distances[1] <= np.sqrt(2)+stay_on_ground_tolerance)
    mp.AddConstraint(distances[2] <= np.sqrt(2)+stay_on_ground_tolerance)
    mp.AddConstraint(distances[3] <= np.sqrt(2)+stay_on_ground_tolerance)


def fix_corner_to_ground(mp, state, corner_index=0, x_coord=-0.5, dim=2):
    distances = get_corner_distances(state[:], dim)
    # make left corner on the ground in specified position
    x_pos = get_corner_x_positions(state, dim)
    mp.AddConstraint(distances[corner_index] == 0.0)
    mp.AddConstraint(x_pos[corner_index] == x_coord)


max_ground_force = 100
def dont_pull_on_ground(mp, force, dim=2):
    for j in range(len(force)):
        mp.AddConstraint(force[j] <= max_ground_force)
        mp.AddConstraint(force[j] >= 0)


complimentarity_constraint_thresh = 0.0
mu = 0.001 # friction force
def complimentarity_constraint(mp, state, force, dim=2):
    theta = state[dim]

    distances = get_corner_distances(state[:], dim)

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

    val_0 = np.asarray([force[0], force[2], force[4], force[6]])
    val_1 = np.asarray([force[1], force[3], force[5], force[7]])
    mp.AddConstraint(val_0.dot(distances) <= complimentarity_constraint_thresh)
    mp.AddConstraint(val_0.dot(distances) >= -complimentarity_constraint_thresh)
    mp.AddConstraint(val_1.dot(distances) <= complimentarity_constraint_thresh)
    mp.AddConstraint(val_1.dot(distances) >= -complimentarity_constraint_thresh)


# set the entire start state
def set_initial_state(mp, x, current_state, dim):
    for i in range(len(x)):
        mp.AddConstraint(x[i] == current_state[i])
