import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (SignalLogger, CompliantMaterial, ConstantVectorSource, DirectCollocation, DiagramBuilder, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult, AddModelInstancesFromSdfString,
                         MathematicalProgram, Simulator, BasicVector, )
from underactuated import (FindResource, PlanarRigidBodyVisualizer)


timestep = 0.0

# tree = RigidBodyTree(FindResource("double_pendulum/double_pendulum.urdf"),
#                      FloatingBaseType.kFixed)
# tree = RigidBodyTree(FindResource("cubli/cubli.sdf"),
#                      FloatingBaseType.kFixed)
builder = DiagramBuilder()
tree = RigidBodyTree()
# AddModelInstancesFromSdfString(
#     open("underactuated/src/cubli/cubli.sdf", 'r').read(),
#     FloatingBaseType.kFixed,
#     None, tree)

# tree
tree = RigidBodyTree(FindResource("cubli/cubli.urdf"),
                     FloatingBaseType.kFixed)
# plant = RigidBodyPlant(tree, timestep)
plant = RigidBodyPlant(tree)
nx = tree.get_num_positions() + tree.get_num_velocities()

allmaterials = CompliantMaterial()
allmaterials.set_youngs_modulus(1E8) # default 1E9
allmaterials.set_dissipation(1.0) # default 0.32
allmaterials.set_friction(1.0) # default 0.9.
plant.set_default_compliant_material(allmaterials)

context = plant.CreateDefaultContext()

print(tree.get_num_positions())

# ETHAN

robot = builder.AddSystem(plant)
# builder.ExportInput(robot.get_input_port(0))

torque = 0.0
torque_system = builder.AddSystem(ConstantVectorSource(
    np.ones((tree.get_num_actuators(), 1))*torque))
builder.Connect(torque_system.get_output_port(0),
                robot.get_input_port(0))

# what do xlim and ylim mean, what about -5 for the ground element
vis = builder.AddSystem(PlanarRigidBodyVisualizer(tree, xlim=[-2.5, 2.5], ylim=[-1, 2.5]))
builder.Connect(robot.get_output_port(0),
                vis.get_input_port(0))

# And also log
signalLogRate = 60
signalLogger = builder.AddSystem(SignalLogger(nx))
signalLogger._DeclarePeriodicPublish(1. / signalLogRate, 0.0)
builder.Connect(robot.get_output_port(0),
                signalLogger.get_input_port(0))

diagram = builder.Build()

simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

context = simulator.get_mutable_context()
# context.FixInputPort(0, BasicVector([0., 0.]))  # Zero input torques
state = context.get_mutable_state().get_mutable_continuous_state().get_mutable_vector()
# state.SetFromVector((0., 0., 0., 0.))  # (theta1, theta2, theta1dot, theta2dot)
# state.SetFromVector((1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.))  # (theta1, theta2, theta1dot, theta2dot)
# state.SetFromVector((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.))
num_states = tree.get_num_positions() + tree.get_num_velocities()
print num_states

state.SetFromVector((-1.0,0.,0.5,0.,0.,0.,0.,1.0,0.,1.0,0.,0.,0.,0.))

integrator = simulator.get_mutable_integrator()
integrator.set_fixed_step_mode(True)
integrator.set_maximum_step_size(0.0005)

# state.SetFromVector((2,2,10,0,0,0,0,0,0,0))
simulator.StepTo(2)

print(state.CopyToVector())

ani = vis.animate(signalLogger, repeat=True)
plt.show()


#-------


# dircol = DirectCollocation(plant, context, num_time_samples=21,
#                            minimum_timestep=0.1, maximum_timestep=0.4)

# dircol.AddEqualTimeIntervalsConstraints()

# initial_state = (0., 0., 0., 0.)
# dircol.AddBoundingBoxConstraint(initial_state, initial_state,
#                                 dircol.initial_state())
# More elegant version is blocked on drake #8315:
# dircol.AddLinearConstraint(dircol.initial_state() == initial_state)

# final_state = (0., math.pi, 0., 0.)
# dircol.AddBoundingBoxConstraint(final_state, final_state,
#                                 dircol.final_state())
# dircol.AddLinearConstraint(dircol.final_state() == final_state)

# R = 10  # Cost on input "effort".
# u = dircol.input()
# dircol.AddRunningCost(R*u[0]**2)

# Add a final cost equal to the total duration.
# dircol.AddFinalCost(dircol.time())

# initial_x_trajectory = \
#     PiecewisePolynomial.FirstOrderHold([0., 4.],
#                                        np.column_stack((initial_state,
#                                                         final_state)))
# dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

# result = dircol.Solve()
# assert(result == SolutionResult.kSolutionFound)

# x_trajectory = dircol.ReconstructStateTrajectory()

# vis = PlanarRigidBodyVisualizer(tree, xlim=[-2.5, 2.5], ylim=[-1, 2.5])
# ani = vis.animate(x_trajectory, repeat=True)

# u_trajectory = dircol.ReconstructInputTrajectory()
# times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 100)
# u_lookup = np.vectorize(u_trajectory.value)
# u_values = u_lookup(times)

# plt.figure()
# plt.plot(times, u_values)
# plt.xlabel('time (seconds)')
# plt.ylabel('force (Newtons)')
#


# keep the window open
# plt.show()
