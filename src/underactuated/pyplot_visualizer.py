# -*- coding: utf8 -*-

import numpy as np
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pydrake.all import (LeafSystem, PiecewisePolynomial, SignalLogger,
                         VectorSystem)


class PyPlotVisualizer(LeafSystem):
    '''
        Base class from planar visualization
        that relies on pyplot.

        In the configuration set up here,
        this visualizer provides one visualization
        window (self.fig) with axes (self.ax).

        Subclasses must:
        - During initialization, set up the figure
        bounds and register and input port
        with the appropriate message type.
        - Override the draw method to parse the
        input and draw the robot in the appropriate
        state.
    '''

    def __init__(self, draw_timestep=0.033333, facecolor=[1, 1, 1],
                 figsize=None):
        LeafSystem.__init__(self)

        self.set_name('pyplot_visualization')
        self.timestep = draw_timestep
        self._DeclarePeriodicPublish(draw_timestep, 0.0)

        (self.fig, self.ax) = plt.subplots(facecolor=facecolor,
                                           figsize=figsize)
        self.ax.axis('equal')
        self.ax.axis('off')
        self.fig.show()

    def _DoPublish(self, context, event):
        self.draw(context)
        self.fig.canvas.draw()
        plt.pause(1e-10)

    def draw(self, context):
        print "SUBCLASSES MUST IMPLEMENT."

    def animate(self, log, resample=True, repeat=False):
        # log - a reference to a pydrake.systems.primitives.SignalLogger that
        # contains the plant state after running a simulation.
        # resample -- should we do a resampling operation to make
        # the samples more consistent in time? This can be disabled
        # if you know the draw_timestep passed into the constructor exactly
        # matches the sample timestep of the log.
        # repeat - should the resulting animation repeat?

        if type(log) is SignalLogger:
            t = log.sample_times()
            x = log.data()

            if resample:
                import scipy.interpolate

                t_resample = np.arange(0, t[-1], self.timestep)
                x = scipy.interpolate.interp1d(t, x, kind='linear', axis=1)(t_resample)  # noqa
                t = t_resample

        # TODO(russt): Replace PiecewisePolynomial with Trajectory if I ever
        # add the pydrake bindings for the base class.
        elif type(log) is PiecewisePolynomial:
            t = np.arange(log.start_time(), log.end_time(), self.timestep)
            x = np.hstack([log.value(time) for time in t])

        def animate_update(i):
            self.draw(x[:, i])

        ani = animation.FuncAnimation(self.fig,
                                      animate_update,
                                      t.shape[0],
                                      interval=1000*self.timestep,
                                      repeat=repeat)
        return ani


class SliderSystem(VectorSystem):
    def __init__(self, ax, title, min, max):
        # 0 inputs, 1 output.
        VectorSystem.__init__(self, 0, 1)
        self.value = 0
        self.slider = Slider(ax, title, min, max, valinit=self.value)
        self.slider.on_changed(self.update)

    def update(self, val):
        self.value = val

    def _DoCalcVectorOutput(self, context, unused, unused2, output):
        output[:] = self.value
