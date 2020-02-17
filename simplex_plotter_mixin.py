# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:53:00 2020

@author: Dylan Colli

Purpose: Class for plotting simplex optimization progress.
"""

import matplotlib.pyplot as plt
import numpy as np

class SimplexPlotterMixin:
  """Mixin for plotting progress of simplex optimization"""
  def initialize_optimization_figure(self):
    """Initializes the figure and starts plots for optimization plotting"""
    self.initialize_figure()
    self.make_plots()

  def initialize_figure(self):
    """Initializes the figure used to show optimization progress"""
    if not self.fig:
      self.fig = plt.figure(figsize=[9, 5])
    plt.ion()

  def make_plots(self):
    """Makes interactive plots for the convergence and best fit."""
    self.fig.set_tight_layout(True)

    self.convergence_ax = self.fig.add_subplot(131)
    self.convergence_plot, = self.convergence_ax.plot(self.error_values)
    self.convergence_ax.set_title("Error vs Optimization Iteration")
    self.convergence_ax.set_xlabel("Iteration Number")
    self.convergence_ax.set_ylabel("Sum of Squared Errors")
    self.convergence_ax.set_yscale("log")

    self.fit_ax = self.fig.add_subplot(132)
    self.fit_ax.plot(self.target_data[:,0], self.target_data[:,1:], label="Target")
    self.fit_plot, = self.fit_ax.plot(self.target_data[:,0], np.ones_like(self.target_data[:,0] * 0),
      label="Model")
    self.fit_ax.set_title("Current Fit")
    self.fit_ax.legend()

    self.p_ax = self.fig.add_subplot(133)
    self.p_plots = []
    for i in range(len(self.p_values)):
      self.p_plots.append(self.p_ax.plot([0], self.p_values[i], label=self.p_objs[i].p_lookup[-1])[0])
    self.p_ax.set_title("P-values v Optimization Iteration")
    self.p_ax.legend()
    self.p_ax.set_ylim([0, 1])

    # Set the tight layout to update each time the figure is drawn. This is necessary so the axis
    # labels don't overlap.
    self.fig.set_tight_layout(True)

    plt.show()

    # draw the data on the plot
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

  def update_plots(self):
    """Updates the plots made with self.make_plots()"""
    # Update the fit plot
    self.fit_plot.set_ydata(self.fit_data[self.time_steps_to_steady_state:])
    new_upper_y_lim = 1.05 * np.max((self.target_data[:,1], 
      self.fit_data[self.time_steps_to_steady_state:]))
    new_lower_y_lim = 0.95 * np.min((self.target_data[:,1],
      self.fit_data[self.time_steps_to_steady_state:]))
    self.fit_ax.set_ylim(bottom=new_lower_y_lim, top=new_upper_y_lim)

    # Update the error plot.
    self.convergence_plot.set_xdata(range(1, len(self.error_values) + 1))
    self.convergence_plot.set_ydata(self.error_values)
    self.convergence_ax.set_xlim(right=len(self.error_values))
    self.convergence_ax.set_ylim(bottom=0.95*np.min(self.error_values),
      top=1.05 * np.max(self.error_values))

    # Update the p-value plot
    min_p = 1e25
    max_p = -1e25
    for i in range(self.p_values.shape[0]):
      min_p = min(self.p_value_history[i] + [min_p])
      max_p = max(self.p_value_history[i] + [max_p])
      self.p_plots[i].set_xdata(range(0,self.iteration_number + 1))
      self.p_plots[i].set_ydata(self.p_value_history[i])
    self.p_ax.set_xlim(right=self.iteration_number)

    p_range = max_p - min_p
    new_y_bottom = min_p - 0.1 * p_range
    new_y_top = max_p + 0.1 * p_range
    self.p_ax.set_ylim(bottom=new_y_bottom, top=new_y_top)

    # draw the new data on the plot
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()