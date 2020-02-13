# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:20:00 2019

@author: Dylan Colli

Purpose: Worker class for Optimization. Can be called by itself for simple simplex optimization or
         driven by MasterWorker object for more complex particle swarm optimization.
"""

import os
import sys
import json
import types

import numpy as np
import matplotlib.pyplot as plt

from . import params
from .simulation_runner import SimulationRunner
from .parameter_manipulator_mixin import ParameterManipulatorMixin
print ("WARNING: Currently relying on FiberSim repository outside of this repository. Refactor!!")
sys.path.append("../../Models/FiberSim/Python_files/")
from util import run, instruct, protocol

class Worker(ParameterManipulatorMixin, SimulationRunner):
  def __init__(self,
               original_model_file,
               working_model_file,
               best_model_file,
               optimization_template_file,
               display_progress=True,
               optimization_figure=None,
               min_error_callback=None,
               *args, **kwargs):
    """Initializes a Worker object
    
    Parameters
    ----------
    original_model_file : str
        The file path to the original JSON model file.
    working_model_file: str
        The file path to the model file that is used to inform the current simulation.
    best_model_file : str
        The file path to where the best model file will be saved.
    optimization_template_file : str
        The file path to the optimization template, where the initial p-values are stored.
    display_progress : bool, optional
        Whether to display the progress using multiple matplotlib plots, by default True.
    optimization_figure : matplotlib.figure, optional
        The figure that will be updated, by default None and will be initialized with 
        `initialize_figure()`
    min_error_callback : MethodType, optional
        The callback that is called when a new minimum error is reached. By default None, such that
        nothing happens upon reaching a new minimum error. If a function is specified, must be in
        the MethodType syntax. The function is bound to the object during initialization.
    """
    super().__init__(model_file=working_model_file, *args, **kwargs)
    self.original_model_file = original_model_file
    self.working_model_file = working_model_file
    self.best_model_file = best_model_file
    self.optimization_template_file = optimization_template_file
    self.display_progress = display_progress
    self.fig = optimization_figure

    # Graft the minimum error callback method onto this instance of Worker.
    self.min_error_callback = min_error_callback
    if self.min_error_callback:
      self.min_error_callback = types.MethodType(self.min_error_callback, self)

    print ("WARNING: Assuming a fit mode of only time")
    print ("WARNING: Assuming fit variable of only 'muslce_force' now.")

    # Initialize variable for keeping track of iterations.
    self.iteration_number = 0

    # Initialize the empty parameter interpolation lists.
    self.p_objs = []

    # Initialize variables for holding error.
    self.best_error = np.inf
    self.error_values = []

    # Initialize the dictionary structures for JSON parsing.
    self.options_dict = {}
    self.optimization_template_dict = {}
    self.model_dict = {}

    # Read in the options file to set the number of repeats.
    self.read_options_file()

    # Read in the original JSON model file.
    self.read_original_model_file()

    # Read in the optimization structure.
    self.read_optimization_structure()

    # Form the initial p_value array and p-value history.
    self.p_values = np.asarray([obj.p_value for obj in self.p_objs])
    self.p_value_history = [[obj.p_value] for obj in self.p_objs]

    # Make the plots for optimization bookkeeping.
    if self.display_progress: self.initialize_optimization_figure()

    # Delete the previous dumps if they're present.
    files_to_delete = ["parameter_history.txt", "p_history.txt", "errors.txt"]
    for file_name in files_to_delete:
      file_name = os.path.join(self.output_dir, file_name)
      if os.path.isfile(file_name):
        os.remove(file_name)

  def fit_worker(self, p_values=np.asarray([])):
    """Runs a single trial for this worker and returns error compared to target data."""
    if p_values.size == 0:
      raise RuntimeError("fit_worker function must be called by optimizer with initial"
        +"p_values!!")

    # Update the p_values for this iteration of the optimization function.
    self.p_values = p_values

    # Update the worker's parameters.
    self.update_parameters()

    # Write some log information.
    self.record_extreme_p_values()

    # Write the new parameters into the working JSON model file.
    self.write_working_model_file()

    # Run this FiberSim simulation.
    exit_code = self.run_simulation()

    # Read results from this simulation.
    self.read_simulation_results()

    # Compare the results to objective function.
    this_error = self.get_simulation_error()

    if this_error < self.best_error:
      self.best_error = this_error

      # Write the working model file as the best file.
      with open(self.best_model_file, 'w') as f:
        json.dump(self.model_dict, f, indent=2)
      
      # Call the callback for minimum error.
      if self.min_error_callback:
        self.min_error_callback()

    return this_error

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
  
  def update_worker(self, p_values):
    """Updates the worker after every optimization iteration. NOT function evaluation."""
    # Keep track of things for this iteration.
    self.iteration_number += 1
    this_error = self.get_simulation_error()
    self.error_values.append(this_error)

    # Dump the error information.
    with open(os.path.join(self.output_dir, "errors.txt"), 'a+') as f:
      f.write(str(this_error)+'\n')

    for i in range(self.p_values.shape[0]):
      self.p_value_history[i].append(self.p_values[i])

    self.dump_param_information()

    if self.display_progress: self.update_plots()
