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
    """Initializes a MasterWorker object
    
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

  # def read_options_file(self):
  #   """Reads options file into class dictionary."""
  #   with open(self.options_file, 'r') as f:
  #     self.options_dict = json.load(f)

  # def update_parameters(self):
  #   """Updates the parameters for this worker from the p_vals array."""
  #   for i in range(self.p_values.shape[0]):
  #     # Update the parameter p_value from the array.
  #     self.p_objs[i].p_value = self.p_values[i]

  #     # Calculate the new parameter value.
  #     new_param_val = self.p_objs[i].calculate_parameter()

  #     # Set the new parameter value.
  #     self.recurs_update_parameter(self.p_objs[i].p_lookup.copy(), new_param_val)
  
  # def record_extreme_p_values(self):
  #   """Writes to WARNINGS.log in output directory when extreme p_values are encountered."""
  #   idxs_greater = [i for i, p_value in enumerate(self.p_values) if p_value > 0.95]
  #   idxs_lesser = [i for i, p_value in enumerate(self.p_values) if p_value < 0.05]
  #   if idxs_greater:
  #     # Get the names of the parameters that are too high.
  #     with open(os.path.join(self.output_dir, "WARNINGS.log"), 'a+') as f:
  #       str_to_write = ""
  #       #for obj in self.p_objs[idxs_greater]:
  #       for idx in idxs_greater:
  #         obj = self.p_objs[idx]
  #         str_to_write += "WARNING: P value for parameter \"{}\" > 0.95; iteration {}!\n".format(
  #           obj.p_lookup[-1], self.iteration_number)
  #         str_to_write += "\tvalue = {}\n".format(obj.p_value)
  #       f.write(str_to_write)
  #   if idxs_lesser:
  #     # Get the names of the parameters that are too low.
  #     with open(os.path.join(self.output_dir, "WARNINGS.log"), 'a+') as f:
  #       str_to_write = ""
  #       # for obj in self.p_objs[idxs_lesser]:
  #       for idx in idxs_lesser:
  #         obj = self.p_objs[idx]
  #         str_to_write += "WARNING: P value for parameter \"{}\" < 0.05; iteration {}!\n".format(
  #           obj.p_lookup[-1], self.iteration_number, ) 
  #         str_to_write += "\tvalue = {}\n".format(obj.p_value) 
  #       f.write(str_to_write)

  # def recurs_update_parameter(self, p_lookup, new_param_val, traversed_model_dict=None):
  #   """Updates the parameter values for this worker in the model dictionary."""
  #   # Get the new key for this node.
  #   new_k = p_lookup.pop(0)

  #   # Check to see if we're at the end of the parameter lookup path.
  #   if len(p_lookup) > 0:
  #     # Traverse the dictionary by one node.
  #     if not traversed_model_dict:
  #       traversed_model_dict = self.model_dict
  #     traversed_model_dict = traversed_model_dict[new_k]
      
  #     # Call the recursive function.
  #     self.recurs_update_parameter(p_lookup, new_param_val, traversed_model_dict)
    
  #   else:
  #     # Check to see if this is a rate parameter to set.
  #     if '@' in new_k:
  #       self.set_rate_param(new_k, new_param_val, traversed_model_dict)
  #     else:
  #       self.set_regular_param(new_k, new_param_val, traversed_model_dict)

  # def set_rate_param(self, rate_key, new_value, traversed_model_dict):
  #   """Sets the rate parameter that has been found by recurs_update_parameter."""
  #   instruct.set_rate_param(rate_key, new_value, traversed_model_dict)

  # def set_regular_param(self, param_key, new_value, traversed_model_dict):
  #   """Sets the parameter in the model dictionary to the new value."""
  #   traversed_model_dict[param_key] = new_value

  # def read_optimization_structure(self):
  #   """Reads the optimization structure into a dictionary.
    
  #   Important note: Python 3.7 dictionaries now preserve insertion order so we can be sure that
  #   the order of this dictionary will stay the same across the simulation. If we were using 
  #   Python < 3.7, this would not be the case and we'd have to use an ordereddict.
  #   """
  #   # Read the optimization structure into the dictionary object.
  #   with open(self.optimization_json_template_string, 'r') as f:
  #     self.optimization_template_dict = json.load(f)
    
  #   # Set the initial p_values by recursively searching the dictionary.
  #   self.recurs_read_param(this_dict=self.optimization_template_dict)

  # def recurs_read_param(self, key=None, this_dict=None, param_path=[]):
  #   """Recursively traverses the optimization dictionary structure to set p-values."""
  #   traverse = False
  #   if not key:
  #     # We know this is the first function call in the recursive call and we need to traverse
  #     # the dictionary.
  #     traverse = True

  #   elif isinstance(this_dict[key], dict):
  #     # Add this key as a node to the parameter path.
  #     param_path.append(key)

  #     # We know this is a dictionary, check if it's an optimization dictionary with p-values.
  #     if "p_value" not in this_dict[key].keys():
  #       traverse = True
  #       this_dict = this_dict[key]
      
  #     # Otherwise, we can set the p_value here.
  #     else:
  #       # Check to make sure the optimization template has been specified correctly.
  #       p_mode = this_dict[key]["p_mode"]
  #       p_min = this_dict[key]["min_value"]
  #       p_max = this_dict[key]["max_value"]
  #       p_value = this_dict[key]["p_value"]

  #       assert (p_mode == "lin" or p_mode == "log"), (
  #         "p_mode for parameter \"{}\" in optimization template must be \"lin\" or \"log\"!".format(
  #           key))
  #       assert (isinstance(p_value, (int, float))), (
  #         "p_value for parameter \"{}\" in optimization template must be a number!".format(key))
  #       assert (isinstance(p_min, (int, float))), (
  #         "min_value for parameter \"{}\" in optimization template must be a number!".format(key))
  #       assert (isinstance(p_max, (int, float))), (
  #         "max_value for parameter \"{}\" in optimization template must be a number!".format(key))

  #       # Store the values for this parameter.
  #       self.p_objs.append( params.ParamObject(p_min, p_max, p_value, p_mode, param_path) )
    
  #   if traverse:
  #     for sub_key in this_dict.keys():
  #       self.recurs_read_param(key=sub_key, this_dict=this_dict, param_path=param_path.copy())

  # def write_working_model_file(self):
  #   """Writes the working JSON model file into the class."""
  #   with open(self.working_json_model_file_string, 'w') as f:
  #     json.dump(self.model_dict, f, indent=2)

  # def read_original_model_file(self):
  #   """Reads the original JSON model file into the class."""
  #   with open(self.original_json_model_file_string, 'r') as f:
  #     self.model_dict = json.load(f)

  # def dump_param_information(self):
  #   """Dumps the parameter information for this iteration of the optimizer."""
  #   # Get the file name.
  #   file_name = os.path.join(self.output_dir, "parameter_history.txt")

  #   # If the file hasn't been written yet, open it and write the headers.
  #   if not os.path.isfile(file_name):
  #     # Open the file for writing.
  #     f = open(file_name, 'w')

  #     # Get the parameter names.
  #     param_names = [obj.p_lookup[-1] for obj in self.p_objs]

  #     # Write the headers.
  #     str_to_write = '\t'.join(param_names) + '\n'

  #   else:
  #     # Open the file for appending
  #     f = open(file_name, 'a')
  #     str_to_write = ""
    
  #   # Get the parameter values.
  #   param_values = [str(obj.calculated_value) for obj in self.p_objs]

  #   # Put the information in the string to write.
  #   str_to_write += '\t'.join(param_values) + '\n'

  #   # Write the information for the parameters.
  #   f.write(str_to_write)

  #   # Tidy up.
  #   f.close()

  #   # Do the same thing for p_value history.
  #   file_name = os.path.join(self.output_dir, "p_history.txt")
  #   if not os.path.isfile(file_name):
  #     f = open(file_name, 'w')
  #     param_names = [obj.p_lookup[-1] for obj in self.p_objs]
  #     str_to_write = '\t'.join(param_names) + '\n'
  #   else:
  #     f = open(file_name, 'a')
  #     str_to_write = ""
    
  #   # Get the p values.
  #   p_values = [str(obj.p_value) for obj in self.p_objs]

  #   # Put the information in the string to write.
  #   str_to_write += '\t'.join(p_values) + '\n'

  #   # Write the information for the parameters.
  #   f.write(str_to_write)

  #   # Tidy up.
  #   f.close()

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
