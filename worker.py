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
from .simplex_plotter_mixin import SimplexPlotterMixin
print ("WARNING: Currently relying on FiberSim repository outside of this repository. Refactor!!")
sys.path.append("../../Models/FiberSim/Python_files/")
from util import run, instruct, protocol

class Worker(SimplexPlotterMixin, ParameterManipulatorMixin, SimulationRunner):
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

    # Set up all of the class attributes for manipulating parameters.
    self.setup_parameters()

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
      self.write_best_model_file()
      
      # Call the callback for minimum error.
      if self.min_error_callback:
        self.min_error_callback()

    return this_error
  
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
