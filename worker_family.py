"""
Created on Tue Feb 11 16:24:00 2020

@author: Dylan Colli

Purpose: WorkerFamily class for fitting a family of simulations. Can be called by itself for simple 
         simplex optimization or driven by MasterWorker object for more complex particle swarm 
         optimization.
"""
import os
import shutil

import numpy as np

from .worker import Worker
from .simulation_runner import SimulationRunner
from .parameter_manipulator_mixin import ParameterManipulatorMixin


class WorkerFamily(ParameterManipulatorMixin):
  """Class for fitting a family of simulations"""
  def __init__(self, fibersim_file, options_file, original_model_file, working_model_file, 
    best_model_file, protocol_files, fit_mode, fit_variable, optimization_template_file, output_dir,
    target_data, time_steps_to_steady_state=2500, compute_rolling_average=False, 
    display_progress=False):
    """Initializes a WorkerFamily object
    
    Parameters
    ----------
    fibersim_file : str
        Path to the FiberSim executable.
    options_file : str
        Path to the options file.
    original_model_file : str
        Path to the original model file.
    working_model_file : str
        Path to the working model file. This is overwritten and thus should not be an existing file.
    best_model_file : str
        Path to where the best model file should be stored.
    protocol_files : list of str
        List of paths to the protocol files that will be used for this family of simulations.
    fit_mode : str
        The fit mode for this optimization job. One of ["time", "end_point"]
    fit_variable : str
        The variable that is fit in this optimization job. Must be "muscle_force" currently.
    optimization_template_file : str
        Path to the optimization template.
    output_dir : str
        Path to the output directory for this optimization job.
    target_data : str or numpy.ndarray
        Either the path to the target data for this simulation or the target data in a 
        numpy.ndarray.
        NOTE: With the WorkerFamily, this should be an array of just the points that each child is
        attempting to fit to. For example, if you are wishing to fit 10 points for this family of
        simulations, you must supply data that is SOLELY those points.
    time_steps_to_steady_state : int, optional
        The number of time steps it takes the model to reach steady state. This is ignored when
        `fit_mode` == "end_point".
    compute_rolling_average : bool, optional
        Whether to compute a rolling average to smooth out any stochasticity, by default False.
    display_progress : bool, optional
        Whether to display the progress of the optimization job via a matplotlib figure, by default
        False.
    """
    self.fibersim_file = fibersim_file
    self.options_file = options_file
    self.original_model_file = original_model_file
    self.working_model_file = working_model_file
    self.best_model_file = best_model_file 
    self.protocol_files = protocol_files
    self.fit_mode = fit_mode
    self.fit_variable = fit_variable
    self.optimization_template_file = optimization_template_file
    self.output_dir = output_dir
    self.target_data = target_data
    self.time_steps_to_steady_state = time_steps_to_steady_state
    self.compute_rolling_average = compute_rolling_average
    self.display_progress = display_progress
    
    self.children = []
    self.error_values = []
    self.iteration_number = 0
    self.best_error = np.inf

    self.setup_parameters()
    self.read_target_data()

    child_dict = {
      "fibersim_file":self.fibersim_file,
      "options_file":self.options_file,
      "model_file":self.working_model_file,
      "fit_mode":self.fit_mode,
      "fit_variable":self.fit_variable,
      "time_steps_to_steady_state":self.time_steps_to_steady_state,
      "compute_rolling_average":self.compute_rolling_average,
    }

    # Initialize a child for each protocol file.
    for i, prot_file in enumerate(self.protocol_files):
      # Create a new directory structure for each of the family's children.
      child_base_dir = os.path.join(self.output_dir, "child_"+str(i))
      child_dict["protocol_file"] = prot_file
      child_dict["output_dir"] = os.path.join(child_base_dir, "results")

      # Clean any previously made child directories and make a new one.
      if os.path.isdir(child_base_dir):
        shutil.rmtree(child_base_dir)
      os.makedirs(child_dict["output_dir"])

      # Give each child their respective singular point to fit in the target data.
      child_dict["target_data"] = self.target_data[i, 1]

      # Initialize the child.
      child_obj = SimulationRunner(**child_dict)
      self.children.append(child_obj)
      
  def read_target_data(self):
    """Sets up the target data for passing to children"""
    # We have to read in the data if it's not already an array.
    if isinstance(self.target_data, str):
      self.target_data = np.loadtxt(self.target_data)
    elif not isinstance(self.target_data, np.ndarray):
      raise TypeError("target_data must be either a string describing a path to the target data "
        "TXT  file or a numpy.ndarray!")
    
    if self.target_data.shape[0] != len(self.protocol_files) or self.target_data.shape[1] != 2:
      print (self.target_data.shape)
      raise ValueError("Target data must have shape == [len(protocol_files), 2]!")
    
  def fit(self, p_value_array):
    """Calls the fit function for each child of this family and returns collective error"""
    # Update the parameters and dump the new model file.
    self.p_values = p_value_array
    self.update_parameters()
    self.write_working_model_file()

    # Dump the information we're wanting to log.
    self.dump_param_information()
    self.record_extreme_p_values()

    # Create error storage for our family of simulations.
    error = 0

    for i, child in enumerate(self.children):
      print ("Running child #{}/{}".format(i + 1, len(self.children)))
      error += child.fit_worker(p_value_array)

    if error < self.best_error:
      self.best_error = error
      self.write_best_model_file()

    self.error_values.append(error)

    return error
