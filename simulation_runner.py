"""Holds the SimulationRunner class"""
import os
import sys
import subprocess

import numpy as np

print ("WARNING: Currently relying on FiberSim repository outside of this repository. Refactor!!")
sys.path.append("../../Models/FiberSim/Python_files/")
from util import run, instruct, protocol

class SimulationRunner:
  """Basic class for running simulations in "unintelligent" way"""
  def __init__(self, fibersim_file, options_file, model_file, protocol_file, fit_mode, 
    fit_variable, output_dir, target_data, time_steps_to_steady_state=2500, 
    compute_rolling_average=False):
    """Initializes a SimulationRunner object
    
    Parameters
    ----------
    fibersim_file : str
        The path to the FiberSim executable.
    options_file : str
        The path to the options file.
    model_file : str
        The path to the model file.
    protocol_file : str
        The path to the protocol file.
    fit_mode : str
        The mode with which this `SimulationRunner` is assessing its error. One of ["time", 
          "end_point"].
    fit_variable : str
        The variable with which the SimulationRunner should assess its error. Currently only valid
        option is "muscle_force".
    output_dir : str
        The path to the output directory for this simulation.
    target_data : str or numpy.ndarray
        Either the path to the target data for this simulation or a numpy.ndarray of the target 
        data.
    time_steps_to_steady_state : int, optional
        The number of time steps it takes for the simulation to reach steady state, by default 2500.
        These points are ignored in the evaluation of the `SimulationRunner`s error.
    compute_rolling_average : bool, optional
        Whether to compute a rolling average of the data to smooth out signal, by default False.
    """
    self.fibersim_file = fibersim_file
    self.options_file = options_file
    self.model_file = model_file
    self.protocol_file = protocol_file
    self.fit_mode = fit_mode
    self.fit_variable = fit_variable
    self.output_dir = output_dir
    self.target_data = target_data
    self.time_steps_to_steady_state = time_steps_to_steady_state
    self.compute_rolling_average = compute_rolling_average
    self.fit_data = None

    # Read in the simulation times from the protocol file. We'll be using these for interpolating
    # the target data.
    self.sim_times = protocol.get_sim_time(self.protocol_file)

    self.read_target_data()

  def read_target_data(self):
    """Reads in the objective function data and interpolates based on simulation times."""
    if isinstance(self.target_data, str):
      if self.fit_variable == "muscle_force":
        print ("Assuming 'muscle_force' data is a formatted TXT file.")
        self.target_data = np.loadtxt(self.target_data)

        # Get the time step for the simulation.
        delta_ts = [self.target_data[i+1, 0] - self.target_data[i, 0] for i in range(
          self.target_data.shape[0] - 1)]
        self.time_step = np.mean(delta_ts)

        # Do some error-checking.
        if not np.isclose(self.time_step, 0.001):
          # raise RuntimeError("Time step other than 1 millisecond not supported!")
          print ("WARNING: TIME STEPS OTHER THAN 1 MILLISECOND NOT SUPPORTED!")
      else:
        raise RuntimeError("Fit variable not recognized!!")
    
    if self.fit_mode == "time":
      # Linearly interpolate the target data according to the simulation time steps.
      times_to_interpolate = (np.asarray(self.sim_times[self.time_steps_to_steady_state:])
        - self.sim_times[self.time_steps_to_steady_state])
      interpolated_values = np.interp(times_to_interpolate, self.target_data[:, 0], 
        self.target_data[:, 1])
      
      # Concatenate these back together to form the newly interpolated target data and store it.
      self.target_data = np.stack((times_to_interpolate, interpolated_values), axis=-1)
    elif self.fit_mode != "end_point":
      raise RuntimeError("`Worker.fit_mode` not understood!")

  def run_simulation(self):
    """Runs the simulation that you are optimizing"""
    cmd = [self.fibersim_file, self.options_file, self.model_file, self.protocol_file, 
      self.output_dir]

    # Start the process.
    exit_code = subprocess.call(cmd)

    return exit_code

  def read_simulation_results(self):
    """Returns an averaged result value from the repeats ran by run_fibersim_simulation()."""      
    # Check which optimization mode we're using to point to the right file.
    if self.fit_variable == "muscle_force":
      file_str = "forces.txt"
    else:
      raise RuntimeError("Can't optimize anything but 'muscle_force'!!!")
    
    full_path = os.path.join(self.output_dir, file_str)

    # Read in the appropriate results file.
    if self.fit_variable == "muscle_force":
      # Just pull out the total muscle force from the forces file. This is in the last column
      # of the file.
      self.fit_data = np.loadtxt(full_path, skiprows=1)[:, -1]

    # Do the rolling average if we so choose.
    if self.compute_rolling_average:
      rolling_window_size = 5
      kernel = np.ones(rolling_window_size) / rolling_window_size
      self.fit_data = np.convolve(self.fit_data, kernel, 'valid')
