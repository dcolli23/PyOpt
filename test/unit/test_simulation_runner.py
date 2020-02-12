"""Tests the SimulationRunner implementation"""
import os
import sys
import copy
import shutil

import pytest
import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt.simulation_runner import SimulationRunner

sys.path.append("../../Models/FiberSim/Python_files/")
from util import protocol

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)

SIM_RUNNER_DICT = {
  "fibersim_file":None, 
  "options_file":os.path.join(TEST_DATA_ROOT, "sim_options.json"), 
  "model_file":os.path.join(TEST_DATA_ROOT, "model.json"), 
  "protocol_file":os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"), 
  "fit_mode":"time", 
  "fit_variable":"muscle_force", 
  "output_dir":TEST_RESULT_DIR,
  "target_data":os.path.join(TEST_DATA_ROOT, "target_data_muscle_force.txt"), 
  "time_steps_to_steady_state":0, 
  "compute_rolling_average":False
} 

def test_simulation_runner_initialization():
  sr = SimulationRunner(**SIM_RUNNER_DICT)

  assert (sr), "SimulationRunner did not initialize correctly!"

def test_get_simulation_times():
  sr = SimulationRunner(**SIM_RUNNER_DICT)

  truth_times = protocol.get_sim_time(SIM_RUNNER_DICT["protocol_file"])

  assert (sr.sim_times == truth_times), "SimulationRunner did not read protocol times correctly!"

def test_read_target_data_muscle_force_time_string():
  sr = SimulationRunner(**SIM_RUNNER_DICT)

  target_data_truth = np.loadtxt(SIM_RUNNER_DICT["target_data"])
  sim_times = protocol.get_sim_time(SIM_RUNNER_DICT["protocol_file"])

  # Linearly interpolate the target data according to the simulation time steps.
  times_to_interpolate = (np.asarray(sim_times[SIM_RUNNER_DICT["time_steps_to_steady_state"]:])
    - sim_times[SIM_RUNNER_DICT["time_steps_to_steady_state"]])
  interpolated_values = np.interp(times_to_interpolate, target_data_truth[:, 0], 
    target_data_truth[:, 1])
  
  # Concatenate these back together to form the newly interpolated target data and store it.
  target_data_truth = np.stack((times_to_interpolate, interpolated_values), axis=-1)

  assert (np.array_equal(sr.target_data, target_data_truth)), ("SimulationRunner did not read "
    "target data correctly!")
  
def test_read_target_data_muscle_force_time_numpy_array():
  target_data_orig = np.loadtxt(SIM_RUNNER_DICT["target_data"])
  this_sr_dict = copy.copy(SIM_RUNNER_DICT)
  this_sr_dict["target_data"] = target_data_orig

  sim_times = protocol.get_sim_time(SIM_RUNNER_DICT["protocol_file"])

  # Linearly interpolate the target data according to the simulation time steps.
  times_to_interpolate = (np.asarray(sim_times[SIM_RUNNER_DICT["time_steps_to_steady_state"]:])
    - sim_times[SIM_RUNNER_DICT["time_steps_to_steady_state"]])
  interpolated_values = np.interp(times_to_interpolate, target_data_orig[:, 0], 
    target_data_orig[:, 1])
  
  # Concatenate these back together to form the newly interpolated target data and store it.
  target_data_truth = np.stack((times_to_interpolate, interpolated_values), axis=-1)

  sr = SimulationRunner(**SIM_RUNNER_DICT)

  assert (np.array_equal(sr.target_data, target_data_truth)), ("SimulationRunner did not read "
    "target data correctly!")
  
# def test_read_target_data_muscle_force_end_point_string():
#   this_sr_dict = copy.copy(SIM_RUNNER_DICT)
#   this_sr_dict["fit_mode"] = "end_point"
#   target_data_truth = np.loadtxt(SIM_RUNNER_DICT["target_data"])
#   target_data_truth = np.mean(target_data_truth[-10])

# def test_read_target_data_muscle_force_end_point_numpy_array():
#   this_sr_dict = copy.copy(SIM_RUNNER_DICT)
#   this_sr_dict["fit_mode"] = "end_point"

def test_read_simulation_results_force():
  force_file = os.path.join(TEST_DATA_ROOT, "forces.txt")
  results_truth = np.loadtxt(force_file, skiprows=1)[:, -1]
  this_sr_dict = copy.copy(SIM_RUNNER_DICT)
  this_sr_dict["output_dir"] = TEST_DATA_ROOT
  sr = SimulationRunner(**this_sr_dict)

  sr.read_simulation_results()

  assert (np.array_equal(sr.fit_data, results_truth)), ("SimulationRunner did not read in "
    "simulation results correctly!")

# def test_run_simulation():
#   """This uses the `echo` windows command to test that the simulation runs correctly"""
#   this_sr_dict = copy.copy(SIM_RUNNER_DICT)
#   this_sr_dict["fibersim_file"] = "echo"
#   this_sr_dict["options_file"] = "Hello,"
#   this_sr_dict["model_file"] = "World!"
#   # this_sr_dict["protocol_file"] = ">"
#   # this_sr_dict["output_dir"] = os.path.realpath(os.path.join(TEST_RESULT_DIR, "sim_run_test.txt"))
#   sr = SimulationRunner(**this_sr_dict)

#   sr.run_simulation()

#   # Read the newly created file.
#   with open(this_sr_dict["output_dir"], 'r') as f:
#     text = f.read()

#   assert (text == "Hello, World!"), "SimulationRunner did not run simulation correctly!"