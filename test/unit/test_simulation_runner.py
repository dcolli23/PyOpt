"""Tests the SimulationRunner implementation"""
import os
import sys
import copy
import shutil

import pytest
import matplotlib.pyplot as plt

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt.simulation_runner import SimulationRunner

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
  "target_data":os.path.join(TEST_DATA_ROOT, "forces.txt"), 
  "time_steps_to_steady_state":2500, 
  "compute_rolling_average":False
}

def test_simulation_runner_initialization():
  sr = SimulationRunner(**SIM_RUNNER_DICT)

  assert (sr), "SimulationRunner did not initialize correctly!"