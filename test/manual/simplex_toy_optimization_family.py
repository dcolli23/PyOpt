"""This tests the simplex optimization for a family of solutions, using the WorkerFamily class

The functionality of the WorkerFamily class should be (and is at this point) rigorously tested with 
automated unit tests. However, a sanity check using optimization on a toy problem can be reassuring.
"""
import os
import sys
import copy
import time

import numpy as np
from scipy import optimize

ROOT = os.path.realpath(os.path.dirname(__file__))
PyOpt_ROOT = os.path.join(ROOT, "..", "..", "..")
sys.path.append(PyOpt_ROOT)
from PyOpt import WorkerFamily
from PyOpt.params import ParamObject
from PyOpt import SimulationRunner

FIBERSIM_ROOT = os.path.join(PyOpt_ROOT, "..", "Models", "FiberSim", "Python_files")
sys.path.append(FIBERSIM_ROOT)
from util import protocol

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "manual")
if not os.path.isdir(TEST_RESULT_DIR):
  os.makedirs(TEST_RESULT_DIR)

FIBERSIM_EXE = None
SIM_OPTIONS = None
ORIGINAL_MODEL = os.path.join(TEST_DATA_ROOT, "original_model.json")
WORKING_MODEL = os.path.join(TEST_RESULT_DIR, "working_model.json")
BEST_MODEL_FILE = os.path.join(TEST_RESULT_DIR, "best_model.json")
FIT_MODE = "end_point"
FIT_VARIABLE = "muscle_force"
OPTIMIZATION_TEMPLATE_FILE = os.path.join(TEST_RESULT_DIR, "optimization_template.json")
TARGET_DATA_FILE = os.path.join(TEST_RESULT_DIR, "fake_target_data.txt")
TIME_STEPS_TO_STEADY_STATE = 0
NUM_CHILDREN = 8

def main():
  protocol_files = [None for i in range(NUM_CHILDREN)]
  child_idxs = [i for i in range(NUM_CHILDREN)]
  target_data = np.asarray(list(zip(copy.copy(child_idxs), copy.copy(child_idxs))), dtype=np.float)

  # Write the methods that we want to manipulate.
  WorkerFamily.setup_parameters = new_setup_parameters
  WorkerFamily.update_parameters = new_update_parameters
  WorkerFamily.write_working_model_file = lambda x: None
  protocol.get_sim_time = lambda x: None
  SimulationRunner.run_simulation = new_run_simulation
  SimulationRunner.read_simulation_results = lambda x: None
  WorkerFamily.write_best_model_file = lambda x: None
  
  wf = WorkerFamily(FIBERSIM_EXE, SIM_OPTIONS, ORIGINAL_MODEL, WORKING_MODEL, BEST_MODEL_FILE, 
    protocol_files, FIT_MODE, FIT_VARIABLE, OPTIMIZATION_TEMPLATE_FILE, TEST_RESULT_DIR, 
    target_data, TIME_STEPS_TO_STEADY_STATE, display_progress=True)

  # Mark all of the children with their index.
  for i in range(len(wf.children)):
    wf.children[i].index = i

  optimize.minimize(wf.fit, wf.p_values, method="Nelder-Mead", callback=wf.update_family)

####################################################################################################
### Methods we're overwriting.
####################################################################################################
def new_setup_parameters(self):
    self.p_objs = []

    for i in range(NUM_CHILDREN):
      self.p_objs.append(ParamObject(-1, 10, 0.5, "lin", [str(i)]))
    
    self.p_values = []
    self.p_value_history = []
    for obj in self.p_objs:
      p_value = obj.p_value
      self.p_values.append(p_value)
      self.p_value_history.append([p_value])

def new_run_simulation(self):
  self.fit_data = self.calculated_value

def new_update_parameters(self):
  for i in range(len(self.children)):
     # Update the parameter p_value from the array.
    self.p_objs[i].p_value = self.p_values[i]

    # Calculate the new parameter value.
    new_param_val = self.p_objs[i].calculate_parameter()

    self.children[i].calculated_value = new_param_val

if __name__ == "__main__":
  main()