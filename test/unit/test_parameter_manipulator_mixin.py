"""Tests the ParameterManipulatorMixin initialization"""
import os
import sys
import copy
import shutil
import json

import pytest
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(PYOPT_ROOT)
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt.parameter_manipulator_mixin import ParameterManipulatorMixin
from PyOpt.worker import Worker
from params import ParamObject

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)

PARAM_DICT = {
  "fibersim_file": None,
  "protocol_file": os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
  "options_file": os.path.join(TEST_DATA_ROOT, "sim_options.json"),
  "fit_mode": "time",
  "fit_variable": "muscle_force",
  "original_model_file": os.path.join(TEST_DATA_ROOT, "original_model.json"),
  "working_model_file": os.path.join(TEST_RESULT_DIR, "working_model.json"),
  "best_model_file": os.path.join(TEST_RESULT_DIR, "best_model.json"),
  "optimization_template_file": os.path.join(TEST_DATA_ROOT, "optimization_template.json"),
  "output_dir": TEST_RESULT_DIR,
  "target_data": os.path.join(TEST_DATA_ROOT, "target_data_muscle_force.txt"),
  "time_steps_to_steady_state":1,
  "compute_rolling_average":False,
  "display_progress":False
}

def test_mixin_read_options_file():
  # Read in the options file.
  with open(PARAM_DICT["options_file"], 'r') as f:
    options_truth = json.load(f)
  
  # Initialize the mixin.
  pmm = ParameterManipulatorMixin()
  pmm.options_file = PARAM_DICT["options_file"]
  pmm.read_options_file()

  # This is a shallow equivalence check but it will do for now since the options dictionary is flat.
  assert (pmm.options_dict == options_truth), ("ParameterManipulationMixin did not read options "
    "file correctly!")

def test_set_regular_param():
  test = {"a":1, "b":2}
  pmm = ParameterManipulatorMixin()
  pmm.set_regular_param("a", 2, test)

  assert (test["a"] == 2), "ParameterManipulatorMixin did not set regular param correctly!"

def test_set_rate_param_mixin():
  test = [
    "1  gaussian  a - 33.958  0.0    0.0   0.0   -1.0   0.0    0.0",
    "2  energy    d - 1.38    1.757  2.81  0.639  6.763 2.87   0.1",
    "3  sig2      n - 5.88   20.96 -1.04  0.706  0.0   0.0    0.0",
  ]

  pmm = ParameterManipulatorMixin()
  pmm.set_rate_param("1@3", 300, test)
  pmm.set_rate_param("2@7", 0, test)
  pmm.set_rate_param("3@1", 217, test)

  for i, rate_eqn in enumerate(test):
    test[i] = rate_eqn.split()
  
  assert (test[0][6] == '300'), "ParameterManipulatorMixin did not set rate param correctly!"
  assert (test[1][10] == '0'), "ParameterManipulatorMixin did not set rate param correctly!"
  assert (test[2][4] == '217'), "ParameterManipulatorMixin did not set rate param correctly!"

def test_write_working_model_file():
  truth = {"a":1}

  pmm = ParameterManipulatorMixin()
  pmm.working_model_file = PARAM_DICT["working_model_file"]
  pmm.model_dict = truth
  pmm.write_working_model_file()

  with open(pmm.working_model_file, 'r') as f:
    test = json.load(f)

  assert (test == truth), "ParameterManipulatorMixin did not write working model file correctly!"

def test_read_original_model_file():
  with open(PARAM_DICT["original_model_file"], 'r') as f:
    truth = json.load(f)
  
  pmm = ParameterManipulatorMixin()
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.read_original_model_file()

  # Again, this is a shallow test of equivalence but this is fine for right now.
  assert (pmm.model_dict == truth), ("ParameterManipulatorMixin did not read original model file "
    "correctly!")

def test_read_optimization_structure():
  pmm = ParameterManipulatorMixin()
  pmm.p_objs = []
  pmm.optimization_template_file = PARAM_DICT["optimization_template_file"]
  pmm.read_optimization_structure()

  obj_1_test = pmm.p_objs[0]
  obj_2_test = pmm.p_objs[1]

  obj_1_truth = ParamObject(600, 700, 0.85, "lin", ["kinetics", "rate_equations", "3@1"])
  obj_2_truth = ParamObject(50, 150, 0.15, "lin", ["kinetics", "rate_equations", "5@1"])

  obj_test = [obj_1_test, obj_2_test]
  obj_truth = [obj_1_truth, obj_2_truth]
  for i, ot in enumerate(obj_test):
    for key, value in vars(ot).items():
      test = value
      truth = vars(obj_truth[i])[key]
      assert (test == truth), ("ParameterManipulatorMixin did not set ParamObject correctly!")

def test_update_parameters():
  pmm = ParameterManipulatorMixin()
  pmm.p_values = np.asarray([0.5, 0.5])
  pmm.p_objs = []
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.read_original_model_file()
  pmm.optimization_template_file = PARAM_DICT["optimization_template_file"]
  pmm.read_optimization_structure()

  param1_truth = 650
  param2_truth = 100

  pmm.update_parameters()

  assert (pmm.p_objs[0].calculated_value == param1_truth), ("ParameterManipulationMixin did not "
    "update parameter correctly!")
  assert (pmm.p_objs[1].calculated_value == param2_truth), ("ParameterManipulationMixin did not "
    "update parameter correctly!")



