"""Tests the ParameterManipulatorMixin initialization"""
import os
import sys
import copy
import shutil
import json

import pytest
import matplotlib.pyplot as plt
import numpy as np
import jgrapht

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
  elif os.path.isfile(path):
    os.remove(path)

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

CALCULATED_PARAMETER_VALUES = [
  4e7,
  10**(-16.75),
  5.5,
  5.5,
  10**(0.5)
]

P_VALUE_TRUTH = [0.5, 0.5, 0.5, 0.5, 0.5]

def test_set_regular_param():
  test = {"a":1, "b":2}
  pmm = ParameterManipulatorMixin()
  pmm.set_regular_param("a", 2, test)

  assert (test["a"] == 2), "ParameterManipulatorMixin did not set regular param correctly!"

def test_write_working_model_file():
  truth = {"a":1}

  pmm = ParameterManipulatorMixin()
  pmm.working_model_file = PARAM_DICT["working_model_file"]
  pmm.model_dict = truth
  pmm.write_working_model_file()

  with open(pmm.working_model_file, 'r') as f:
    test = json.load(f)

  assert (test == truth), "ParameterManipulatorMixin did not write working model file correctly!"

def test_write_best_model_file():
  pmm = ParameterManipulatorMixin()
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.best_model_file = PARAM_DICT["best_model_file"]
  pmm.read_original_model_file()
  pmm.write_best_model_file()

  with open(pmm.original_model_file, 'r') as f:
    model_truth = json.load(f)
  with open(pmm.best_model_file, 'r') as f:
    model_test = json.load(f)

  # shallow equivalence test.
  assert (model_test == model_truth), ("ParameterManipulatorMixin did not write best model "
    "correctly!")

def test_read_original_model_file():
  with open(PARAM_DICT["original_model_file"], 'r') as f:
    truth = json.load(f)
  
  pmm = ParameterManipulatorMixin()
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.read_original_model_file()

  # Again, this is a shallow test of equivalence but this is fine for right now.
  assert (pmm.model_dict == truth), ("ParameterManipulatorMixin did not read original model file "
    "correctly!")

  # Check to make sure the flattened dictionary was read in correctly. The key in this dictionary
  # is the index with which the value in the dictionary should be located.
  entries = {
    0: (["muscle"], ["no_of_half_sarcomeres", 1]),
    -1: (["c_kinetics", "scheme", 1, "transition", 0, "rate_parameters"], [2, 4])
  }
  for index, truth in entries.items():
    assert (pmm._flattened_model_dict[index] == truth), ("ParameterManipulatorMixin did not flatten "
      "model dictionary correctly!")

def test_read_optimization_structure():
  pmm = ParameterManipulatorMixin()
  pmm.p_objs = []
  pmm.optimization_template_file = PARAM_DICT["optimization_template_file"]
  with open(PARAM_DICT["original_model_file"], 'r') as f:
    mod_dict = json.load(f)
  pmm._flattened_model_dict = jgrapht.flatten_tree(mod_dict)
  pmm.read_optimization_structure()
  
  assert (pmm.p_objs), "ParameterManipulatorMixin did not read any parameters in optimization structure!"

  truth_objects = [
    ParamObject(2e7, 6e7, 0.5, "lin", ["thin_parameters", "a_k_on"]),
    ParamObject(-15.5, -18, 0.5, "log", ["titin_parameters", "t_k_stiff"]),
    ParamObject(1, 10, 0.5, "lin", ["m_kinetics", "scheme", 0, "transition", 0, "rate_parameters", 
      2]),
    ParamObject(3.0, 8.0, 0.5, "lin", ["m_kinetics", "scheme", 1, "transition", 0, "extension"]),
    ParamObject(0, 1, 0.5, "log", ["c_kinetics", "scheme", 1, "transition", 0, "rate_parameters", 
      0])
  ] 

  test_objects = pmm.p_objs
  msg = "ParameterManipulatorMixin did not set ParamObject correctly!"
  for i, obj_truth in enumerate(truth_objects):
    assert (test_objects[i].min_value == obj_truth.min_value), msg
    assert (test_objects[i].max_value == obj_truth.max_value), msg
    assert (test_objects[i].p_value == obj_truth.p_value), msg
    for j in range(len(obj_truth.p_lookup)):
      assert (test_objects[i].p_lookup[j] == obj_truth.p_lookup[j])
    assert (test_objects[i].calculated_value == obj_truth.calculated_value)

def test_update_parameters():
  pmm = ParameterManipulatorMixin()
  pmm.p_values = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5])
  pmm.p_objs = []
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.read_original_model_file()
  pmm.optimization_template_file = PARAM_DICT["optimization_template_file"]
  pmm.read_optimization_structure()

  pmm.update_parameters()

  for i, truth in enumerate(CALCULATED_PARAMETER_VALUES):
    assert (pmm.p_objs[i].calculated_value == truth), ("ParameterManipulatorMixin did not "
    "update parameter correctly!")

def test_param_dump():
  pmm = ParameterManipulatorMixin()
  pmm.p_values = np.asarray(P_VALUE_TRUTH)
  pmm.p_objs = []
  pmm.output_dir = PARAM_DICT["output_dir"]
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.read_original_model_file()
  pmm.optimization_template_file = PARAM_DICT["optimization_template_file"]
  pmm.read_optimization_structure()
  pmm.update_parameters()
  pmm.dump_param_information()

  param_dump = np.loadtxt(os.path.join(pmm.output_dir, "parameter_history.txt"), skiprows=1)
  p_dump = np.loadtxt(os.path.join(pmm.output_dir, "p_history.txt"), skiprows=1)

  for i in range(len(P_VALUE_TRUTH)):
    p_truth = pmm.p_values[i]
    param_value_truth = CALCULATED_PARAMETER_VALUES[i]
    assert (p_dump[i] == p_truth), ("ParameterManipulatorMixin did not dump p value correctly!")
    assert(param_dump[i] == param_value_truth), ("ParameterManipulatorMixin did not dump parameter "
      "correctly!")

def test_record_extreme_p_values():
  pmm = ParameterManipulatorMixin()
  pmm.p_values = np.asarray([0.01, 0.99, 0.5, 0.5, 0.5])
  pmm.p_objs = []
  pmm.output_dir = PARAM_DICT["output_dir"]
  pmm.iteration_number = 1
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.read_original_model_file()
  pmm.optimization_template_file = PARAM_DICT["optimization_template_file"]
  pmm.read_optimization_structure()
  pmm.update_parameters()

  pmm.record_extreme_p_values()

  warnings_text_truth = ("WARNING: P value for parameter \"t_k_stiff\" > 0.95; iteration 1!\n"
    "\tvalue = 0.99\n"
    "WARNING: P value for parameter \"a_k_on\" < 0.05; iteration 1!\n"
    "\tvalue = 0.01\n")
  with open(os.path.join(pmm.output_dir, "WARNINGS.log"), 'r') as f:
    warnings_text_test = f.read()

  assert (warnings_text_test == warnings_text_truth), ("ParameterManipulatorMixin did not record "
    "p value/parameter value warnings correctly!")

def test_setup_parameters():
  pmm = ParameterManipulatorMixin()
  pmm.options_file = PARAM_DICT["options_file"]
  pmm.original_model_file = PARAM_DICT["original_model_file"]
  pmm.optimization_template_file = PARAM_DICT["optimization_template_file"]
  pmm.setup_parameters()

  p_val_truth = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5])
  p_value_history_truth = [[0.5], [0.5], [0.5], [0.5], [0.5]]
  for i in range(len(p_val_truth)):
    assert (pmm.p_values[i] == p_val_truth[i]), ("ParameterManipulatorMixin did not set up p "
      "values correctly!")

    assert (pmm.p_value_history[i][0] == p_value_history_truth[i][0]), ("ParameterManipulatorMixin "
      "did not set up p value history correctly!")
  

