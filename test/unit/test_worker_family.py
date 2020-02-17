"""Tests the WorkerFamily implementation"""
import os
import sys
import copy
import shutil

import pytest
import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt.worker_family import WorkerFamily

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)
  elif os.path.isfile(path):
    os.remove(path)

PROTOCOL_FILES = [
  os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
  os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
  os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt")
]
WF_DICT = {
  "fibersim_file": None,
  "options_file":os.path.join(TEST_DATA_ROOT, "sim_options.json"),
  "original_model_file":os.path.join(TEST_DATA_ROOT, "original_model.json"),
  "working_model_file":os.path.join(TEST_RESULT_DIR, "working_model_file.json"),
  "best_model_file":os.path.join(TEST_RESULT_DIR, "best_model_worker_family.json"),
  "protocol_files":PROTOCOL_FILES,
  "fit_mode":"end_point",
  "fit_variable":"muscle_force",
  "target_data":np.arange(len(PROTOCOL_FILES)),
  "optimization_template_file":os.path.join(TEST_DATA_ROOT, "optimization_template.json"),
  "output_dir":TEST_RESULT_DIR,
  "time_steps_to_steady_state":1,
  "compute_rolling_average":False,
  "display_progress":False
}

def test_worker_family_initialization():
  wf = WorkerFamily(**WF_DICT)

  msg = "WorkerFamily did not initialize correctly!"
  assert (wf.iteration_number == 0), msg
  assert (wf.best_error == np.inf), msg
  assert (wf.error_values == []), msg
  assert (len(wf.children) == len(WF_DICT["protocol_files"])), ("WorkerFamily did not initialize "
    "correct number of children!")

def test_worker_family_fit_function():
  wf = WorkerFamily(**WF_DICT)

  for child in wf.children:
    child.fit_worker = lambda x: 1
  
  dummy_p_values = np.asarray([1])

  assert (wf.fit(dummy_p_values) == len(wf.children)), "Family fit function did not validate!"

def test_worker_family_read_target_data_string():
  this_wf_dict = copy.copy(WF_DICT)
  this_wf_dict["target_data"] = os.path.join(WF_DICT["output_dir"], "wf_test_target_data.txt")
  data = np.arange(len(WF_DICT["protocol_files"]))
  np.savetxt(this_wf_dict["target_data"], data)

  wf = WorkerFamily(**this_wf_dict)

  assert (np.array_equal(wf.target_data, data)), "WorkerFamily did not read target_data correctly!"

def test_worker_family_read_target_data_array():
  this_wf_dict = copy.copy(WF_DICT)
  data = np.arange(len(WF_DICT["protocol_files"]))
  this_wf_dict["target_data"] = data

  wf = WorkerFamily(**this_wf_dict)

  assert (np.array_equal(wf.target_data, data)), "WorkerFamily did not read target_data correctly!"
  
def test_worker_family_read_target_data_raises_type_error():
  this_wf_dict = copy.copy(WF_DICT)
  this_wf_dict["target_data"] = {"garbage":True}  

  with pytest.raises(TypeError):
    wf = WorkerFamily(**this_wf_dict)
  
def test_worker_family_read_target_data_verifies_data_length_num_children_match():
  this_wf_dict = copy.copy(WF_DICT)
  this_wf_dict["target_data"] = np.arange(len(this_wf_dict["protocol_files"]) + 1)

  with pytest.raises(ValueError):
    wf = WorkerFamily(**this_wf_dict)

