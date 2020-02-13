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
from PyOpt import worker_family

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)
  elif os.path.isfile(path):
    os.remove(path)

WF_DICT = {
  "fibersim_file": None,
  "options_file":os.path.join(TEST_DATA_ROOT, "sim_options.json"),
  "original_model_file":os.path.join(TEST_DATA_ROOT, "original_model.json"),
  "working_model_file":os.path.join(TEST_RESULT_DIR, "working_model_file.json"),
  "best_model_file":os.path.join(TEST_RESULT_DIR, "best_model_worker_family.json"),
  "protocol_files":[
    os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
    os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
    os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt")
  ],
  "fit_mode":"end_point",
  "fit_variable":"muscle_force",
  "optimization_template_file":os.path.join(TEST_DATA_ROOT, "optimization_template.json"),
  "output_dir":TEST_RESULT_DIR,
  "target_data":os.path.join(TEST_DATA_ROOT, "target_data_muscle_force.txt"),
  "time_steps_to_steady_state":1,
  "compute_rolling_average":False,
  "display_progress":False
}

def test_worker_family_initialization():
  wf = worker_family.WorkerFamily(**WF_DICT)

  assert (len(wf.children) == len(WF_DICT["protocol_files"])), ("WorkerFamily did not initialize "
    "correct number of children!")

def test_worker_family_fit_function():
  wf = worker_family.WorkerFamily(**WF_DICT)

  for child in wf.children:
    child.fit_worker = lambda x: 1
  
  dummy_p_values = np.asarray([1])

  assert (wf.fit(dummy_p_values) == len(wf.children)), "Family fit function did not validate!"
  

