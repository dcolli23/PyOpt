"""Tests the Worker initialization"""
import os
import sys
import copy
import shutil

import pytest
import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt import worker

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)
  elif os.path.isfile(path):
    os.remove(path)

WORKER_DICT = {
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

def test_worker_direct_initialization():
  """Tests the initialization of a Worker object via direct supply of parameters"""
  w = worker.Worker(**WORKER_DICT)
  assert (w)

def test_worker_fig_initialization():
  """Tests the initialization of a Worker's figure"""
  # Monkey-patch the Worker's figure and plot initialization to make sure they're called.
  def initialize_figure(self):
    self.figure_method_called = True
  def make_plots(self):
    self.plot_method_called = True
  
  this_w_dict = copy.copy(WORKER_DICT)
  this_w_dict["display_progress"] = True
  
  old_initialize_figure = worker.Worker.initialize_figure
  worker.Worker.initialize_figure = initialize_figure
  old_make_plots = worker.Worker.make_plots
  worker.Worker.make_plots = make_plots
  worker.Worker.figure_method_called = False
  worker.Worker.plot_method_called = False

  w = worker.Worker(**this_w_dict)

  assert (w.figure_method_called), "Figure method was not called!"
  assert (w.plot_method_called), "Make plot method was not called!"

  # Do some cleanup to fix the monkeypatching.
  worker.Worker.initialize_figure = old_initialize_figure
  worker.Worker.make_plots = old_make_plots
  del worker.Worker.figure_method_called
  del worker.Worker.plot_method_called

def test_monkeypatch_bleed():
  """Tests that the monkeypatching done for tests does not bleed across tests"""
  w = worker.Worker(**WORKER_DICT)

  # This handles the expected attribute error.
  with pytest.raises(AttributeError):
    w.figure_method_called

def test_worker_with_min_error_callback():
  """Tests the worker when initialized with a newly defined minimum error callback"""
  this_w_dict = copy.copy(WORKER_DICT)

  def new_min_error_callback(self):
    """The new minimum error callback for the initialized worker"""
    return True

  this_w_dict["min_error_callback"] = new_min_error_callback

  w = worker.Worker(**this_w_dict)

  assert (w.min_error_callback()), "Minimum error callback not updated!"

def test_worker_pass_in_target_data_array():
  this_w_dict = copy.copy(WORKER_DICT)

  # Read the target data in.
  this_w_dict["target_data"] = np.loadtxt(this_w_dict["target_data"])
  w = worker.Worker(**this_w_dict)
  assert (w), "Worker did not initialize correctly!"

