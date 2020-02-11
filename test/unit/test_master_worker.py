"""Tests the MasterWorker initialization"""
import os
import sys
import copy
import shutil

import pytest
import matplotlib.pyplot as plt


ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt import master_worker

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)

MASTER_WORKER_DICT = {
  "fibersim_file_string": None,
  "protocol_file_string": os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
  "options_file_string": os.path.join(TEST_DATA_ROOT, "sim_options.json"),
  "fit_mode": "time",
  "fit_variable": "muscle_force",
  "original_json_model_file_string": os.path.join(TEST_DATA_ROOT, "original_model.json"),
  "best_model_file_string": os.path.join(TEST_RESULT_DIR, "best_model.json"),
  "optimization_json_template_string": os.path.join(TEST_DATA_ROOT, "optimization_template.json"),
  "output_dir_string": TEST_RESULT_DIR,
  "target_data_string": os.path.join(TEST_DATA_ROOT, "forces.txt"),
  "n_workers": 2,
  "time_steps_to_steady_state":1,
  "compute_rolling_average":False,
  "display_progress":False,
  "plot_animation_file_root":None
}

def test_master_worker_direct_initialization():
  """Tests the initialization of a MasterWorker object via direct supply of parameters"""
  mw = master_worker.MasterWorker(**MASTER_WORKER_DICT)
  assert (mw)

def test_master_worker_fig_initialization():
  """Tests the initialization of a MasterWorker's figure"""
  # Monkey-patch the MasterWorker's figure and plot initialization to make sure they're called.
  def initialize_figure(self):
    self.figure_method_called = True
  def make_plots(self):
    self.plot_method_called = True
  
  this_mw_dict = copy.copy(MASTER_WORKER_DICT)
  this_mw_dict["display_progress"] = True
  
  old_initialize_figure = master_worker.MasterWorker.initialize_figure
  master_worker.MasterWorker.initialize_figure = initialize_figure
  old_make_plots = master_worker.MasterWorker.make_plots
  master_worker.MasterWorker.make_plots = make_plots
  master_worker.MasterWorker.figure_method_called = False
  master_worker.MasterWorker.plot_method_called = False

  mw = master_worker.MasterWorker(**this_mw_dict)

  assert (mw.figure_method_called), "Figure method was not called!"
  assert (mw.plot_method_called), "Make plot method was not called!"

  # Do some monkeypatch cleanup.
  master_worker.MasterWorker.initialize_figure = old_initialize_figure
  master_worker.MasterWorker.make_plots = old_make_plots
  del master_worker.MasterWorker.figure_method_called
  del master_worker.MasterWorker.plot_method_called

def test_master_worker_monkeypatch_bleed():
  mw = master_worker.MasterWorker(**MASTER_WORKER_DICT)

  # This handles the expected attribute error.
  with pytest.raises(AttributeError):
    mw.figure_method_called
  
def test_master_worker_with_min_error_callback():
  """Tests the master worker when initialized with a newly defined minimum error callback"""
  this_mw_dict = copy.copy(MASTER_WORKER_DICT)

  def new_min_error_callback(self):
    """The new minimum error callback for the initialized worker"""
    return True
  
  this_mw_dict["min_error_callback"] = new_min_error_callback

  mw = master_worker.MasterWorker(**this_mw_dict)

  assert (mw.min_error_callback()), "Minimum error callback not updated!"

def test_master_worker_save_plot_animation():
  """Tests that the master worker can save a plot animation"""
  mw = master_worker.MasterWorker(**MASTER_WORKER_DICT)

  true_animation_output_dir = os.path.join(TEST_RESULT_DIR, "optimization_results", "animation")
  mw.plot_animation_file_root = "test_"
  mw.error_values = [1, 2, 3, 4, 5]

  true_file_name = os.path.join(true_animation_output_dir, 
    mw.plot_animation_file_root+str(len(mw.error_values))+".png")

  mw.fig = plt.figure()
  
  mw.save_plot_snapshot()

  assert (os.path.isfile(true_file_name)), "MasterWorker not saving animation correctly!"


