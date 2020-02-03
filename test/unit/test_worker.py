"""Tests the Worker initialization"""
import os
import sys
import copy


ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt import worker

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

WORKER_DICT = {
  "fibersim_file_string": None,
  "protocol_file_string": os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
  "options_file_string": os.path.join(TEST_DATA_ROOT, "sim_options.json"),
  "fit_mode": "time",
  "fit_variable": "muscle_force",
  "original_json_model_file_string": os.path.join(TEST_DATA_ROOT, "original_model.json"),
  "working_json_model_file_string": os.path.join(TEST_RESULT_DIR, "working_model.json"),
  "best_model_file_string": os.path.join(TEST_RESULT_DIR, "best_model.json"),
  "optimization_json_template_string": os.path.join(TEST_DATA_ROOT, "optimization_template.json"),
  "output_dir_string": TEST_RESULT_DIR,
  "target_data_string": os.path.join(TEST_DATA_ROOT, "forces.txt"),
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
  
  worker.Worker.initialize_figure = initialize_figure
  worker.Worker.make_plots = make_plots
  worker.Worker.figure_method_called = False
  worker.Worker.plot_method_called = False

  w = worker.Worker(**this_w_dict)

  assert (w.figure_method_called), "Figure method was not called!"
  assert (w.plot_method_called), "Make plot method was not called!"