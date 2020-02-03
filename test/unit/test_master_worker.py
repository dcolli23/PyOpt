"""Tests the MasterWorker initialization"""
import os
import sys

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt import master_worker

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

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
  "plot_animation_string":None
}

def test_master_worker_direct_initialization():
  """Tests the initialization of a MasterWorker object via direct supply of parameters"""
  mw = master_worker.MasterWorker(**MASTER_WORKER_DICT)
  assert (mw)

def test_master_worker_dict_initialization():
  """Tests the initialization of a MasterWorker object from a dictionary"""
  mw = master_worker.from_dict(MASTER_WORKER_DICT)
  assert (mw)
