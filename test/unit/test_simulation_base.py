"""Tests the SimulationBase initialization"""
import os
import sys
import copy
import shutil

import pytest
import matplotlib.pyplot as plt

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(PYOPT_ROOT)
from _simulation_base import SimulationBase

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)

SB_DICT = {
  "fibersim_file":None,
  "protocol_file": os.path.join(TEST_DATA_ROOT, "optimization_protocol.txt"),
  "model_file": os.path.join(TEST_DATA_ROOT, "original_model.json"),
  "options_file": os.path.join(TEST_DATA_ROOT, "sim_options.json"),
  "output_dir": TEST_RESULT_DIR
}

def test_simulation_base_initialization():
  sb = SimulationBase(**SB_DICT)

  assert (sb), "SimulationBase did not initialize correctly!"
