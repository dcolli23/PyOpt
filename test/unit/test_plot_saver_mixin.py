"""Tests the PlotSaverMixin class"""
import os
import sys
import copy
import shutil

import pytest
import matplotlib.pyplot as plt

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt.plot_saver_mixin import PlotSaverMixin

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
# for path in os.listdir(TEST_RESULT_DIR):
#   path = os.path.join(TEST_RESULT_DIR, path)
#   if os.path.isdir(path):
#     shutil.rmtree(path)
#   elif os.path.isfile(path):
#     os.remove(path)

def test_save_plot_snapshot_with_specified_names():
  ps = PlotSaverMixin()
  ps.animation_output_dir = TEST_RESULT_DIR
  ps.plot_animation_file_root = "plot_saver_mixin_test_"
  ps.error_values = [1]
  
  ps.fig = plt.figure()
  ax = ps.fig.add_subplot(111)
  ax.plot(range(10), range(10))

  ps.save_plot_snapshot()

  saved_fig_path_truth = os.path.join(ps.animation_output_dir, ps.plot_animation_file_root
    +str(len(ps.error_values))+".png")

  assert (os.path.isfile(saved_fig_path_truth)), "PlotSaverMixin did not save figure correctly!"
  
def test_save_plot_snapshot_with_default_names():
  ps = PlotSaverMixin()
  ps.output_dir = TEST_RESULT_DIR
  default_animation_file_root = "optimization_plot_"
  ps.error_values = [1]
  
  ps.fig = plt.figure()
  ax = ps.fig.add_subplot(111)
  ax.plot(range(10), range(10))

  ps.save_plot_snapshot()

  saved_fig_path_truth = os.path.join(ps.output_dir, default_animation_file_root
    +str(len(ps.error_values))+".png")

  assert (os.path.isfile(saved_fig_path_truth)), "PlotSaverMixin did not save figure correctly!"

