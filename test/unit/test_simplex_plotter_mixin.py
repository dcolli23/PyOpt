"""Tests the SimplexPlotterMixin implementation"""
import os
import sys
import copy
import shutil

import pytest
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.realpath(os.path.dirname(__file__))
PYOPT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(os.path.join(PYOPT_ROOT, ".."))
from PyOpt.simplex_plotter_mixin import SimplexPlotterMixin

TEST_DATA_ROOT = os.path.join(ROOT, "..", "data", "unit")
TEST_RESULT_DIR = os.path.join(ROOT, "..", "output", "unit")

# Clear out the previous test results.
for path in os.listdir(TEST_RESULT_DIR):
  path = os.path.join(TEST_RESULT_DIR, path)
  if os.path.isdir(path):
    shutil.rmtree(path)
  elif os.path.isfile(path):
    os.remove(path)

PLOTTER_DICT = {
  "fig":None
}

def test_fig_initialization():
  """Tests the initialization of the SimplexPlotterMixin figure initialization"""
  # Monkey-patch the Worker's figure and plot initialization to make sure they're called.
  def initialize_figure(self):
    self.figure_method_called = True
  def make_plots(self):
    self.plot_method_called = True
    
  old_initialize_figure = SimplexPlotterMixin.initialize_figure
  SimplexPlotterMixin.initialize_figure = initialize_figure
  old_make_plots = SimplexPlotterMixin.make_plots
  SimplexPlotterMixin.make_plots = make_plots
  SimplexPlotterMixin.figure_method_called = False
  SimplexPlotterMixin.plot_method_called = False

  sp = SimplexPlotterMixin()
  sp.initialize_optimization_figure()

  assert (sp.figure_method_called), "SimplexPlotterMixin figure method was not called!"
  assert (sp.plot_method_called), "SimplexPlotterMixin make plot method was not called!"

  # Do some cleanup to fix the monkeypatching.
  SimplexPlotterMixin.initialize_figure = old_initialize_figure
  SimplexPlotterMixin.make_plots = old_make_plots
  del SimplexPlotterMixin.figure_method_called
  del SimplexPlotterMixin.plot_method_called
