"""Tests the PyOpt implementation as a whole"""

import os
import sys

ROOT = os.path.realpath(os.path.dirname(__file__))
PyOpt_ROOT = os.path.join(ROOT, "..", "..", "..")
sys.path.append(PyOpt_ROOT)

def test_pyopt_import():
  import PyOpt

def test_pyopt_worker_family_import():
  from PyOpt.worker_family import WorkerFamily