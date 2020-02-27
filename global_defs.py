"""For specifying global constants"""
import os
import sys

ROOT = os.path.realpath(os.path.dirname(__file__))

print ("WARNING: Currently relying on FiberSim repository outside of this repository. Refactor!!")
FIBERSIM_ROOT = os.path.join(ROOT, "..", "fibersim")
FIBERSIM_MODULES_ROOT = os.path.join(FIBERSIM_ROOT, "python_files")