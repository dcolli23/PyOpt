# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:20:00 2019

@author: Dylan Colli

Purpose: Class for bookkeeping of parameters that are optimized.
"""

import sys
import numpy as np

class ParamObject():
  """Object for holding parameter information."""
  def __init__(self, min_value, max_value, p_value, p_mode, p_lookup):
    self.min_value = min_value
    self.max_value = max_value
    self.p_value = p_value
    self.p_mode = p_mode
    self.p_lookup = p_lookup
    self.calculated_value = self.calculate_parameter()
  
  def calculate_parameter(self):
    """Calculates the interpolated parameter for this object."""
    param_value = calculate_value(self.p_lookup[-1], self.min_value, self.max_value, 
      self.p_value, self.p_mode)

    self.calculated_value = param_value

    return param_value

def bracket_p_value(p_val):
  """Brackets the p_value between 0 and 1"""
  p_val = abs(p_val)
  if divmod(p_val, 1)[0] % 2 == 0:
    p_val = p_val % 1
  else:
    p_val = 1 - (p_val % 1)

  return p_val

def calculate_value(p_name, min_val, max_val, p_val, p_mode, verbose=False):
  """Returns the calculated parameter value."""
  # Find the correct p value.
  p_val = bracket_p_value(p_val)

  # Set the parameter value based on the adjustability and interpolation mode.
  if p_mode == "lin":
    # This is the linear interpolation mode.
    param_value = p_val * (max_val - min_val) + min_val
  elif p_mode == "log":
    # This is the logarithmic interpolation mode.
    power = p_val * (max_val - min_val) + min_val
    param_value = 10**power
  else:
    raise RuntimeError("Unknown parameter interpolation mode encountered in instruction file for "
      +"parameter \"{}\"".format(p_name))

  if verbose: print(p_name+": ", param_value)

  return param_value

def back_calculate_p_value(p_name, min_val, max_val, calc_val, p_mode, verbose=False):
  """Back calculates the p-value needed to give the calculated value (calc_val)"""
  if p_mode == "log":
    calc_val = np.log10(calc_val)
  elif p_mode == "lin":
    pass
  else:
    raise RuntimeError("Unknown parameter interpolation mode encountered in instruction file for "
      +"parameter \"{}\"".format(p_name))
  
  p_val = (calc_val - min_val) / (max_val - min_val)

  if verbose: print(p_name+": ", p_val)

  return p_val