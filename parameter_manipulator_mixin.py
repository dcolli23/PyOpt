"""
Created on Thur Feb 13 11:20:00 2020

@author: Dylan Colli

Purpose: Mixin for manipulating parameters/instruction files from optimization template.
"""
import os
import sys
import json

import numpy as np
import jgrapht

sys.path.append("../../Models/FiberSim/Python_files/")
from util import instruct

from .params import ParamObject

class ParameterManipulatorMixin:
  """Mixin class for manipulating parameters/instruction files from optimization templates."""
  def setup_parameters(self):
    """Sets up all of the class attributes necessary to manipulate parameters"""
    # Initialize the empty parameter interpolation lists.
    self.p_objs = []

    # Read in the original JSON model file.
    self.read_original_model_file()

    # Read in the optimization structure.
    self.read_optimization_structure()

    # Form the initial p_value array and p-value history.
    self.p_values = np.asarray([obj.p_value for obj in self.p_objs])
    self.p_value_history = [[obj.p_value] for obj in self.p_objs]

  def update_parameters(self):
    """Updates the parameters for this worker from the p_vals array."""
    for i in range(self.p_values.shape[0]):
      # Update the parameter p_value from the array.
      self.p_objs[i].p_value = self.p_values[i]

      # Calculate the new parameter value.
      new_param_val = self.p_objs[i].calculate_parameter()

      # Set the new parameter value.
      self.__recurs_update_parameter(self.p_objs[i].p_lookup.copy(), new_param_val)
  
  def record_extreme_p_values(self):
    """Writes to WARNINGS.log in output directory when extreme p_values are encountered."""
    idxs_greater = [i for i, p_value in enumerate(self.p_values) if p_value > 0.95]
    idxs_lesser = [i for i, p_value in enumerate(self.p_values) if p_value < 0.05]
    if idxs_greater:
      # Get the names of the parameters that are too high.
      with open(os.path.join(self.output_dir, "WARNINGS.log"), 'a+') as f:
        str_to_write = ""
        #for obj in self.p_objs[idxs_greater]:
        for idx in idxs_greater:
          obj = self.p_objs[idx]
          str_to_write += "WARNING: P value for parameter \"{}\" > 0.95; iteration {}!\n".format(
            obj.p_lookup[-1], self.iteration_number)
          str_to_write += "\tvalue = {}\n".format(obj.p_value)
        f.write(str_to_write)
    if idxs_lesser:
      # Get the names of the parameters that are too low.
      with open(os.path.join(self.output_dir, "WARNINGS.log"), 'a+') as f:
        str_to_write = ""
        # for obj in self.p_objs[idxs_lesser]:
        for idx in idxs_lesser:
          obj = self.p_objs[idx]
          str_to_write += "WARNING: P value for parameter \"{}\" < 0.05; iteration {}!\n".format(
            obj.p_lookup[-1], self.iteration_number, ) 
          str_to_write += "\tvalue = {}\n".format(obj.p_value) 
        f.write(str_to_write)

  def set_regular_param(self, param_key, new_value, traversed_model_dict):
    """Sets the parameter in the model dictionary to the new value."""
    traversed_model_dict[param_key] = new_value

  def read_optimization_structure(self):
    """Reads the optimization structure into a dictionary.
    
    Important note: Python 3.7 dictionaries now preserve insertion order so we can be sure that
    the order of this dictionary will stay the same across the simulation. If we were using 
    Python < 3.7, this would not be the case and we'd have to use an ordereddict.
    """
    # Read the optimization structure into the dictionary object.
    with open(self.optimization_template_file, 'r') as f:
      self.optimization_template_dict = json.load(f)

    # Make parameter objects for all parameters in the optimization template.
    self.__form_p_objects_from_template()

  def write_working_model_file(self):
    """Writes the working JSON model file into the class."""
    with open(self.working_model_file, 'w') as f:
      json.dump(self.model_dict, f, indent=2)

  def write_best_model_file(self):
    with open(self.best_model_file, 'w') as f:
      json.dump(self.model_dict, f, indent=2)

  def read_original_model_file(self):
    """Reads the original JSON model file into the class."""
    with open(self.original_model_file, 'r') as f:
      self.model_dict = json.load(f)
    
    # Flatten the dictionary so we can search the parameters.
    self._flattened_model_dict = jgrapht.flatten_tree(self.model_dict)

  def dump_param_information(self):
    """Dumps the parameter information for this iteration of the optimizer."""
    # Get the file name.
    file_name = os.path.join(self.output_dir, "parameter_history.txt")

    # If the file hasn't been written yet, open it and write the headers.
    if not os.path.isfile(file_name):
      f = open(file_name, 'w')
      param_names = ['|'.join([str(value) for value in obj.p_lookup]) for obj in self.p_objs]
      str_to_write = '\t'.join(param_names) + '\n'
    else:
      # Open the file for appending
      f = open(file_name, 'a')
      str_to_write = ""
    
    param_values = [str(obj.calculated_value) for obj in self.p_objs]
    str_to_write += '\t'.join(param_values) + '\n'

    # Write the information for the parameters.
    f.write(str_to_write)

    # Tidy up.
    f.close()

    # Do the same thing for p_value history.
    file_name = os.path.join(self.output_dir, "p_history.txt")
    if not os.path.isfile(file_name):
      f = open(file_name, 'w')
      param_names = ['|'.join([str(value) for value in obj.p_lookup]) for obj in self.p_objs]
      str_to_write = '\t'.join(param_names) + '\n'
    else:
      f = open(file_name, 'a')
      str_to_write = ""
    
    # Get the p values.
    p_values = [str(obj.p_value) for obj in self.p_objs]

    # Put the information in the string to write.
    str_to_write += '\t'.join(p_values) + '\n'

    # Write the information for the parameters.
    f.write(str_to_write)

    # Tidy up.
    f.close()

  def __form_p_objects_from_template(self):
    """Forms the parameter objects from the optimization template"""
    param_list = self.optimization_template_dict["FiberSim_optimization"]["optimization_structure"][
      "parameters"]
    for param_template in param_list:
      # Check to see if the parameter is a rate parameter.
      if (param_template["name"].startswith("c_kinetics") 
          or param_template["name"].startswith("m_kinetics")):
        param_path = param_template["name"].split('|')
        # Convert all of the integer strings to integers.
        for i, string in enumerate(param_path):
          try:
            value = int(string) - 1
          except:
            value = string
          param_path[i] = value

      else:
        # We're going to have to search the flattened model dictionary for the parameter path.
        leaf_index = [i for i, leaf in enumerate(self._flattened_model_dict) if 
          leaf[1][0] == param_template["name"]]
        assert (len(leaf_index) <= 1), "Parameter, \"{}\", is not specific to one parameter".format(
          param_template["name"])
        assert (len(leaf_index) >= 1), "Parameter, \"{}\", was not found!".format(
          param_template["name"])
        leaf_index = leaf_index[0]
        param_path = self._flattened_model_dict[leaf_index][0] + [param_template["name"]]
  
      p_obj = ParamObject(param_template["min_value"], param_template["max_value"], 
        param_template["p_value"], param_template["p_mode"], param_path,
        # display_name=param_template["name"]
      )

      self.p_objs.append(p_obj)

  def __recurs_update_parameter(self, p_lookup, new_param_val, traversed_model_dict=None):
    """Updates the parameter values for this worker in the model dictionary."""
    # Get the new key for this node.
    new_k = p_lookup.pop(0)

    # Check to see if we're at the end of the parameter lookup path.
    if len(p_lookup) > 0:
      # Traverse the dictionary by one node.
      if not traversed_model_dict:
        traversed_model_dict = self.model_dict
      traversed_model_dict = traversed_model_dict[new_k]
      
      # Call the recursive function.
      self.__recurs_update_parameter(p_lookup, new_param_val, traversed_model_dict)
    
    else:
      self.set_regular_param(new_k, new_param_val, traversed_model_dict)