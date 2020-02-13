"""
Created on Thur Feb 13 11:20:00 2020

@author: Dylan Colli

Purpose: Mixin for manipulating parameters/instruction files from optimization template.
"""
import os
import sys
import json

sys.path.append("../../Models/FiberSim/Python_files/")
from util import instruct

from params import ParamObject

class ParameterManipulatorMixin:
  """Mixin class for manipulating parameters/instruction files from optimization templates."""
  def read_options_file(self):
    """Reads options file into class dictionary."""
    with open(self.options_file, 'r') as f:
      self.options_dict = json.load(f)

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

  def set_rate_param(self, rate_key, new_value, traversed_model_dict):
    """Sets the rate parameter in the given traversed model dictionary
    
    Inputs:
      rate_key -> str. The key for the rate equation number and parameter that you would like to 
                  change. This is in the '@' notation. For example, if we want to change rate 
                  equation 2, parameter 5, the rate_key would be "2@5".
      new_value -> float. The new value you would like to update the rate parameter to.
      traversed_model_dict -> list. The model dictionary traversed down to the "rate_equations" 
                              object. For example, if you read in the model dictionary via the 
                              'json' module, you would call this function using:
                                model["kinetics"]["rate_equations"]
    Example use:
      If we read in the model file we would like to modify via the 'json' module into a dictionary 
      called "model" and we want to set the first parameter of rate equation #5 to 10.0, we would call
      this function as follows:
        set_rate_param("5@1", 10.0, model["kinetics"]["rate_equations"])

    Returns:
      No return. This function modifies the dictionary. 
    """
    rate_eqn_idx, rate_param_idx = [int(i) - 1 for i in rate_key.split('@')]
    
    # Skip the pre-parameter information. 
    rate_param_idx += 4

    parsed_rate_eqn = traversed_model_dict[rate_eqn_idx].split()
    parsed_rate_eqn[rate_param_idx] = str(new_value)

    # Join and replace the parsed line.
    traversed_model_dict[rate_eqn_idx] = "  ".join(parsed_rate_eqn)

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
    
    # Set the initial p_values by recursively searching the dictionary.
    self.__recurs_read_param(this_dict=self.optimization_template_dict)

  def write_working_model_file(self):
    """Writes the working JSON model file into the class."""
    with open(self.working_model_file, 'w') as f:
      json.dump(self.model_dict, f, indent=2)

  def read_original_model_file(self):
    """Reads the original JSON model file into the class."""
    with open(self.original_model_file, 'r') as f:
      self.model_dict = json.load(f)

  def dump_param_information(self):
    """Dumps the parameter information for this iteration of the optimizer."""
    # Get the file name.
    file_name = os.path.join(self.output_dir, "parameter_history.txt")

    # If the file hasn't been written yet, open it and write the headers.
    if not os.path.isfile(file_name):
      f = open(file_name, 'w')
      param_names = [obj.p_lookup[-1] for obj in self.p_objs]
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
      param_names = [obj.p_lookup[-1] for obj in self.p_objs]
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

  def __recurs_read_param(self, key=None, this_dict=None, param_path=[]):
    """Recursively traverses the optimization dictionary structure to set p-values."""
    traverse = False
    if not key:
      # We know this is the first function call in the recursive call and we need to traverse
      # the dictionary.
      traverse = True

    elif isinstance(this_dict[key], dict):
      # Add this key as a node to the parameter path.
      param_path.append(key)

      # We know this is a dictionary, check if it's an optimization dictionary with p-values.
      if "p_value" not in this_dict[key].keys():
        traverse = True
        this_dict = this_dict[key]
      
      # Otherwise, we can set the p_value here.
      else:
        # Check to make sure the optimization template has been specified correctly.
        p_mode = this_dict[key]["p_mode"]
        p_min = this_dict[key]["min_value"]
        p_max = this_dict[key]["max_value"]
        p_value = this_dict[key]["p_value"]

        assert (p_mode == "lin" or p_mode == "log"), (
          "p_mode for parameter \"{}\" in optimization template must be \"lin\" or \"log\"!".format(
            key))
        assert (isinstance(p_value, (int, float))), (
          "p_value for parameter \"{}\" in optimization template must be a number!".format(key))
        assert (isinstance(p_min, (int, float))), (
          "min_value for parameter \"{}\" in optimization template must be a number!".format(key))
        assert (isinstance(p_max, (int, float))), (
          "max_value for parameter \"{}\" in optimization template must be a number!".format(key))

        # Store the values for this parameter.
        self.p_objs.append( ParamObject(p_min, p_max, p_value, p_mode, param_path) )
    
    if traverse:
      for sub_key in this_dict.keys():
        self.__recurs_read_param(key=sub_key, this_dict=this_dict, param_path=param_path.copy())

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
      # Check to see if this is a rate parameter to set.
      if '@' in new_k:
        self.set_rate_param(new_k, new_param_val, traversed_model_dict)
      else:
        self.set_regular_param(new_k, new_param_val, traversed_model_dict)