"""Holds the very basic functionality that we'll need for running optimization"""

class SimulationBase:
  """The most basic unit of optimization simulations"""
  def __init__(self, fibersim_file, options_file, model_file, protocol_file, output_dir, *args, 
    **kwargs):
    """Initializes the SimulationBase class
    
    Parameters
    ----------
    fibersim_file : str
        The path to the FiberSim executable.
    options_file : str
        The path to the options file.
    model_file : str
        The path to the model file.
    protocol_file : str
        The path to the protocol file.
    output_dir : str
        The path to the output directory for this simulation.
    """
    self.fibersim_file = fibersim_file
    self.options_file = options_file
    self.model_file = model_file
    self.protocol_file = protocol_file
    self.output_dir = output_dir