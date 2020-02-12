"""Holds the SimulationRunner class"""

class SimulationRunner:
  """Basic class for running simulations in "unintelligent" way"""
  def __init__(self, fibersim_file, options_file, model_file, protocol_file, fit_mode, 
    fit_variable, output_dir, target_data, time_steps_to_steady_state=2500, 
    compute_rolling_average=False):
    """Initializes a SimulationRunner object
    
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
    fit_mode : str
        The mode with which this `SimulationRunner` is assessing its error. One of ["time", 
          "end_point"].
    fit_variable : str
        The variable with which the SimulationRunner should assess its error. Currently only valid
        option is "muscle_force".
    output_dir : str
        The path to the output directory for this simulation.
    target_data : str or numpy.ndarray
        Either the path to the target data for this simulation or a numpy.ndarray of the target 
        data.
    time_steps_to_steady_state : int, optional
        The number of time steps it takes for the simulation to reach steady state, by default 2500.
        These points are ignored in the evaluation of the `SimulationRunner`s error.
    compute_rolling_average : bool, optional
        Whether to compute a rolling average of the data to smooth out signal, by default False.
    """
    self.fibersim_file = fibersim_file
    self.options_file = options_file
    self.model_file = model_file
    self.protocol_file = protocol_file
    self.fit_mode = fit_mode
    self.fit_variable = fit_variable
    self.output_dir = output_dir
    self.target_data = target_data
    self.time_steps_to_steady_state = time_steps_to_steady_state
    self.compute_rolling_average = compute_rolling_average

    