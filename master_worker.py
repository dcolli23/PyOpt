# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:20:00 2019

@author: Dylan Colli

Purpose: Class for Particle Swarm Optimization.
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import imageio

from . import worker
from . import display_util as du

class MasterWorker:
  """Class for particle swarm optimization."""
  def __init__(self,
               fibersim_file_string,
               protocol_file_string,
               options_file_string,
               fit_mode,
               fit_variable,
               original_json_model_file_string,
               best_model_file_string,
               optimization_json_template_string,
               output_dir_string,
               target_data_string,
               n_workers,
               time_steps_to_steady_state=2500,
               compute_rolling_average=False,
               display_progress=True,
               plot_animation_string="./plot_animation.mp4"):
    """Initializes a MasterWorker object
    
    Parameters
    ----------
    fibersim_file_string : str
        The file path to the FiberSim executable.
    protocol_file_string : str
        The file path to the protocol file for this simulation.
    options_file_string : str
        The file path to the options file.
    fit_mode : str
        The fit mode for this simulation. Currently only accepts "time".
    fit_variable : str
        The variable that is being fit in this optimization job. Currently only accepts 
        "muscle_force".
    original_json_model_file_string : str
        The file path to the original JSON model file.
    best_model_file_string : str
        The file path to where the best model file will be saved.
    optimization_json_template_string : str
        The file path to the optimization template, where the initial p-values are stored.
    output_dir_string : str
        The path to the output directory for this simulation.
    target_data_string : str
        The file path to the target data. Must be in a two column format where the first column is
        the time and the second column is the data for the `fit_variable`. 
        Note:: The time column for this must currently be in 1 millisecond increments.
    n_workers : int
        The number of workers used in this particle swarm optimization.
    time_steps_to_steady_state : int, optional
        The number of time steps it takes for FiberSim to reach steady state, by default 2500.
    compute_rolling_average : bool, optional
        Whether or not to compute a rolling average to smooth out any stochasticity, by default 
        False.
    display_progress : bool, optional
        Whether to display the progress using multiple matplotlib plots, by default True.
    plot_animation_string : str, optional
        The file string for the output optimization animation showing progress, by default 
        "./plot_animation.mp4".
    """
    self.fibersim_file_string = fibersim_file_string
    self.protocol_file_string = protocol_file_string
    self.options_file_string = options_file_string
    self.fit_mode = fit_mode
    self.fit_variable = fit_variable
    self.original_json_model_file_string = original_json_model_file_string
    self.best_model_file_string = best_model_file_string
    self.optimization_json_template_string = optimization_json_template_string
    self.output_dir_string = output_dir_string
    self.target_data_string = target_data_string
    self.n_workers = n_workers
    self.time_steps_to_steady_state = time_steps_to_steady_state
    self.compute_rolling_average = compute_rolling_average
    self.display_progress = display_progress
    self.plot_animation_string = plot_animation_string

    # Create the output directory if it does not already exist.
    if not os.path.isdir(self.output_dir_string):
      os.mkdir(self.output_dir_string)

    # Set the minimum error across all workers to a ridiculously high value such that any new
    # simulation in the first round of parameter evaluation will overwrite it as the "best".
    self.min_error_all_workers = np.inf
    self.error_values = []
    self.best_error_values = []

    # Create a the dictionary we'll use to initialize all of our workers.
    worker_dict = {
      "fibersim_file_string":fibersim_file_string,
      "protocol_file_string":protocol_file_string,
      "options_file_string":options_file_string,
      "fit_mode":fit_mode,
      "fit_variable":fit_variable,
      "original_json_model_file_string":original_json_model_file_string,
      "best_model_file_string":best_model_file_string,
      "optimization_json_template_string":optimization_json_template_string,
      "target_data_string":target_data_string,
      "time_steps_to_steady_state":time_steps_to_steady_state,
      "compute_rolling_average":compute_rolling_average,
      "display_progress":False
    }

    # Create the list where we'll store all of the workers.
    self.workers = []
    self.best_worker_index = 0

    # Create all of the workers.
    for i in range(n_workers):
      # We're going to create a new directory structure for each of our workers.
      worker_base_dir = os.path.join(self.output_dir_string, "worker_" + str(i))
      if not os.path.isdir(worker_base_dir):
        os.mkdir(worker_base_dir)

      # Add/modify the dictionary key/value pairs that the worker needs to be initialized.
      worker_dict["working_json_model_file_string"] = os.path.join(worker_base_dir,
        "working_model.json")
      worker_dict["output_dir_string"] = os.path.join(worker_base_dir, "results")

      # Create the output directory for this worker if it has not been created previously.
      if not os.path.isdir(worker_dict["output_dir_string"]):
        os.mkdir(worker_dict["output_dir_string"])

      # Initialize the worker.
      worker_obj = worker.Worker(**worker_dict)

      # Add the worker to our list of workers.
      self.workers.append(worker_obj)
    
    # Create the plot if we wish to visualize progress through the optimization task.
    if self.display_progress:
      self.initialize_optimization_plots()
      optimization_plot_path = os.path.join(output_dir_string, "optimization_plot_movie.wmv")
      self.plot_writer = imageio.get_writer(optimization_plot_path, fps=2)
      
  def drive_workers(self, particle_array):
    """Drives the workers for an iteration of particle swarm optimization.

    NOTE: This is meant to be passed to the PySwarms optimizer and not called explicitly.
    
    Inputs:
      particle_array (np.ndarray) - shape (n_workers, n_params)

    Returns:
      errors (np.ndarray) - evaluated errors for all of the workers with size (n_workers, )  
    """
    # Do some obligatory error-checking.
    assert (isinstance(particle_array, np.ndarray)), "particle_array must be numpy array!!"
    assert (particle_array.shape[0] == self.n_workers), ("First dimension of particle_array must "
      +"match number of workers!")

    # Create a numpy array for storing our errors.
    this_iteration_error = np.ones(self.n_workers) * np.inf

    # Sequentially run the workers. Since a singular worker almost always takes up all of our 
    # processing power, there's no sense in parallelizing this part of the optimization. Since this
    # is the case, we can run sequentially with little fear of poorly performing code.
    for i, worker in enumerate(self.workers):
      print ("Running worker #{}/{}".format(i + 1, len(self.workers)))
      # Pull out the p-values for this worker.
      p_values = particle_array[i, :]

      # Fit the worker with the new p-values.
      worker_error = worker.fit_worker(p_values=p_values)

      # Store the error in the error storage we created.
      this_iteration_error[i] = worker_error

      # Dump the worker's parameter history with the latest p-values.
      worker.dump_param_information()
    
    # Find the best error across this iteration's error.
    this_its_best_error = np.min(this_iteration_error)

    # Check to see if the minimum error got any better.
    if this_its_best_error < self.min_error_all_workers:
      # This means that the best model file has been updated and we need to update the minimum
      # error for each of our workers.
      self.min_error_all_workers = this_its_best_error
      for worker in self.workers:
        worker.best_error = self.min_error_all_workers
      
      # Find the worker with the minimum error so we can plot this as the best fit.
      self.best_worker_index = np.argmin(this_iteration_error)
      best_worker = self.workers[self.best_worker_index]
      self.__update_fit_plot(best_worker)

      # Append this new best error to the list of best errors.
      self.best_error_values.append(self.min_error_all_workers)
    else:
      # We just need to add a redundant best error value since this iterations error value isn't
      # better than the best we already had.
      self.best_error_values.append(self.best_error_values[-1])

    # Update the best error for this iteration for display purposes.
    self.error_values.append(this_its_best_error)

    if self.display_progress:
      # Update the convergence and p-value plots.
      self.__update_plots()

      # Save the plot snapshot.
      snapshot = du.get_img_from_fig(self.fig, dpi=180)
      self.plot_writer.append_data(snapshot)

    return this_iteration_error

  def initialize_optimization_plots(self):
    """Initializes the plots that keeps track of this optimization job's progress"""
    self.__initialize_figure()
    self.__make_plots()

  def __initialize_figure(self):
    """Initializes the figure for plotting optimization information"""
    self.fig = plt.figure(figsize=[18, 5])
    plt.ion()

  def __make_plots(self):
    """Sets up the plots for visualizing progress of PSO"""
    # Plot the global convergence (best error over time).
    self.global_convergence_ax = self.fig.add_subplot(151)
    self.global_convergence_plot, = self.global_convergence_ax.plot(self.best_error_values)
    self.global_convergence_ax.set_title("Best Error")
    self.global_convergence_ax.set_xlabel("Iteration Number")
    self.global_convergence_ax.set_ylabel("Sum of Squared Errors")
    self.global_convergence_ax.set_yscale("log")

    # Plot the convergence (error over time).
    self.convergence_ax = self.fig.add_subplot(152)
    self.convergence_plot, = self.convergence_ax.plot(self.error_values)
    self.convergence_ax.set_title("Iteration's Min Error")
    self.convergence_ax.set_xlabel("Iteration Number")
    self.convergence_ax.set_ylabel("Sum of Squared Errors")
    self.convergence_ax.set_yscale("log")

    # Plot the best fit thus far.
    self.fit_ax = self.fig.add_subplot(153)
    
    # Use the first worker's target data since they're all the same.
    target_data = self.workers[0].target_data
    self.fit_ax.plot(target_data[:,0], target_data[:,1:], color='k', label="Target")
    self.fit_plot, = self.fit_ax.plot(target_data[:,0], np.ones_like(target_data[:,0] * 0),
      label="Model") # basically making a dummy plot for our model right now.
    self.fit_ax.set_title("Current Best Fit")
    self.fit_ax.legend()

    # Plot all of the worker's fits.
    self.all_fit_ax = self.fig.add_subplot(154)
    self.all_fit_plots = []
    for worker in self.workers:
      # Make a dummy "place holder" plot for this worker's fit.
      this_fit_plot, = self.all_fit_ax.plot(target_data[:, 0], np.ones_like(target_data[:,0]) * 0)
      self.all_fit_plots.append(this_fit_plot)
    self.all_fit_ax.plot(target_data[:,0], target_data[:, 1:], color='k', label="Target")
    self.all_fit_ax.set_title("Fit of All Workers")
    self.all_fit_ax.legend()

    # Make a plot for the current p-values of all of our workers.
    self.p_ax = self.fig.add_subplot(155)
    self.p_ax.set_title("P-values for Workers")
    y_values = range(len(self.workers[0].p_objs))

    # Get the names of all of our parameters we're fitting.
    y_labels = [obj.p_lookup[-1] for obj in self.workers[0].p_objs]

    # Make a container for holding the p-plots. This will make it possible to update these at each 
    # iteration.
    self.p_plots = []

    # Make all of plots.
    for worker in self.workers:
      these_ps = [obj.p_value for obj in worker.p_objs]
      this_p_plot, = self.p_ax.plot(these_ps, y_values, marker='o')
      # this_p_plot = self.p_ax.scatter(these_ps, y_values)
      self.p_plots.append(this_p_plot)
    
    # Set the y labels.
    self.p_ax.set_yticks(y_values)
    self.p_ax.set_yticklabels(y_labels)

    # Set the tight layout to update each time the figure is drawn. This is necessary so the axis
    # labels don't overlap.
    self.fig.set_tight_layout(True)

    plt.show()

    # draw the data on the plot
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

  def __update_plots(self):
    """Updates plots for visualizing the PSO"""
    # Update the global best convergence (best error over time).
    self.global_convergence_plot.set_xdata(range(1, len(self.best_error_values) + 1))
    self.global_convergence_plot.set_ydata(self.best_error_values)
    self.global_convergence_ax.set_xlim(right=len(self.best_error_values))
    self.global_convergence_ax.set_ylim(bottom=0.95*np.min(self.best_error_values),
      top=1.05 * np.max(self.best_error_values))

    # Update the convergence (error over time) plot.
    self.convergence_plot.set_xdata(range(1, len(self.error_values) + 1))
    self.convergence_plot.set_ydata(self.error_values)
    self.convergence_ax.set_xlim(right=len(self.error_values))
    self.convergence_ax.set_ylim(bottom=0.95*np.min(self.error_values),
      top=1.05 * np.max(self.error_values))

    # The best fit plot is updated seperately so we can skip this.

    # Update the fit plots for all of the workers.
    min_y = 1e50
    max_y = -1e50
    for i, worker in enumerate(self.workers):
      new_fit = worker.fit_data[self.time_steps_to_steady_state:]
      min_y = np.min((np.min(new_fit), min_y))
      max_y = np.max((np.max(new_fit), max_y))
      self.all_fit_plots[i].set_ydata(new_fit)
    self.all_fit_ax.set_ylim(bottom = 0.95 * min_y, top = 1.05 * max_y)

    # Update the p-value plots.
    min_p_value = np.inf
    max_p_value = -np.inf
    for i, worker in enumerate(self.workers):
      # Get the p-values.
      p_values = [obj.p_value for obj in worker.p_objs]

      # Replot the p-values.
      self.p_plots[i].set_xdata(p_values)

      # Find the minimum and maximum
      min_p_value = min(p_values + [min_p_value])
      max_p_value = max(p_values + [max_p_value])
    
    # Update the x range of the p-value plots.
    self.p_ax.set_xlim(
      left = 0.95 * min_p_value,
      right = 1.05 * max_p_value
    )

    # Draw the new data on the figure.
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

    # Pause the plot so it can process mouse events.
    plt.pause(0.1)
  
  def __update_fit_plot(self, best_worker):
    """Updates the best fit plot with the newest best worker"""
    self.fit_plot.set_ydata(best_worker.fit_data[self.time_steps_to_steady_state:])

    # Change the color to match the color of the best worker.
    best_color = self.p_plots[self.best_worker_index].get_color()
    self.fit_plot.set_color(best_color)

    # Update the legend to reflect the new color.
    self.fit_ax.legend()

    new_upper_y_lim = 1.05 * np.max((best_worker.target_data[:,1], 
      best_worker.fit_data[self.time_steps_to_steady_state:]))

    new_lower_y_lim = 0.95 * np.min((best_worker.target_data[:,1],
      best_worker.fit_data[self.time_steps_to_steady_state:]))

    self.fit_ax.set_ylim(bottom=new_lower_y_lim, top=new_upper_y_lim)

    # Draw the new data on the plot.
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()
