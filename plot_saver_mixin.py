"""
Created on Tue Feb 11 16:24:00 2020

@author: Dylan Colli

Purpose: Mixin class for saving optimization plots.
"""
import os

import cv2

import PyOpt.display_util as du


class PlotSaverMixin:
  """Mixin class for saving optimization plots"""
  def save_plot_snapshot(self):
    """Saves a snapshot of the plots and saves to the animation output directory

    Notes: 
      - If `self.animation_output_directory` is not specified, then defaults to 
        `self.output_directory`
      - If `self.plot_animation_file_root` is not specified, then defaults to "optimization_plot_"
    """
    snapshot = du.get_img_from_fig(self.fig, dpi=180)

    try:
      out_dir = self.animation_output_dir
    except AttributeError:
      self.animation_output_dir = self.output_dir
      out_dir = self.animation_output_dir
    try:
      file_root = self.plot_animation_file_root
    except AttributeError:
      self.plot_animation_file_root = "optimization_plot_"
      file_root = self.plot_animation_file_root

    output_image_path = os.path.join(out_dir, file_root+str(len(self.error_values))+".png")
    cv2.imwrite(output_image_path, snapshot)