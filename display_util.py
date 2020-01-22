"""
Created on Wed Jan 22 10:03:00 2020

@author: Dylan Colli

Purpose: Pretty display of optimization progress.
"""

import io

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

def get_img_from_fig(fig, dpi=180):
  """Returns a numpy array of given figure at given dots per inch"""
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=dpi)
  buf.seek(0)
  img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  buf.close()
  img = cv2.imdecode(img_arr, 1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img

