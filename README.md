# spinalcord-segmentation-1D-template-matching
"""
This algorithm is intended to segment a human cervical spinal cord. It was created for research purposes only and is not intended for use in a clinical setting.

The main algorithm is written in Python, with subroutines written in C++.
Below are the basic steps for running the algoithm.
To execute the program, you will need Python 2.7. The packages listed at the bottom of this document are required in various parts of the program and need to be installed.

You are required to use cMake to build the C++ soure code on your platform.

Once installed properly, you will need to edit the list of subjects used in the python file, corresponding to the names of actual subject folders. The original data folder location is stored in the setup_03.py file, which will need to be modified.

The algorithm uses images which have been saved as NRRD files. This can be accomplished in many ways, but the author used 3D Slicer for all image manipulation.

The algorithm was built with certain image acquisitions parameters in mind and therefore it is likely that if different parameters were used, re-orienting the image might be required. This involves changing the orientation parameters in the algorithm.

This code has only been used on a Intel Core i7-2600K @3.4 GHz with 24 GB RAM, running Ubuntu 10.24.

For further detailed instructions, please refer to the academic paper which led you to this repository.

Required Python packages:
import itk 
import gc
import resource
from mayavi import tools
import numpy as np
import math
import cmath
import cPickle as pickle
import os
import time
from time import sleep
import multiprocessing
import socket
import subprocess
from scipy.stats import mode
from scipy import optimize
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab2
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import MultipleLocator
from pylab import *
from matplotlib.ticker import MultipleLocator
from matplotlib import ticker
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import minimum_filter1d
from scipy.ndimage.filters import gaussian_filter1d
from sklearn import mixture
from sklearn import linear_model
from sklearn.cluster import KMeans #@UnresolvedImport
from mayavi import mlab
from scipy.interpolate import griddata
