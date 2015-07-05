"""
Spinal Cord Segmentation by 1D Template Matching
"""
from calculus import *
import spline
from setup_03 import *

import logging
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
    
np.set_printoptions(threshold=np.nan)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

"""
Global definitions
"""
itk.auto_progress(2)
dimension = 3

spine_grid_d = np.arange(0, 200, 0.1) 
spine_grid_theta = np.arange(0, 2 * np.pi, np.pi / 180) # 0.0174532925 radian increments
spine_grid_d_func = np.arange(0,150,0.5)
radial_grid = np.asarray(np.linspace(1, 5, 5), np.float32) #radial search distance in mm
theta_grid = np.arange(180, dtype=np.float32) * 2 * np.pi / 180

InternalPixelType = itk.F
InternalImageType = itk.Image[ InternalPixelType, dimension ]
InternalImageType2D = itk.Image[ InternalPixelType, 2 ]

GradientPixelType = itk.CovariantVector[ itk.F , 2 ]
GradientImageType = itk.Image[GradientPixelType, 2 ]
GradientInternalImageType = itk.Image[ InternalPixelType, 2]

LabelPixelType = itk.US
LabelImageType = itk.Image[ LabelPixelType, dimension ]

OutputPixelType = itk.US
OutputImageType = itk.Image[ OutputPixelType, dimension ]

functional_image_spacing = np.array([2.000, 1.0938, 1.0938], dtype=np.float32)
anatomical_image_spacing = np.array([0.30011, 0.39060, 0.39060], dtype=np.float32)


global_calc_gradients = True
global_smoothing = False

#===============================================================================
# Helper functions
#===============================================================================
def get_filename(name, data_name="", data_file_extension=data_file_extension):
    """
    Given data name returns appropriate filename.
    """
    if data_name != "":
        filename = os.path.join(data_path, name,
                                name + "-" + data_name + "." + data_file_extension)
    else:
        filename = os.path.join(data_path, name,
                                name + "." + data_file_extension)

    return filename


def load_image(filename, PixelType, volume_spacing_mm, dimension=dimension):
    """
    Returns an itk.Image. Every 'spacing' voxel is used (e.g 1st, 2nd, 3rd,...).
    """
    logger.info("Loading: %s" % filename)
    ImageType = itk.Image[PixelType, dimension]
    data_reader = itk.ImageFileReader[ ImageType ].New()
    data_reader.SetFileName(filename)
    data_array = itk.PyBuffer[ImageType].GetArrayFromImage(data_reader.GetOutput())

    data_image = itk.PyBuffer[ImageType].GetImageFromArray(data_array)
    origin = data_image.GetOrigin()

    spacing = data_image.GetSpacing()

    data_image.GetSpacing().SetElement(0, float(volume_spacing_mm[0]))
    data_image.GetSpacing().SetElement(1, float(volume_spacing_mm[1]))
    data_image.GetSpacing().SetElement(2, float(volume_spacing_mm[2]))

    origin = data_image.GetOrigin()

    spacing = data_image.GetSpacing()
        
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.

    del data_array
            
    return data_image


def set_spacing(name, data_name='', label = 0, new_name = 'new', save_old = False):
    filename = get_filename(name, data_name)
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    logger.info("Loading: %s" % filename)
    
    if label == 1:
        ImageType = itk.Image[LabelPixelType, dimension ]
    else:
        ImageType = itk.Image[InternalPixelType, dimension]
        
    data_reader = itk.ImageFileReader[ ImageType ].New()
    data_reader.SetFileName(filename)
    data_array = itk.PyBuffer[ImageType].GetArrayFromImage(data_reader.GetOutput())

    data_image = itk.PyBuffer[ImageType].GetImageFromArray(data_array)
    
    
    spacing = data_image.GetSpacing()
    print 'File Spacing (Pre): %s / %s / %s' % (spacing.GetElement(0), spacing.GetElement(1), spacing.GetElement(2))
    data_image.GetSpacing().SetElement(0, float(volume_spacing_mm[0]))
    data_image.GetSpacing().SetElement(1, float(volume_spacing_mm[1]))
    data_image.GetSpacing().SetElement(2, float(volume_spacing_mm[2]))
    
    
    #save the old version as a new file
    if save_old == True:
        filename_old = get_filename(name, '%s_old' % data_name)
        data_writer = itk.ImageFileWriter[ ImageType ].New()
        data_writer.SetFileName(filename_old)
        data_writer.SetInput(data_image)
        data_writer.Write()
    
    #set the new spacing
    print 'Data Shape: ', data_array.shape
    data_image.GetSpacing().SetElement(0, float(volume_spacing_mm[0]))
    data_image.GetSpacing().SetElement(1, float(volume_spacing_mm[1]))
    data_image.GetSpacing().SetElement(2, float(volume_spacing_mm[2]))
     
    spacing = data_image.GetSpacing()
    print 'File Spacing (Post): %s / %s / %s' % (spacing.GetElement(0), spacing.GetElement(1), spacing.GetElement(2))
    
    
    origin = data_image.GetOrigin()
    print 'Origin (Pre): %s / %s / %s' % (origin.GetElement(0), origin.GetElement(1), origin.GetElement(2))
    
    origin = data_image.GetOrigin()
    print 'Origin (Post): %s / %s / %s' % (origin.GetElement(0), origin.GetElement(1), origin.GetElement(2))
    
    
    #save the new version as the old file name
    
    filename_new = get_filename(name, '%s' % (new_name))
    data_writer = itk.ImageFileWriter[ ImageType ].New()
    data_writer.SetFileName(filename_new)
    data_writer.SetInput(data_image)
    data_writer.Write()
    

def save_image(filename, image, PixelType, volume_spacing_mm, dimension):
    """
    Saves an itk.Image.
    """
    logger.info("Saving: %s" % filename)
    ImageType = itk.Image[PixelType, dimension]
    data_writer = itk.ImageFileWriter[ ImageType ].New()
    data_writer.SetFileName(filename)
    image.GetSpacing().SetElement(0, float(volume_spacing_mm[0]))
    image.GetSpacing().SetElement(1, float(volume_spacing_mm[1]))
    image.GetSpacing().SetElement(2, float(volume_spacing_mm[2]))
    spacing = image.GetSpacing()
    print 'Saved File Spacing: %s / %s / %s' % (spacing.GetElement(0), spacing.GetElement(1), spacing.GetElement(2))
    data_writer.SetInput(image)
    data_writer.Write()

def load_data(name, data_name=''):
    """
    Returns an itk.Image object of data of subject 'name'.
    """
    filename = get_filename(name, data_name)
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    return load_image(filename, InternalPixelType, volume_spacing_mm, dimension)

def load_label(name, data_name='label'):
    """
    Returns an itk.Image object of data of subject 'name'.
    """
    filename = get_filename(name, data_name)
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    return load_image(filename, LabelPixelType, volume_spacing_mm, dimension)

def save_data(image, name, data_name="data"):
    """
    Save an itk.Image object of data of subject 'name'.
    """
    filename = get_filename(name, data_name)
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    save_image(filename, image, InternalPixelType, volume_spacing_mm, dimension)

def save_data_2D(image, name, data_name="data"):
    """
    Save an itk.Image object of data of subject 'name'.
    """
    filename = get_filename(name, data_name)
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    save_image(filename, image, InternalPixelType, volume_spacing_mm, 2)


def save_label(image, name, data_name="label_data"):
    """
    Save an itk.Image object of data of subject 'name'.
    """
    filename = get_filename(name, data_name)
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    save_image(filename, image, LabelPixelType, volume_spacing_mm, dimension)

def save_pickle(data, name, data_name="data"):
    """
    Saves a Python pickled object.
    """
    filename = get_filename(name, data_name, 'pickle')
    #print 'Filename is=',filename
    logger.info("Saving: %s" % filename)
    pickle.dump(data, open(filename, 'w'))

def save_pickle_non_subject_related(data, name):
    """
    Saves a Python pickled object.
    """
    filename = os.path.join(data_path, name + ".pickle")
    #print 'Filename is=',filename
    logger.info("Saving: %s" % filename)
    pickle.dump(data, open(filename, 'w'))

def load_pickle_non_subject_related(name):
    """
    Loads a Python pickled object.
    """
    filename = os.path.join(data_path, name + ".pickle")
    logger.info("Loading: %s" % filename)
    return pickle.load(open(filename, 'r'))

def load_pickle(name, data_name="data"):
    """
    Loads a Python pickled object.
    """
    filename = get_filename(name, data_name, 'pickle')
    logger.info("Loading: %s" % filename)
    return pickle.load(open(filename, 'r'))

def find_nearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return array[idx]

def find_nearest_index(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx

def load_image_array(name, label = False, sub_name = ''):
    if label == False:
        data_image = load_data(name, sub_name)
        return itk.PyBuffer[InternalImageType].GetArrayFromImage(data_image)
    else:
        label_image = load_label(name, sub_name)
        return itk.PyBuffer[LabelImageType].GetArrayFromImage(label_image)
        
def generate_spline_coords(N, M):
    t = np.asarray(np.linspace(0, 1, N), np.float32)
    theta = np.arange(M, dtype=np.float32) * 2 * np.pi / M
    return t, theta  

def spine_coord_to_grid(d, theta, data, grid_d=(spine_grid_d, spine_grid_theta)):
    """
    Given d (distance along the spine), theta (angles) and 
    data (a matrix of dimensions d x theta) data, interpolates the data into
    a standard grid grid_d (a tuple of (d, theta) ).
    """
    x, y = np.meshgrid(theta, d) #x will be an array sized (d.y by theta.x) and have all of the values of theta in the same rows, y will be the same size, but have the rows of d transposed into columns 
    points = zip(x.flatten(), y.flatten())
    values = data.flatten()
    xi = np.meshgrid(grid_d[1], grid_d[0])
    return griddata(points=points, values=values, xi=xi, fill_value=0)


def output_smoothed_image(name, data, output_name, sigma = 1.0):

    # smooth data with gaussian filter
    SmoothingFilterType = itk.SmoothingRecursiveGaussianImageFilter[InternalImageType,InternalImageType]
    smoothing = SmoothingFilterType.New()
    smoothing.SetInput(data)

    smoothing.SetSigma(sigma)
    smoothing.Update()
    
    itk_image = smoothing.GetOutput()
    spacing = itk_image.GetSpacing()
    print 'Spacing: %s / %s / %s' % (spacing.GetElement(0), spacing.GetElement(1), spacing.GetElement(2))
    if sigma != -1.0:
        save_data(smoothing.GetOutput(), name, output_name)
    else:
        save_data(data, name, output_name)
        
      


def calculate_gradient_of_image(name):
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    if global_smoothing == True:
        output_smoothed_image(name, load_data(name), 'smoothing', sigma = 1.0) #if sigma is -1.0 it means to not smooth the image at all
    else:
        output_smoothed_image(name, load_data(name), 'smoothing', sigma = -1.0) #if sigma is -1.0 it means to not smooth the image at all
    
    image_array = load_image_array(name, label = False)
    new_gradient = np.zeros(image_array.shape, dtype = np.ushort)
    print 'Starting CPP function...'
    gradient_calc(volume_spacing_mm, image_array, new_gradient) #found in calculus.cpp
 
    temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(new_gradient)
    save_label(temp_image, name, 'gradient_magnitude')
    
    

def transpose_image(name, subname = '', output_name = 'new_output', label = False, x = 0, y = 0, z = 0, rotate = False, num_rot = 1, flip_up = False, flip_lr = False, extra_adjust = False):
    

    if label == True:
        image_array = load_image_array(name, label = True, sub_name = subname)
    else:
        image_array = load_image_array(name, label = False, sub_name = subname)
        

    temp_new_image = np.transpose(image_array, (x,y,z))
    #print temp_new_image.shape
    if extra_adjust == True:
        new_image = temp_new_image[:,:,154/2:666-154/2] #This is based on trial and error for osirix exports which add padding
    else:
        new_image = temp_new_image
    #print new_image.shape 
    
    if flip_up == True:
        new_image = np.flipud(new_image)
    
    if flip_lr == True:
        new_image = np.fliplr(new_image)
        
    if rotate == True:
        new_image = np.rot90(new_image, num_rot)
    
    if label == True:
        temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(new_image)
        save_label(temp_image, name, output_name)
    else:
        temp_image = itk.PyBuffer[InternalImageType].GetImageFromArray(new_image)
        save_data(temp_image, name, output_name)
    

def convert_spacing(subject_names):
    
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        image_array = load_image_array(name, label = True, sub_name = 'manual_filled_spine_old spacing')
        data_image = itk.PyBuffer[LabelImageType].GetImageFromArray(image_array)
        data_image.GetSpacing().SetElement(0, float(volume_spacing_mm[0]))
        data_image.GetSpacing().SetElement(1, float(volume_spacing_mm[1]))
        data_image.GetSpacing().SetElement(2, float(volume_spacing_mm[2]))
                
        save_label(data_image, name, 'manual_filled_spine')

#===============================================================================
# Pipeline functions
#===============================================================================

def set_initial_spline_points(name, label_name, output_name):    
    """
    Uses the Slicer inputed labels to define the initial spline
    Store the values in a pickle
    """
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    #data = load_image_array(name, label = False)
    label = load_image_array(name, label = True, sub_name = label_name)
                
    PMJ_I = np.array(zip(*np.where(label == labels['PMJ']))) #returns the location of the PMJ
    
    center_I = np.array(zip(*np.where(label == labels['cord'])))
    
    s = spline.CRSpline() 
    t = np.linspace(0, 1, 20).reshape((-1, 1)) #makes an 8x1 matrix
    t = np.vstack((np.linspace(t[0], t[1], 4).reshape((-1, 1)), t[2:])) 
    
    P0 = np.mean(PMJ_I, axis=0) 
    print center_I
    I = np.argsort(center_I[:, 1])  #This is an index of sorted values
    
    spine_end_I = center_I[I[-1:-10:-1], :] 
    P1 = np.mean(spine_end_I, axis=0) 
       
    num_points = 20
    
    print 'Len(I): ', len(I)
    step_value = np.max([1, len(I) / num_points]) #Uses 1 in case len(I) is less that num_points
    print 'Step value: ', step_value
    
    P_back_temp = center_I[I[::(step_value)], :] 
    
    """
    This section calculates the linear distance between points. It's used to interpolate to ensure the points are evenly spaced
    """
    temp_z = 0
    temp_points = []
    temp_points.append(P0)
    distance = []
    distance.append(0)
    total_distance = 0
    last_point = P0
    for a in range(P_back_temp.shape[0]):
        if P_back_temp[a, 1] != temp_z:
            temp_points.append(P_back_temp[a,:]) 
            
            dist1 = (P_back_temp[a, 0] - last_point[0]) * volume_spacing_mm[0]
            dist2 = (P_back_temp[a, 1] - last_point[1]) * volume_spacing_mm[1]
            dist3 = (P_back_temp[a, 2] - last_point[2]) * volume_spacing_mm[2]
            dist = np.sqrt(dist1 ** 2 + dist2 ** 2 + dist3 ** 2)
            total_distance += dist
            distance.append(total_distance)
            
            temp_z = P_back_temp[a, 1]
            last_point = P_back_temp[a, :]
    
    """
    Calculate the distance to the last point, only if P1 is actually lower down that the last point in P_back, which isn't always the case
    """
    if P1[1] > P_back_temp[-1, 1]: 
        dist1 = (P1[0] - P_back_temp[-1, 0]) * volume_spacing_mm[0]
        dist2 = (P1[1] - P_back_temp[-1, 1]) * volume_spacing_mm[1]
        dist3 = (P1[2] - P_back_temp[-1, 2]) * volume_spacing_mm[2]
        dist = np.sqrt(dist1 ** 2 + dist2 ** 2 + dist3 ** 2)
        total_distance += dist
        distance.append(total_distance)
        temp_points.append(P1)
    
    #print 'Unique Points: %s' % temp_points
    #print 'Distance Between Points: %s' % distance
    #print 'Total Distance: %s' % total_distance
    
    """
    This section linearly interpolates between user selected points to ensure almost equal spacing down the spline
    """
    
    distance_interval = 10 #this is the distance in mm in which you want to space the points  
    new_point_array = []
    counter = 1
    curr_dist = counter * distance_interval
    
    while curr_dist < total_distance:
        #print 'Target Distance: %s' % curr_dist
        count = 0 #This will be the index of the nearest distance
        for temp_dist in distance:
            if temp_dist > curr_dist:
                pass
            else:
                count += 1
        
        #print 'Next largest distance: %s' % distance[count]
        #print 'Count: %s' % count
        point1 = temp_points[count - 1]
        point2 =  temp_points[count]
        fraction = (curr_dist - distance[count - 1]) / (distance[count] - distance[count - 1])
        #print 'Fraction: %s' % fraction
        new_x = point1[0] + (point2[0] - point1[0]) * fraction 
        new_y = point1[1] + (point2[1] - point1[1]) * fraction
        new_z = point1[2] + (point2[2] - point1[2]) * fraction
        new_point = np.array([new_x, new_y, new_z])
        new_point_array.append(new_point)
        #print 'Point1: %s; Point2: %s; New Point: %s' % (point1, point2, new_point)
    
        counter += 1
        curr_dist = counter * distance_interval
        
    P_back = np.zeros((len(new_point_array) + 1, 3), dtype = np.int)
    P_back[0,:] = P0
    count = 1
    for point in new_point_array:
        P_back[count, :] = point  
        count += 1  
    
    print 'P_back=', P_back
    
    s.add_spline_points(P_back.astype(np.float32)) #adds spline points at the start, end, and 10 evenly spaced rootlet locations in between 
    print 'P_back=', P_back
    
    
    t = np.linspace(0, 1, 10).astype(np.float32)
    dist_mm = s.get_interpolated_spline_distance(t, volume_spacing_mm)
    print 'dist_mm (before)=', dist_mm
    max_dist = dist_mm[dist_mm.shape[0] - 1]
    mm_space = np.linspace(0, max_dist, 10).astype(np.float32)
    t2 = s.get_relative_spline_distance(mm_space, volume_spacing_mm)
    #print 'mm_space=', mm_space
    
    CP = s.get_interpolated_spline_point(t2.astype(np.float32).flatten()) #gets the interpolated spline points of 's' at the locations identified by 't', which I believe is the % distance down the spline from the PMJ
    
    #print 'CP before new spline: %s' % CP
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    dist_mm = s.get_interpolated_spline_distance(t, volume_spacing_mm)
    #print 'dist_mm (after)=', dist_mm
    mm_test = np.linspace(0, 150, 10).astype(np.float32)
    #print 'CP being saved: %s' % CP
    save_pickle(CP, name, output_name)
    
    print 'Done setting initial spline points'


def compute_templates_from_spine_mask(subject_names, index_points, overwrite = False, axial_slice = True, label_name = '', save_results = False):
    """
    Returns the fixed length templates from an image in a 2D array and an index value of the edge of the spine for each of those templates
    template (2D) = returns an array with the template values, each row represents a template (radial set of points from the image)
    index_values (1D) = an array of the index values for each template where the edge of the image lies
    distance_values (1D) = an array of the distance values for each template where the edge of the image lies
    """
    
    num_index_points = index_points
    
    count = 0
    for name in subject_names:
        print 'Subject: %s' % name
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name) 
        try:
            theta_start, theta_end = load_mask_characteristics(name)
        except:
            theta_start = 0.0
            theta_end = 360
        
        image_array = load_image_array(name, label = False)
        try:
            label_array = load_image_array(name, label = True, sub_name = label_name)
        except:
            pass
        
        
        if global_calc_gradients == True:
            calculate_gradient_of_image(name)
        
        try:
            gradient_array = load_image_array(name, label = False, sub_name = 'gradient_magnitude')
        except (IOError, RuntimeError):
            logger.info('File: %s does not exist. Calculating gradient image' % ('gradient_magnitude'))
            calculate_gradient_of_image(name)
            gradient_array = load_image_array(name, label = False, sub_name = 'gradient_magnitude')
            
        try:
            manual_filled_spine_array = load_image_array(name, label = True, sub_name = 'manual_filled_spine')
        except:
            logger.info('File: %s does not exist. Need to manually create a spine mask' % ('manual_filled_spine'))
            return
             
        if label_name == '':
            CP = load_pickle(name, 'spline_control_points_manual')
        else:
            set_initial_spline_points(name, label_name = label_name, output_name = 'spline_control_points_manual_fill')
            CP = load_pickle(name, 'spline_control_points_manual_fill')
        
        s = spline.CRSpline()
        s.add_spline_points(CP.astype(np.float32))
        t_end = 1.0
        dist_mm = s.get_interpolated_spline_distance(t_end, volume_spacing_mm)
        
        N = int(dist_mm / volume_spacing_mm[1])
        M = 180
        t, theta = generate_spline_coords(N, M)
        print 'N Values: %s' % N
        
        theta_1 = int(M / 360.0 * theta_start)
        theta_2 = int(M / 360.0 * theta_end)
        print 'Theta Start Angle: %s; Theta End Angle: %s' % (theta_start, theta_end)
        print 'Theta 1 Angle: %s; Theta 2 Angle: %s' % (theta_1, theta_2)
        
        start_percent_down_spine = 0.0
        
        template, index_values, distance_values, edge_3d_points = s.create_templates(t, theta, axial_slice, start_percent_down_spine, theta_1, theta_2, gradient_array, manual_filled_spine_array, num_index_points, volume_spacing_mm, image_orientation)
        
        if count == 0:
            template_consolidated = template
            index_values_consolidated = index_values
            distance_values_consolidated = distance_values
            edge_3d_points_consolidated = edge_3d_points
            
        else:
            template_consolidated = np.append(template_consolidated, template, axis = 0)
            index_values_consolidated = np.append(index_values_consolidated, index_values, axis = 0)
            distance_values_consolidated = np.append(distance_values_consolidated, distance_values, axis = 0)
            edge_3d_points_consolidated = np.append(edge_3d_points_consolidated, edge_3d_points, axis = 0)
        
        count += 1
    
    
    if overwrite == False:
        try:
            templateDB, indexValuesDB, distanceValuesDB = load_pickle('analysis', 'template_database')
            
            if template.shape[1] != templateDB.shape[1]:
                logger.error('Database template row size of %s does not match newly found templates of size %s' %(templateDB.shape[1], template.shape[1]))
                return
            
            #Appends the old then new templates together into a single file
            templateNew = np.zeros((templateDB.shape[0] + template.shape[0], template.shape[1]), dtype = np.float32)
            for a in range(templateDB.shape[0]):
                for b in range(templateDB.shape[1]):
                    templateNew[a,b] = templateDB[a,b]
            for c in range(template.shape[0]):
                for d in range(template.shape[1]):
                    templateNew[templateDB.shape[0] + c, d] = template[c, d]
            
            indexValuesNew = np.zeros((indexValuesDB.shape[0] + index_values.shape[0]), dtype = np.float32)
            for aa in range(indexValuesDB.shape[0]):
                    indexValuesNew[aa] = indexValuesDB[aa]
            for bb in range(index_values.shape[0]):
                    indexValuesNew[indexValuesDB.shape[0] + bb] = index_values[bb]
            
            distanceValuesNew = np.zeros((distanceValuesDB.shape[0] + distance_values.shape[0]), dtype = np.float32)
            for aa in range(distanceValuesDB.shape[0]):
                    distanceValuesNew[aa] = distanceValuesDB[aa]
            for bb in range(distance_values.shape[0]):
                    distanceValuesNew[indexValuesDB.shape[0] + bb] = distance_values[bb]
            
            logger.info('New templates will be appended to existing database')
            
        except IOError:
            logger.info('Current template does not exist, a new one will be created')
            templateNew = template.astype(np.float32)
            indexValuesNew = index_values.astype(np.int32)
            distanceValuesNew = distance_values.astype(np.float32)
            
    else: #overwrite = true    
        templateNew = template_consolidated.astype(np.float32)
        indexValuesNew = index_values_consolidated.astype(np.int32)
        distanceValuesNew = distance_values_consolidated.astype(np.float32)
        edge3dPointsNew = edge_3d_points_consolidated.astype(np.float32)
    
    
    #Check if any distances are zero and delete them from the list, for graphing purposes
    counter = 1
    for a in range(distanceValuesNew.shape[0]):
        if distanceValuesNew[a] != 0:
            counter += 1
    
    tempDistanceValues = np.zeros((counter), dtype = np.float32)
    
    count = 0
    for a in range(distanceValuesNew.shape[0]):
        if distanceValuesNew[a] != 0:
            tempDistanceValues[count] = distanceValuesNew[a]
            count += 1
    
            
    sorted_array = np.sort(tempDistanceValues)
    plt.figure()
    x_range = np.arange(0, tempDistanceValues.shape[0], 1)

    plt.scatter(x_range, sorted_array, s = 0.1, color = 'r', marker = ',')
    plt.title('Distance to Edge of Spinal Cord in Image Templates')
    plt.xlabel('Template Number')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.set_xlim([0, tempDistanceValues.shape[0]])
    plt.xticks(rotation = 45)
    plt.grid(True)
    plt.ylabel('Distance to Edge (mm)')
    #plt.show()
    plt.savefig('Template Profiles.jpg', dpi = 150)
    
    output_name = ''
    for name in subject_names:
        output_name += '%s' % name
    
    if save_results:
        print 'Number of templates: %s' % len(templateNew)
        save_pickle((templateNew, indexValuesNew, distanceValuesNew, edge3dPointsNew), 'analysis', 'template_database_%s' % output_name)

    logger.critical('Template database update complete')


def clean_template_database():
    
    templateDB, indexValuesDB, distanceValuesDB = load_pickle('analysis', 'template_database')
        
    #1. delete any duplicate entries or close matches
    indexValuesDB = np.array(indexValuesDB, dtype = np.int32)
    maxOffset = 1
    method = 1
    matchThreshold = 0.95
    rangeThreshold = 4
    
    max_segment_length = 10
    segments = int(templateDB.shape[0] / max_segment_length)
    
    start_seg = 0 #all segments set to start_seg = 0
    end_seg = 20 #all segments set to: end_seg = segments - 1
    
    totalCount = 0
    maxEdgeRange = 0
    maxMatches = 0
    index = 0
    
    
    while index < templateDB.shape[0]:
        indexToDelete = []
        testArray = np.zeros((1, templateDB.shape[1]), dtype = templateDB.dtype)
    
        testArray[0, :] = templateDB[index, :]
                                
        testMatchPercent, testTemplateIndex, testEdgeIndex, newEdge = convolve_arrays_fast(matchThreshold, maxOffset, method, testArray, templateDB, indexValuesDB)
    
        matches = 0
        lower = 100
        upper = 0
        for a in range(testMatchPercent.shape[1]):
                if testMatchPercent[0,a] >= matchThreshold:
                    matches += 1
                    if testEdgeIndex[0,a] < lower:
                        lower = testEdgeIndex[0,a]
                    if testEdgeIndex[0,a] > upper:
                        upper = testEdgeIndex[0,a]
                
        edgeRange = upper - lower
        if edgeRange > maxEdgeRange:
            maxEdgeRange = edgeRange
        
        if matches > maxMatches:
            maxMatches = matches
        
        #Looks through the list of matches and records the index of any matches, excluding the first one
        if matches > 1 and edgeRange <= rangeThreshold:
            for e in range(testTemplateIndex.shape[1]):
                if testMatchPercent[0,e] < 0.99: #prevents the row being tested from being removed
                    if (testTemplateIndex[0, e] in indexToDelete) or (testTemplateIndex[0, e] == 0):
                        pass
                    else:
                        indexToDelete.append(testTemplateIndex[0, e])
                        totalCount += 1    
        
        #Removes values from the original list
        print 'index: %s, matches: %s, edgeRange: %s, len(indexToDelete): %s' %(index, matches, edgeRange, len(indexToDelete))
        #print 'indexValuesDB.shape: ', indexValuesDB.shape
        templateNew = np.zeros((templateDB.shape[0] - len(indexToDelete), templateDB.shape[1]), dtype = templateDB.dtype)
        indexValuesNew = np.zeros((indexValuesDB.shape[0] - len(indexToDelete)), dtype = indexValuesDB.dtype)
        distanceValuesNew = np.zeros((distanceValuesDB.shape[0] - len(indexToDelete)), dtype = distanceValuesDB.dtype)
        
        nextIndex = 0
        for aa in range(templateDB.shape[0]):
            if aa in indexToDelete:
                pass
            else:
                templateNew[nextIndex,:] = templateDB[aa,:]
                indexValuesNew[nextIndex] = indexValuesDB[aa]
                distanceValuesNew[nextIndex] = distanceValuesDB[aa]
                nextIndex += 1
        
        templateDB = templateNew
        indexValuesDB = indexValuesNew
        distanceValuesDB = distanceValuesNew
        
        index += 1

        
    
    print 'totalCount: ', totalCount
    print 'maxMatches: ', maxMatches
    print 'maxEdgeRange: ', maxEdgeRange
    print 'templateDB.shape: ', templateDB.shape
    
    save_pickle((templateDB, indexValuesDB, distanceValuesDB), 'analysis', 'template_database_CLEANINGTEST')
            

def get_radial_gradient_values_all(name, index_points, N, theta_M, axial_slice = True, spline_input = 'initial'):
    """
    Returns the fixed length templates from an image in a 2D array
    template = returns an array with the template values, each row represents a template (radial set of points from the image), size is t x THETA_M
    """ 
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    if axial_slice:
        axial_image = 1
    else:
        axial_image = 0
    
    CP = load_pickle(name, 'spline_control_points_' + spline_input)
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    t, theta = generate_spline_coords(N, theta_M)
    
    if global_calc_gradients == True:
        calculate_gradient_of_image(name)
    
    try:
        gradient_array = load_image_array(name, label = False, sub_name = 'gradient_magnitude')
    except (IOError, RuntimeError):
        logger.info('File: %s does not exist. Calculating gradient image' % ('gradient_magnitude'))
        calculate_gradient_of_image(name, smoothing = False)
        gradient_array = load_image_array(name, label = False, sub_name = 'gradient_magnitude')
    
    template = s.get_full_radial_values(t, theta, axial_image, gradient_array, index_points, volume_spacing_mm, image_orientation)
    
    save_pickle(template, 'analysis', '%s-radials_N%s_M%s' %(name, N, theta_M))
    
    logger.info('Radial values saved')
    
    
             

def produce_new_edge_overlay(name, newEdgeIndex, index_type = False, axial_slice = True, subImageNum = '', spline_input = 'initial', output_name = ''):
    """
    Takes an index of edge indexes (which should have been created using template arrays from the same subject)
    and creates an overlay of the spine surface.
    If there is a zero in the input array, i.e., the edge couldn't be determined, the surface will be left blank
    """
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    if axial_slice == True:
        axial_image = 1
    else:
        axial_image = 0
    
    N = newEdgeIndex.shape[0]
    theta_M = newEdgeIndex.shape[1]
    
    t, theta = generate_spline_coords(N, theta_M)
    
    #This image is loaded simply to get the dimensions to create a new overlay image
    image_array = load_image_array(name, label = False, sub_name = subImageNum)
    
    spline_name = 'spline_control_points_' + spline_input
    
    CP = load_pickle(name, spline_name)
    print 'Control Points: ', CP
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    new_spine_label = np.zeros(image_array.shape, dtype = np.ushort)
    
    if index_type == True:
        s.create_new_edge_overlay_from_index_values(t, theta, volume_spacing_mm, image_orientation, axial_image, newEdgeIndex, new_spine_label)
    else:
        s.create_new_edge_overlay_from_distance_values(t, theta, volume_spacing_mm, image_orientation, axial_image, newEdgeIndex, new_spine_label)
    
    temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(new_spine_label)
    save_label(temp_image, name, 'surface_spine_' + output_name)      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        

def calculate_centers_from_filled_image(name, image_orientation, image_name = 'manual_filled_spine', spline_input = 'initial', spline_output = 'refined', subImageNum = ''):
    
    if axial_slices:
        axial_image = 1
    else:
        axial_image = 0
    
    if functional_image:
        subName = subImageNum + '-'
        N = 125
        theta_M = 90
        try:
            manual_filled_spine_array = load_image_array(name, label = True, sub_name = subName + image_name)
        except (IOError, RuntimeError):
            logger.info('File: %s does not exist. Exiting program' %(name + '-' + subName + image_name))
            return
    else:
        subName = ''
        N = N_curr
        theta_M = M_curr
        try:
            manual_filled_spine_array = load_image_array(name, label = True, sub_name = image_name)
        except (IOError, RuntimeError):
            logger.info('File: %s does not exist. Exiting program' %(name + '-' + image_name))
            return
        
    t, theta = generate_spline_coords(N, theta_M)
        
    CP = load_pickle(name, subName + 'spline_control_points_' + spline_input)
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    #The function below will return 0,0,0 for any slices that have 1 or more radials not filled in
    CPTemp = s.calculate_center_from_filled(t, theta, image_orientation, axial_image, manual_filled_spine_array)
    
    count = 0
    for a in range(CPTemp.shape[0]):
        if CPTemp[a,0] != 0 and CPTemp[a,1] != 0 and CPTemp[a,2] != 0:
            count += 1
    
    percentComplete = (count * 1.0) / CPTemp.shape[0] * 100.0  
    logger.critical('%s %% of slices (%s slices) have a full set of radials' % (percentComplete, count))
    CPNewTemp = np.zeros((count, 3), dtype = np.float32)
    
    count = 0
    for b in range(CPTemp.shape[0]):
        if CPTemp[b,0] != 0 and CPTemp[b,1] != 0 and CPTemp[b,2] != 0:
            CPNewTemp[count, 0] = CPTemp[b, 0]
            CPNewTemp[count, 1] = CPTemp[b, 1]
            CPNewTemp[count, 2] = CPTemp[b, 2]
            count += 1
    
    CPNew = CPNewTemp[0::10,:]
    print 'CPOld = ', CP
    print 'CPNew = ', CPNew
    output_name = 'spline_control_points_' + spline_output
    
    save_pickle(CPNew, name, output_name)


def get_radius_values_from_edgeindex(name, edgeIndex, axial_slices = True, spline_input = 'initial', subImageNum = ''):
    """
    Converts edge indexed values into radii values (mm) from the center of each spine plane
    """
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    if axial_slices:
        axial_image = 1
    else:
        axial_image = 0
    
    try:
        CP = load_pickle(name, 'spline_control_points_' + spline_input)
    except : 
        logger.warning('Spline points can not be set. Please check file name.')
        return
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    t, theta = generate_spline_coords(edgeIndex.shape[0], edgeIndex.shape[1])
    
    radiusIndex = s.get_radii_from_edgeindex(t, theta, volume_spacing_mm, image_orientation, axial_image, edgeIndex)
    #print radiusIndex
    
    return radiusIndex
        

def draw_spline(name, intensity, N_curr, M_curr, spline_input = 'initial', image_outname = 'spline_image'):
    
    CP = load_pickle(name, 'spline_control_points_' + spline_input)
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    t, theta = generate_spline_coords(N_curr, M_curr)
    
    
    
    points = s.get_interpolated_spline_point(t) 
    
    temp_image = load_data(name)
    
    for i in range(points.shape[0]):
        temp_image.SetPixel([int(points[i,2]),int(points[i,1]),int(points[i,0])],intensity)
        #temp_image.SetPixel([int(normals[i,2]),int(normals[i,1]),int(normals[i,0])],50)
    save_data(temp_image, name, image_outname)
    
    print 'Done Drawing Spline'


def check_normals(name, image_orientation, subImageNum = '', spline_input = 'initial'):
    
    N = 300
    
    t = np.asarray(np.linspace(0, 1, N), np.float32)
    
    divider = 2 #N divided by this gives the number of normals shown
    
    #This image is loaded simply to get the dimensions to create a new overlay image
    image_array = load_image_array(name, label = False, sub_name = subImageNum)
    label_array = load_image_array(name, label = True, sub_name = 'label')
     
    set_initial_spline_points(name, image_array, label_array, output_name = 'spline_control_points_initial')
    CP = load_pickle(name, 'spline_control_points_' + spline_input)
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    new_spine_label = np.zeros(image_array.shape, dtype = np.ushort)
    
    s.draw_normals(t, image_orientation, divider, new_spine_label)
    
    temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(new_spine_label)
    save_label(temp_image, name, 'normals')
    print CP
    

def interpolate_to_larger_grid(input_radii_grid, z_multiplier, theta_multiplier):
    """ Takes an input of radial distances [t, theta] and interpolates a new set of radial distances [t * multipler, theta] """ 
    
    t_old, theta = generate_spline_coords(input_radii_grid.shape[0], input_radii_grid.shape[1])
    t_new, theta_new = generate_spline_coords(input_radii_grid.shape[0] * z_multiplier, input_radii_grid.shape[1] * theta_multiplier)
    #np.asarray(np.linspace(0, 1, t_old * multiplier), np.float32)
    
    #Interpolates into a new sized grid, i.e., more t-values
    interpolated_radii_grid = spine_coord_to_grid(t_old, theta, input_radii_grid, grid_d=(t_new, theta_new))#.astype(np.float32)
    
    return interpolated_radii_grid


def add_to_segmentation(input_file, existing_file = None):
    if existing_file == None:
        return input_file
    else:
        if (input_file.shape[0] != existing_file.shape[0]) or (input_file.shape[1] != existing_file.shape[1]):
            logger.critical('File sizes do not match, will not add the new file to the existing file.')
            return input_file
        
        for a in range(existing_file.shape[0]):
            for b in range(existing_file.shape[1]):
                if existing_file[a, b] == 0:
                    existing_file[a, b] = input_file[a, b]
        
        count = 0
        for a in range(existing_file.shape[0]):
            for b in range(existing_file.shape[1]):
                if existing_file[a, b] != 0:
                    count += 1
        percentComplete = 100.0 * count / (existing_file.shape[0] * existing_file.shape[1])
        logger.info('%s %% of the edge points have been filled.' % percentComplete)
        return existing_file


def recursive_segmentation(name, template, matchThreshold, min_matches, match_rate_threshold, N_curr, M_curr, index_points, axial_slice = True, spline_input = 'initial', label_name = 'label'):
    
    template, index_values, distance_values, edge_3d_points = load_pickle('analysis', 'template_database_%s' % template)
    print 'Template shape: ', template.shape
    
    get_radial_gradient_values_all(name, index_points, N_curr, M_curr, axial_slice, spline_input = spline_input)
    testArrayInput = load_pickle('analysis', '%s-radials_N%s_M%s' % (name, N_curr, M_curr))
    
    method = 2 #2 = average value, 3 = maximum, 1 = minimum, 4 = average of max and average, 5 = average of min and average
    
    segLength = N_curr * M_curr #Takes 50 slices (~9k radials)
    loopCounter = 0 #the starting point
    lastEdgeIndex = 0
    
    newEdgeFinal = np.zeros((segLength / M_curr, M_curr), dtype = np.int32)
    matchPercent = np.zeros((segLength / M_curr, M_curr), dtype = np.float32)
    matchedTemplateIndex = np.zeros((segLength, 10), dtype = np.int32) #The Y dimension is the maximum number of array indexes that will be stored
    matchPercentOut = np.zeros((segLength, 2), dtype = np.int32) #this keeps tracks of the match% and how many matches are made for each array
            
    logger.info('Starting recursive segmentation...')
    convolve_arrays_recursive(matchThreshold, matchThreshold, min_matches, method, loopCounter, lastEdgeIndex, testArrayInput, template, index_values, newEdgeFinal, matchPercent, matchedTemplateIndex, matchPercentOut)
    
    logger.info('Done recursive segmentation...')
    
    
    save_pickle(newEdgeFinal, 'analysis', '%s-EdgeIndex_N%s_M%s_P%s' %(name, N_curr, M_curr, int(matchThreshold * 100)))
    
    count_0 = 0
    count_50 = 0
    count_100 = 0
    count_1000 = 0
    count_over = 0
    num_matches = 0
    
    for a in range(matchPercentOut.shape[0]):
        if matchPercentOut[a,1] >= match_rate_threshold:
            num_matches += 1
            
    
    match_rate = 100.0 * num_matches / matchPercentOut.shape[0]
    print '%% of patients with more than %s matches at a %s threshold: %s%%' % (min_matches, match_rate_threshold, match_rate)
         
    print '0 matches: %s' % count_0
    print '1-50 matches: %s' % count_50
    print '51-100 matches: %s' % count_100
    print '101-1000 matches: %s' % count_1000
    print '>1000 matches: %s' % count_over
     
    
    return newEdgeFinal, matchPercent, matchedTemplateIndex, match_rate
     
  

def smoothing_filter_1D_c(inputArray, axial_smoothing = True, smoothing_type = 'average', kernelSize = 5):
    if axial_smoothing:
        axial = 1
    else:
        axial = 0
        
    if smoothing_type == 'median':
        method = 1 
    else:
        method = 2 #Mean smoothing
    
    output_array = smoothing_filter_1D(axial, method, kernelSize, inputArray) #calls a c++ function found in calculus.cpp
    
    return output_array
  

def smoothing_filter_1D_python(inputArray, axial_smoothing = True, smoothing_type = 'average', kernelSize = 5):
    """ Smoothes a set of points. If radial_smoothing is true, the 2D input will be converted by rows, then columns. If radial_smoothing is not selected it will convert by columns, then rows. """
    
    if inputArray.dtype == np.int32:
        testArray = np.zeros((kernelSize), dtype = np.int32)
        collapsedArray = np.zeros((inputArray.shape[0] * inputArray.shape[1]), dtype = np.int32)
        outputArrayTemp = np.zeros((inputArray.shape[0] * inputArray.shape[1]), dtype = np.int32)
        outputArray = np.zeros((inputArray.shape[0], inputArray.shape[1]), dtype = np.int32)
    else:
        testArray = np.zeros((kernelSize), dtype = np.float32)
        collapsedArray = np.zeros((inputArray.shape[0] * inputArray.shape[1]), dtype = np.float32)
        outputArrayTemp = np.zeros((inputArray.shape[0] * inputArray.shape[1]), dtype = np.float32)
        outputArray = np.zeros((inputArray.shape[0], inputArray.shape[1]), dtype = np.float32)
    
    
    #Create a 1D array to run the filter over
    if axial_smoothing:
        for a in range(inputArray.shape[0]):
            for b in range(inputArray.shape[1]):
                collapsedArray[a * inputArray.shape[1] + b] = inputArray[a,b]
    else:
        for aa in range(inputArray.shape[1]):
            for bb in range(inputArray.shape[0]):
                collapsedArray[aa * inputArray.shape[0] + bb] = inputArray[bb,aa] 
    
    count = 0        
    for c in range(collapsedArray.shape[0]):
        if count < kernelSize or count > (collapsedArray.shape[0] - kernelSize):
            count += 1
            outputArrayTemp[c] = collapsedArray[c]
        else: #it's somewhere in the target range
            for d in range(kernelSize):
                if d == (kernelSize / 2):
                    testArray[d] = collapsedArray[c]
                else: 
                    #print 'd: ',d
                    #print 'other index: %s' %(c - int(kernelSize / 2) + d)
                    #print count
                    testArray[d] = collapsedArray[c - int(kernelSize / 2) + d]     
             
            #Remove all non-zero elements from the list
            counter_non_zero = 0
            for a in range(testArray.shape[0]):
                if testArray[a] > 0:
                    counter_non_zero += 1
                    
            test_array_non_zero = np.zeros((counter_non_zero), dtype = np.float32)
            counter_new = 0
            for a in range(testArray.shape[0]):
                if testArray[a] > 0:
                    test_array_non_zero[counter_new] = testArray[a]
                    counter_new += 1
            
            
            
            if smoothing_type == 'median':
                #outputArrayTemp[c] = np.median(testArray)
                outputArrayTemp[c] = np.median(test_array_non_zero)
            elif smoothing_type == 'minimum':
                #outputArrayTemp[c] = np.amin(testArray)
                outputArrayTemp[c] = np.amin(test_array_non_zero)
            else:
                #outputArrayTemp[c] = np.average(testArray)
                outputArrayTemp[c] = np.average(test_array_non_zero)
                
            count += 1
    
    #Convert back to 2D array
    if axial_smoothing:
        for e in range(outputArray.shape[0]):
            for f in range(outputArray.shape[1]):
                outputArray[e,f] = outputArrayTemp[e * outputArray.shape[1] + f]
    else:
        for ee in range(outputArray.shape[1]):
            for ff in range(outputArray.shape[0]):
                outputArray[ff,ee] = outputArrayTemp[ee * outputArray.shape[0] + ff]
    
    
    return outputArray


def compare_segmentations_edge_difference(name, input1, input2, subImageNum = '', spline_input = 'initial'):
    
    num_index_points = 50
        
    image_array = load_image_array(name, label = False)
    label_array = load_image_array(name, label = True, sub_name = 'label')
    try:
        gradient_array = load_image_array(name, label = False, sub_name = 'gradient_magnitude')
    except (IOError, RuntimeError):
        logger.info('File: %s does not exist. Calculating gradient image' % ('gradient_magnitude'))
        calculate_gradient_of_image(name)
        gradient_array = load_image_array(name, label = False, sub_name = 'gradient_magnitude')
        
    try:
        test1_array = load_image_array(name, label = True, sub_name = input1)
    except:
        logger.info('File: %s does not exist.' % (input1))
        return
    
    try:
        test2_array = load_image_array(name, label = True, sub_name = input2)
    except:
        logger.info('File: %s does not exist.' % (input2))
        return
    
    try:
        CP = load_pickle(name, 'spline_control_points_' + spline_input)
    except IOError: 
        set_initial_spline_points(name, image_array, label_array, output_name = 'spline_control_points_initial')
        CP = load_pickle(name, 'spline_control_points_initial')
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    t, theta = generate_spline_coords(N_curr, M_curr) 
    
    template1, index_values1, distance_values1 = s.create_templates(t, theta, gradient_array, test1_array, num_index_points, volume_spacing_mm, image_orientation)
    template2, index_values2, distance_values2 = s.create_templates(t, theta, gradient_array, test2_array, num_index_points, volume_spacing_mm, image_orientation)
    
    distanceValues1 = np.zeros((N_curr, M_curr), dtype = np.float32)
    distanceValues2 = np.zeros((N_curr, M_curr), dtype = np.float32)

    for a in range(distanceValues1.shape[0]):
        for b in range(distanceValues1.shape[1]):
            distanceValues1[a,b] = distance_values1[a * M_curr + b]
            distanceValues2[a,b] = distance_values2[a * M_curr + b]
                
    radiusDifferences = np.zeros((distanceValues1.shape), dtype = np.float32)
    
    totalCount = 0
    count1 = 0
    targetDiff = 0.4
    for c in range(distanceValues1.shape[0]):
        for d in range(distanceValues1.shape[1]):
            if distanceValues1[c,d] != 0 and distanceValues2[c,d] != 0:
                if abs(distanceValues1[c,d] - distanceValues2[c,d]) < targetDiff:
                    count1 += 1
                    #radiusDifferences[c,d] = distanceValues1[c,d] - distanceValues2[c,d]
                totalCount += 1
    
    percentFound = 1.0 * count1 / totalCount * 100
    
    logger.info('Percent of edge points with less than %s mm difference: %s %%' % (targetDiff, percentFound))



def compare_segmentations(type, name, input1, input2):   
    #Calculates the intersection / union of the two input labels
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    try:
        test1_array = load_image_array(name, label = True, sub_name = input1)
    except:
        logger.info('File: %s does not exist.' % (input1))
        return
    
    try:
        test2_array = load_image_array(name, label = True, sub_name = input2)
    except:
        logger.info('File: %s does not exist.' % (input2))
        return
    
    if type == 'jaccard':
        matchPercent = jaccard_index(test1_array, test2_array)
        print 'Jaccard Index: ', matchPercent
        del test1_array
        del test2_array
        return matchPercent
    elif type == 'dice':
        roi_output = np.zeros(test1_array.shape, dtype = np.ushort)
        matchPercent = dice_coeff(test1_array, test2_array, roi_output)
        print 'Dice Coefficient: ', matchPercent
        temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(roi_output) 
        save_label(temp_image, name, 'compare-%s_%s' % (input1, input2))
        del test1_array
        del test2_array
        return matchPercent
    elif type == 'hausdorff':
        max_dist = haus_dist(50, 0, volume_spacing_mm, test1_array, test2_array) #offset for search is the first value
        print 'Hausdorff Distance: ', max_dist
        del test1_array
        del test2_array
        return max_dist
    elif type == 'hausdorff-mean':
        max_dist = haus_dist(50, 1, volume_spacing_mm, test1_array, test2_array) #offset for search is the first value
        print 'Mean Distance: ', max_dist
        del test1_array
        del test2_array
        return max_dist
    


def calculate_new_center_from_edge_segmentation(name, edgeSegmentation, matchPercent, N_curr, M_curr, axial_slice = True, spline_input = 'initial', spline_output = 'updated'):
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    if axial_slice:
        axial_slices = 1
    else:
        axial_slices = 0
    
    CP = load_pickle(name, 'spline_control_points_' + spline_input)
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    
    t, theta = generate_spline_coords(N_curr, M_curr) 
    
    #THIS LABEL ARRAY IS NEW, TO TEST HOW THE CENTER CALCULATION IS WORKING
    image_array = load_image_array(name, label = False)
    new_spine_label = np.zeros(image_array.shape, dtype = np.ushort)
    
    CPTemp = s.calculate_center_points_from_edge_segmentation(t, theta, volume_spacing_mm, image_orientation, axial_slices, edgeSegmentation, matchPercent, new_spine_label)
    
    
    count = 0
    for a in range(CPTemp.shape[0]):
        if CPTemp[a,0] != 0 and CPTemp[a,1] != 0 and CPTemp[a,2] != 0:
            count += 1
    
    percentComplete = (count * 1.0) / CPTemp.shape[0] * 100.0  
    logger.critical('%s %% of slices (%s slices) have a full set of radials' % (percentComplete, count))
    CPNewTemp = np.zeros((count, 3), dtype = np.float32)
    
    count = 0
    for b in range(CPTemp.shape[0]):
        if CPTemp[b,0] != 0 and CPTemp[b,1] != 0 and CPTemp[b,2] != 0:
            CPNewTemp[count, 0] = CPTemp[b, 0]
            CPNewTemp[count, 1] = CPTemp[b, 1]
            CPNewTemp[count, 2] = CPTemp[b, 2]
            count += 1
    
    P0 = CP[0,:]
    print 'P0:', P0
    CPNewTemp = np.vstack((P0, CPNewTemp[0::4,:])).astype(np.float32)
    #CPNew = CPNewTemp[0::4,:]
    
    s = spline.CRSpline()
    """ NOTE THAT THIS IS USING CPNEWTEMP NOT CPNEW """
    s.add_spline_points(CPNewTemp.astype(np.float32))
    
    """
    This section ensures that the CPs are spaced evenly down the spline
    """
    t = np.linspace(0, 1, 50).astype(np.float32)
    dist_mm = s.get_interpolated_spline_distance(t, volume_spacing_mm)
    #print 'dist_mm=', dist_mm
    max_dist = dist_mm[dist_mm.shape[0] - 1]
    mm_space = np.linspace(0, max_dist, 10).astype(np.float32)
    t2 = s.get_relative_spline_distance(mm_space, volume_spacing_mm)
    
    CPDistributed = s.get_interpolated_spline_point(t2.astype(np.float32).flatten()) #gets the interpolated spline points of 's' at the locations identified by 't', which I believe is the % distance down the spline from the PMJ
    
    print 'CPOld = ', CP
    print 'CPDistributed = ', CPDistributed
    output_name = 'spline_control_points_' + spline_output
    
    save_pickle(CPDistributed, name, output_name)

def run_convert_spine_segmentation_to_plane(name, save_results=True):
    """
    Given an extracted spine and segmentation image, generates a 2D map
    of the spine per position and direction perpendicular to spine (t, theta). 
    """

    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    t1 = time.time()
    logger.info("Flattening spine of subject: %s" % name)

    data_image = load_data(name)
    data = itk.PyBuffer[InternalImageType].GetArrayFromImage(data_image)

    CP = load_pickle(name, 'spline_control_points_centered')
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32)) #Initializes the spline with the control points that were saved
    
    N = 800
    t = np.asarray(np.linspace(0, 1, N), np.float32)
    d = s.get_interpolated_spline_distance(t, volume_spacing_mm)
    
    """
    M=400
    """
    M = 180;
    #Creates an array of 400 angles from 0 to 2*pi 
    theta = np.arange(M, dtype=np.float32) * 2 * np.pi / M

    
    D = s.parse_spine_cylinder_coord(t, theta, data / data.max(), 1, 20, 3) #t=% of distance down the spline, method is 3 (sum of values*1/radius), 2=sum of values*radius, 1=sum of values

    #Converts from relative positions to absolute/mm positions down the spline
    D_grid = spine_coord_to_grid(d, theta, D)
    #print 'D_grid.shape[0]=',D_grid.shape[0]
    #print 'D_grid.shape[1]=',D_grid.shape[1]
    plt.figure()
    plt.grid(True)
#    plt.plot(d, np.mean(D, axis=1))
#    plt.figure()
#    plt.grid(True)
    plt.plot(spine_grid_d, np.mean(D_grid, axis=1))
    plt.show()
#
#    plt.figure()
#    plt.grid(True)
#    plt.plot(theta, np.mean(D, axis=0))
#    plt.figure()
#    plt.grid(True)
#    plt.plot(spine_grid_theta, np.mean(D_grid, axis=0))

    #spine_grid_d is 0 to 200 (mm) in 0.1mm increments and spine_grid_theta is 0 to 360 in 1 degree increments 
    if save_results:
        # save results when interpolated to normal grid
        save_pickle((spine_grid_d, spine_grid_theta, D_grid), name, 'flattened_spine')

    t2 = time.time()
    logger.info("Done in %i:%.3f [mm:ss]" % (int((t2 - t1) / 60), (t2 - t1) % 60))
    
    print 'Done Conversion to Plane'
    return D_grid

def learn_nerve_rootlets_distribution(subject_names, label_name_1, label_name_2, spline_input = 'centered', save_results=True):
    """
    Based on a subject name, learns empirical distribution of position
    of dorsal rootlets in cylindrical coordinates.
    """
    t1 = time.time()
    logger.info("Learning dorsal rootlets distribution.")
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        print 'Name: %s' % name
        try:
            nerves = load_image_array(name, label = True, sub_name = label_name_1)
            label_name = label_name_1
        except: 
            nerves = load_image_array(name, label = True, sub_name = label_name_2)
            label_name = label_name_2
            
        data = np.array(nerves == labels['dorsal rootlets'], dtype=np.float32) #Loads a '1' into an image array at all locations that rootlets were selected
    
        try:
            CP = load_pickle(name, 'spline_control_points_%s' % spline_input)
        except IOError:
            CP = load_pickle(name, 'spline_control_points_initial')
        
        s = spline.CRSpline()
        s.add_spline_points(CP.astype(np.float32))
        
        print 'Control Points: %s' % CP
        
        N = np.minimum(1000, int(10 * np.sqrt(np.sum((np.max(CP, axis=1) - np.min(CP, axis=1)) ** 2))))
        t = np.asarray(np.linspace(0, 1, N), np.float32)
        d = s.get_interpolated_spline_distance(t, volume_spacing_mm)
        M = 180 #Used to be 800
        theta = np.arange(M, dtype=np.float32) * 2 * np.pi / M
        
        cur_D = s.parse_spine_cylinder_coord(t, theta, data / data.max(), 1, 20, 1) #(t, theta, data, min radius, max radius, method) Searches along each value of t and theta to find the sum of the values along the radial. Method 3 = the sum of all the first moments (value / (r+1)) of all NRs (each has a value of 1)
        
        D_grid = spine_coord_to_grid(d, theta, cur_D) #Interpolates the values (i.e., number of NRs in each position) onto a standard grid
        
        plt.figure()
        plt.grid(True)
        plt.plot(spine_grid_d, np.mean(D_grid, axis=1))
        #plt.show()
        
        """
        Calculate the location of each of the nerve root regions
        """
        C3 = []
        C4 = []
        C5 = []
        C6 = []
        C7 = []
        C8 = []
        T1 = []
        individual_clusters = []
        individual_clusters.append(C3)
        individual_clusters.append(C4)
        individual_clusters.append(C5)
        individual_clusters.append(C6)
        individual_clusters.append(C7)
        individual_clusters.append(C8)
        individual_clusters.append(T1)
    
        labelled_rootlets = D_grid
        labelled_rootlets /= np.sum(labelled_rootlets) #Gives the % of the total labelled rootlets in each position of t and theta
        labelled_rootlets_d = np.sum(labelled_rootlets, axis=1) # gives the total % of the rootlets at each value of d
        
        r = np.round(labelled_rootlets_d * 100000).astype(np.int)
        points = np.repeat(spine_grid_d, r).reshape((-1, 1))
        
        non_zero_points = np.zeros((points.shape), dtype = np.float32)
        curr_NR = 0
        index = 0
        last_point = 0
        for point in points:
            if (point - last_point) > 1:
                curr_NR += 1
            non_zero_points[index] = curr_NR #Saves the NR index of the current point
            last_point = point
            index += 1
        
        count = 0
        #print 'New Clusters: ', np.max(non_zero_points)
        for i in range(np.max(non_zero_points)):
            p = points[non_zero_points == (i + 1)]
            individual_clusters[count].append(p) #Saves all of the points in each NR cluster across all subjects
            count += 1
    
    
        """
        Calculate min and max of individual NR clusters
        """
        cluster_min_max = []
        
        for cluster in individual_clusters:
            combined = []
            for sub_cluster in cluster:
                for item in sub_cluster:
                    combined.append(item)
            #print 'Combined: %s' % combined
            if len(combined) > 0:
                print 'NR Min: %s; Max: %s' % (np.min(combined), np.max(combined))
                cluster_min_max.append([np.min(combined), np.max(combined)])
        
        if save_results:
            plt.figure(figsize = (30,10))
            plt.grid(True)
            plt.title('Nerve Rootlet Markings - Subject %s' % name)
            plt.plot(spine_grid_d, np.mean(D_grid, axis=1), color = 'r', label = '%s' % label_name)
            ax = plt.gca()
            majorLocator = MultipleLocator(5)
            ax.xaxis.set_major_locator(majorLocator)
            plt.legend()
            plt.savefig('Nerve Rootlet Markings - subject %s - label %s.jpg' % (name, label_name))
            
            
            save_pickle(D_grid, 'analysis', '%s-rootlets_distribution-%s' % (name, label_name))
            save_pickle(cluster_min_max, 'analysis', '%s-rootlets_min_max-%s' % (name, label_name))

    print 'Done learning nerve rootlet distribution'
    
    t2 = time.time()
    logger.info("Done in %i:%.3f [mm:ss]" % (int((t2 - t1) / 60), (t2 - t1) % 60))


def inter_observer_rootlet_comparison(subject_names, label_name_1 = 'label', label_name_2 = 'label', save_results = False):
    all_D_1 = np.zeros((len(spine_grid_d), len(spine_grid_theta)))
    all_D_2 = np.zeros((len(spine_grid_d), len(spine_grid_theta)))
    
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        D_grid_1 = load_pickle('analysis', '%s-rootlets_distribution-%s' % (name, label_name_1))
        D_grid_2 = load_pickle('analysis', '%s-rootlets_distribution-%s' % (name, label_name_2))
        all_D_1 = D_grid_1 + all_D_1
        all_D_2 = D_grid_2 + all_D_2
        
        if save_results:
            plt.figure(figsize = (30,10))
            plt.grid(True)
            plt.title('Inter-Observer Nerve Rootlet Marking Comparison - Subject %s' % name)
            plt.plot(spine_grid_d, np.mean(D_grid_1, axis=1), color = 'r', label = '%s' % label_name_1)
            plt.plot(spine_grid_d, np.mean(D_grid_2, axis=1), color = 'b', label = '%s' % label_name_2)
            ax = plt.gca()
            majorLocator = MultipleLocator(5)
            ax.xaxis.set_major_locator(majorLocator)
            plt.legend()
            plt.savefig('inter-observer comparison - subject %s.jpg' % name)
    
    plt.figure(figsize = (30,10))
    plt.grid(True)
    plt.title('Inter-Observer Nerve Rootlet Marking Comparison - %s Subjects Combined' % len(subject_names))
    plt.plot(spine_grid_d, np.mean(all_D_1, axis=1), color = 'r', label = '%s' % label_name_1)
    plt.plot(spine_grid_d, np.mean(all_D_2, axis=1), color = 'b', label = '%s' % label_name_2)
    
    
    plt.legend()
    #plt.show()
    
    if save_results: plt.savefig('inter-observer comparison combined - %s subjects.jpg' % len(subject_names))
    else: plt.show() 
            

def learn_vertebrae_distribution(subject_names, label_name_1, label_name_2, spline_input = 'centered', save_results = True):
    """
    Creates a table of (a subjects) x (n vertebrae), showing the distance from the PMJ of the mid point of each vertebrae
    """ 
    landmark_list = ["C3-RB", "C3-CB", "C4-RB", "C4-CB", "C5-RB", "C5-CB", "C6-RB", "C6-CB", "C7-RB", "C7-CB"]
    
    distribution = np.zeros((len(subject_names), len(landmark_list) / 2))
    C3_mm_location = []
    
    counter = 0
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        individual_distribution_from_pmj = []
        individual_distribution_from_basion = []
        print 'Name is ', name
        try:
            landmarks = load_image_array(name, label = True, sub_name = label_name_1)
            label_name = label_name_1
        except:
            landmarks = load_image_array(name, label = True, sub_name = label_name_2)
            label_name = label_name_2
        
        CP = load_pickle(name, 'spline_control_points_%s' % spline_input)
        s = spline.CRSpline()
        s.add_spline_points(CP.astype(np.float32))
        
        basion_points = np.array(zip(*np.where(landmarks == labels['basion'])))
        basion_mean = np.mean(basion_points, axis=0)
        basion_mean_not_meaningful = basion_mean + 1
        print 'Mean basion point: %s' % basion_mean
        basion_points = np.vstack((basion_mean, basion_mean_not_meaningful)).astype(np.float32)
        
        N = 1000
        t = np.asarray(np.linspace(0, 1, N), np.float32)
            
        proj_t, proj_angles = s.find_spline_proj_point(t, volume_spacing_mm, basion_points)
        dist_mm = s.get_interpolated_spline_distance(proj_t, volume_spacing_mm)
        
        basion_mm = dist_mm[0]
        print 'distance to basion_centered: %s' % basion_mm
        
        #sleep(10)
        
        for n in range(len(landmark_list) / 2): 
            search_point_1 = np.array(zip(*np.where(landmarks == labels[landmark_list[n * 2]]))) 
            search_point_2 = np.array(zip(*np.where(landmarks == labels[landmark_list[(n * 2) + 1]])))

            P0 = np.mean(search_point_1, axis=0)
            P1 = np.mean(search_point_2, axis=0)
            print 'P0=', P0
            print 'P1=', P1
            P = np.vstack((P0, P1)).astype(np.float32)
            #print 'P', P
                
            proj_t, proj_angles = s.find_spline_proj_point(t, volume_spacing_mm, P)
            dist_mm = s.get_interpolated_spline_distance(proj_t, volume_spacing_mm)
            #Distance to the vertebrae from the basion is equal to distance from PMJ to vertebrae less distance from PMJ to basion
            print 'dist_mm array: %s' % dist_mm
            print 'Distance to rostral vertebral point (from PMJ / from basion): %s / %s' % (dist_mm[0], dist_mm[0] - basion_mm) 
            print 'Distance to caudal vertebral point (from PMJ / from basion): %s / %s' % (dist_mm[1], dist_mm[1] - basion_mm)
            dist_mm_0 = dist_mm[0] - basion_mm
            dist_mm_1 = dist_mm[1] - basion_mm
            avg_dist_mm = (dist_mm_0 + dist_mm_1) / 2
            distribution[counter, n] = avg_dist_mm
            individual_distribution_from_pmj.append([dist_mm[0], (dist_mm[0] + dist_mm[1]) / 2, dist_mm[1]])
            individual_distribution_from_basion.append([dist_mm_0, avg_dist_mm, dist_mm_1])
            if n == 0: #Selects the C3 body
                C3_mm_location.append([name, dist_mm_0, dist_mm_1])
        
        save_pickle(individual_distribution_from_pmj, 'analysis', '%s-vertebrae_distribution_from_pmj-%s' % (name, label_name))
        save_pickle(individual_distribution_from_basion, 'analysis', '%s-vertebrae_distribution_from_basion-%s' % (name, label_name))
        
        counter += 1
        
    if save_results:
        save_pickle(distribution, "analysis", 'vertebrae_distribution-%s' % label_name)
        save_pickle(C3_mm_location, "analysis", 'C3_vertebrae_locations-%s' % label_name)
     
    print 'Distribution Table = ', distribution
    print 'C3 Locations = ', C3_mm_location
    

def display_results(name):
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    data = load_image_array(name, label = False)
   

    CP1 = load_pickle(name, 'spline_control_points_centered')
   
    s = spline.CRSpline()
    s.add_spline_points(CP1.astype(np.float32))
   
    pts = s.get_interpolated_spline_point(np.linspace(0, 1, 500).astype(np.float32))
   
    sx, sy, sz = pts.T #.T is the transpose of the array

    (d, theta, D) = load_pickle(name, 'flattened_spine')

    nerves_image = load_label(name, 'label')
    
    nerves = itk.PyBuffer[LabelImageType].GetArrayFromImage(nerves_image)
    
    P = np.array(np.where(nerves == labels['dorsal rootlets'])).astype(np.float32).T
    
    t = np.asarray(np.linspace(0, 1, 800), np.float32)
    proj_t, proj_a, proj_d2s = s.find_spline_proj(t, volume_spacing_mm, P) #proj_t=relative distance down spine, proj_a=angle between each point and spline, proj_d2s=distance to the spline
    
    I = np.argsort(proj_t)
    proj_t = proj_t[I] #relative position down spine
    proj_a = proj_a[I] #Normal*Tangent
    proj_d2s = proj_d2s[I] #distance to spine
    #print 'proj_t=',proj_t
    proj_d = s.get_interpolated_spline_distance(proj_t, volume_spacing_mm)
    proj_P = s.get_interpolated_spline_point(proj_t)


    mlab.figure()
    mlab.plot3d(sx.T, sy.T, sz.T, color=(1.0, 0, 0), name='Spine', tube_radius=1) #This graphs the red spine
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                     name='MRI Data', colormap="jet")

    print 'Done displaying results'


def display_coordinate_system(subject_names, label_name = 'label'):
    
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        data = load_image_array(name, label = False)
        data2 = load_image_array(name, label = True, sub_name = 'surface_spine_interpolated')
        #data = itk.PyBuffer[InternalImageType].GetArrayFromImage(data_image)
    
        CP1 = load_pickle(name, 'spline_control_points_centered')
        s = spline.CRSpline()
        s.add_spline_points(CP1.astype(np.float32))
        pts = s.get_interpolated_spline_point(np.linspace(0, 1, 500).astype(np.float32))
        sx, sy, sz = pts.T #.T is the transpose of the array
    
        nerves_image = load_label(name, label_name)
        nerves = itk.PyBuffer[LabelImageType].GetArrayFromImage(nerves_image)
        
        spine_image = load_label(name, 'surface_spine_interpolated')
        spine = itk.PyBuffer[LabelImageType].GetArrayFromImage(spine_image)
        
        slice_image = load_label(name, 'coordinate_display')
        slices = itk.PyBuffer[LabelImageType].GetArrayFromImage(slice_image)
        
      
        mlab.figure()
        mlab.plot3d(sx.T, sy.T, sz.T, color=(1.0, 0, 0), name='Spine', tube_radius=1) #This graphs the red spine
        #mlab.plot3d(Px.T, Py.T, Pz.T, color=(1.0, 0, 0), name='Spine2', tube_radius=1) #This graphs the red spine
        #mlab.pipeline.volume(mlab.pipeline.scalar_field(data))
        
        #Show the circular slices - just the next line
        mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(slices), opacity = 0.3)
        #mlab.pipeline.volume(mlab.pipeline.scalar_field(data), vmin=0, vmax=0.6)#, colormap = 'black-white')
        
        #Sagitall - not used
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data), plane_orientation='z_axes', slice_index=256, colormap = 'black-white')
        
        #Coronal
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data), plane_orientation='x_axes', slice_index=100, colormap = 'black-white')
        
        #Transerse
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data), plane_orientation='y_axes', slice_index=100, colormap = 'black-white')
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data), plane_orientation='y_axes', slice_index=150, colormap = 'black-white')
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data), plane_orientation='y_axes', slice_index=200, colormap = 'black-white')
        mlab.show()
    
        print 'Done displaying results'


def vertebrae_rootlet_correlation(subject_names, label_name = 'label', measure_vertebrae_from = 'pmj'):
    vertebrae_distances = []
    nerve_root_distances = []
    all_spinal_levels = ['C4', 'C5', 'C6', 'C7', 'C8']
    all_vertebrae_levels = ['C3', 'C4', 'C5', 'C6', 'C7']
    constant_level_difference = []
    
    for i in range(len(all_spinal_levels)):
        vertebrae_distance = 0
        nerve_root_distance = 0
        temp_difference = []
        for name in subject_names:
            volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
            
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name))
            regions = profile.get_all_regions()
            for region in regions:
                if region[0].find(all_vertebrae_levels[i] + ' VB') != -1 and region[4] == measure_vertebrae_from:
                    #print 'Distance to %s: %s' % (constant_level + ' VB', region[2])
                    vertebrae_distance = region[2] #Finds the average level of the vertebral body
                elif region[0].find(all_spinal_levels[i] + ' NR') != -1:
                    nerve_root_distance = region[2] #Finds the average level of the vertebral body
            
            temp_difference.append(vertebrae_distance - nerve_root_distance)
        constant_level_difference.append(temp_difference)
            
    
    
    plt.figure(figsize=(15, 15), dpi=100)
    count = 0
    for distance_difference in constant_level_difference:
        x = [count] * len(distance_difference)
        SD = np.sqrt(np.var(distance_difference))
        AVG = np.mean(distance_difference)
        plt.scatter(x, distance_difference, s = 10)
        #plt.text(count, int(np.max(distance_difference)) + 1, 'SD: %2f' % SD)
        plt.text(count - 0.25, int(np.max(distance_difference)) + 1, 'Avg:\n %2f' % AVG)
        count += 1
    
    x = np.arange(0, len(all_vertebrae_levels), 1)
    y = [6] * len(all_vertebrae_levels)
    plt.plot(x, y, linestyle = '--', color = 'r')
    y = [-6] * len(all_vertebrae_levels)
    plt.plot(x, y, linestyle = '--', color = 'r')
    
    x_ticks = np.arange(0, len(all_vertebrae_levels), 1)
    plt.xticks(x_ticks, all_vertebrae_levels)
    plt.title('Distance Between Vertebrae (mid-point) and Spinal Segment (mid-point)')
    plt.xlabel('Vertebrae / Spinal Segment')
    plt.ylabel('Distance Between Vertebrae and Spinal Segment(mm)')
    plt.savefig('Distance Between Vertebrae and Corresponding Spinal Level - Label: %s' % label_name)
    plt.show()
    plt.close()
    


class SubjectProfile:
    def __init__(self, name):
        self.regions = []
        self.neck_height = 0
        self.total_height = 0
        self.sex = ''
        self.age = 0
        self.name = name
        print 'New profile created'
    
    def add_region(self, name, min, average, max, note = ''):
        self.regions.append([name, min, average, max, note])
    
    def add_neck_height(self, height):
        self.neck_height = height
        
    def add_total_height(self, height):
        self.total_height = height
        
    def add_sex(self, sex):
        self.sex = sex
    
    def add_age(self, age):
        self.age = age
    
    def get_all_regions(self):
        return self.regions
    
    def get_region(self, name, loc_type = ''):
        for region in self.regions:
            if region[0] == name:
                if loc_type == '' or (loc_type != '' and region[4] == loc_type):
                    return region 
    
    def get_specific_regions(self, input_regions, loc = 'mid'):
        output_regions = []
        for in_reg in input_regions:
            if in_reg == 'total_height':
                output_regions.append(self.total_height)
            elif in_reg == 'neck_height':
                output_regions.append(self.neck_height)
            elif in_reg == 'sex':
                output_regions.append(self.sex)
            elif in_reg == 'age':
                output_regions.append(self.age)
            else:
                if loc == 'mid':
                    output_regions.append(self.get_region(in_reg)[2])
                elif loc == 'rost':
                    output_regions.append(self.get_region(in_reg)[1])
                elif loc == 'caud':
                    output_regions.append(self.get_region(in_reg)[3])
                    
        return output_regions
    
    
    def illustrate_profile(self, figure, x_location, width, width_per_subject, measure_vertebrae_from = 'pmj', place_labels = False): #measure from works for vertebrae only
        count = 0
        colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
        plots = []
        nr_to_graph = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
        #plt.figure(figsize=(6, 15), dpi=100)
        lowest_level = 1000
        
        vb_count = 0
        for region in self.regions:
            if region[0].find('VB') != -1 and region[4] == measure_vertebrae_from:
                if place_labels and vb_count == 0:
                    plots.append(figure.bar(left = (x_location * width_per_subject), height = (region[3] - region[1]), width = width, bottom = -region[3], alpha = 0.2, color = 'y', label = 'C3-C7 Vertebral Bodies'))
                else:
                    print '%s: %s - %s' % (region[0], region[1], region[3])
                    plots.append(figure.bar(left = (x_location * width_per_subject), height = (region[3] - region[1]), width = width, bottom = -region[3], alpha = 0.2, color = 'y'))
                
                if place_labels: #This places the VB labels on the first set of vertebrae markers
                    font = 14
                    figure.text((x_location * width_per_subject + 0.15 * width), (-region[3] + 0.6 * (region[3] - region[1])), '%s' % region[0][:2], color = 'k', fontsize = font)
                    figure.text((x_location * width_per_subject + 0.15 * width), (-region[3] + 0.4 * (region[3] - region[1])), 'VB', color = 'k', fontsize = font)
                    
                if region[1] < lowest_level: lowest_level = region[1] #Looks for the highest level in order to set the y-axis level
                print 'lowest_level:', lowest_level
                vb_count += 1 
            elif region[0].find('NR') != -1 and region[0] in nr_to_graph: #Must be a NR
                if place_labels:
                    plots.append(figure.bar(left = (x_location * width_per_subject + width), height = (region[3] - region[1]), width = width, bottom = -region[3], alpha = 0.5, color= colors[count % len(colors)], label = '%s %s' % (region[0][0:2], ' Nerve Segment')))
                else:
                    plots.append(figure.bar(left = (x_location * width_per_subject + width), height = (region[3] - region[1]), width = width, bottom = -region[3], alpha = 0.5, color= colors[count % len(colors)]))
                count += 1
        
        count = 0
        for plot in plots:
            for rect in plot:
                bottom_height = rect.get_y()
                top_height = rect.get_height() + bottom_height
                total_height = math.fabs(bottom_height - top_height)
                left = rect.get_x()
                width = rect.get_width()
                #figure.text(left + width / 3, bottom_height + 2, '%d' % int(bottom_height), color = 'w')
                #figure.text(left + width / 3, top_height - 2, '%d' % int(top_height), color = 'w')
                #figure.text(left + width / 3, bottom_height + 0.5 * total_height, '%s' % self.regions[count][0], color = 'w')
                count += 1
        
        
    def sort_data(self):
        print 'Before Sorting: %s' % self.regions
        sorted_by_average = sorted(self.regions, key=lambda tup: tup[2])
        self.regions = sorted_by_average
        print 'After Sorting: %s' % self.regions
        
    def display_profile(self):
        for region in self.regions:
            print region
        

def calculate_relative_vertebrae_nr_distance(subject_names, label_name1, label_name2 = '', vertebrae_level = 'C5', nr_level = 'C3', save_results = True, measure_vertebrae_from = 'pmj'):
    gap_distances1 = []
    gap_distances2 = []
    gap_differences = []
    
    count = 0
    start_distance = ''
    end_distance = ''
    for name in subject_names: 
        try:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name1))
        except IOError:
            profile = SubjectProfile(name)
            
        regions = profile.get_all_regions()
        for region in regions:
            if region[0].find(vertebrae_level + ' VB') != -1 and region[4] == measure_vertebrae_from:
                print 'Distance to %s: %s' % (vertebrae_level + ' VB', region[2])
                start_distance = region[2]
                #start_distances.append([name, region[2]]) #Finds the average level of the vertebral body
            
            if region[0].find(nr_level + ' NR') != -1:
                print 'Distance to %s: %s' % (nr_level + ' NR', region[2])
                end_distance = region[2]
                #end_distances.append([name, region[2]]) #Finds the average level of the nr segment
        
        print 'Name: %s; Label: %s; Gap Distance: %0.4f' % (name, label_name1, (start_distance - end_distance))
        gap_distances1.append(start_distance - end_distance)
        start_distance = ''
        end_distance = ''
        
        if label_name2 != '':
            try:
                profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name2))
            except IOError:
                profile = SubjectProfile(name)
                
            regions = profile.get_all_regions()
            for region in regions:
                if region[0].find(vertebrae_level + ' VB') != -1 and region[4] == measure_vertebrae_from:
                    print 'Distance to %s: %s' % (vertebrae_level + ' VB', region[2])
                    start_distance = region[2]
                    #start_distances.append([name, region[2]]) #Finds the average level of the vertebral body
                
                if region[0].find(nr_level + ' NR') != -1:
                    print 'Distance to %s: %s' % (nr_level + ' NR', region[2])
                    end_distance = region[2]
                    #end_distances.append([name, region[2]]) #Finds the average level of the nr segment
            
            print 'Name: %s; Label: %s; Gap Distance: %0.4f' % (name, label_name2, (start_distance - end_distance))
            gap_distances2.append(start_distance - end_distance)
            
            gap_differences.append(gap_distances1[count] - gap_distances2[count])
            start_distance = ''
            end_distance = ''
        
        count += 1
        
        
    print 'Mean / min / max / (max - min) of distances from %s VB to %s NR for %s markings: %0.4f / %0.4f / %0.4f / %0.4f' % (vertebrae_level, nr_level, label_name1, np.mean(gap_distances1), np.min(gap_distances1), np.max(gap_distances1), np.max(gap_distances1) - np.min(gap_distances1))
    if label_name2 != '':
        print 'Mean / min / max of distances from %s VB to %s NR for %s markings: %0.4f / %0.4f / %0.4f' % (vertebrae_level, nr_level, label_name2, np.mean(gap_distances2), np.min(gap_distances2), np.max(gap_distances2))
        print 'Mean / min / max of difference in distance from %s VB to %s NR between %s and %s markings: %0.4f / %0.4f / %0.4f' % (vertebrae_level, nr_level, label_name1, label_name2, np.mean(gap_differences), np.min(gap_differences), np.max(gap_differences))
            
    if label_name2 != '':
        print 'All gap differences for level %s' % nr_level
        for gap in gap_differences:
            print gap[0] 
    else:
        print 'All gap differences for level %s' % nr_level
        print (np.max(gap_distances1) - np.min(gap_distances1))


def calculate_inter_marker_error(subject_names, label_name1, label_name2, measure_vertebrae_from = 'pmj'):
    
    gap_all_levels = []
    
    
    start_distance = ''
    end_distance = ''
    
    levels = ['C3', 'C4', 'C5', 'C6', 'C7']
    
    
    for level in levels:
        count = 0
        gap_distances1 = []
        gap_distances2 = []
        gap_differences = []
        for name in subject_names: 
            try:
                profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name1))
            except IOError:
                profile = SubjectProfile(name)
                
            regions = profile.get_all_regions()
            for region in regions:
                if region[0].find(level + ' VB') != -1 and region[4] == measure_vertebrae_from:
                    print 'Distance to %s: %s' % (level + ' VB', region[2])
                    start_distance = region[2]
                    #start_distances.append([name, region[2]]) #Finds the average level of the vertebral body
                
                if region[0].find(level + ' NR') != -1:
                    print 'Distance to %s: %s' % (level + ' NR', region[2])
                    end_distance = region[2][0]
                    #end_distances.append([name, region[2]]) #Finds the average level of the nr segment
            
            print 'Name: %s; Label: %s; Gap Distance: %0.4f' % (name, label_name1, np.abs(start_distance - end_distance))
            gap_distances1.append(np.abs(start_distance - end_distance))
            start_distance = ''
            end_distance = ''
            
            
            try:
                profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name2))
            except IOError:
                profile = SubjectProfile(name)
                
            regions = profile.get_all_regions()
            for region in regions:
                if region[0].find(level + ' VB') != -1 and region[4] == measure_vertebrae_from:
                    print 'Distance to %s: %s' % (level + ' VB', region[2])
                    start_distance = region[2]
                    #start_distances.append([name, region[2]]) #Finds the average level of the vertebral body
                
                if region[0].find(level + ' NR') != -1:
                    print 'Distance to %s: %s' % (level + ' NR', region[2])
                    end_distance = region[2][0]
                    #end_distances.append([name, region[2]]) #Finds the average level of the nr segment
            
            print 'Name: %s; Label: %s; Gap Distance: %0.4f' % (name, label_name2, np.abs(start_distance - end_distance))
            gap_distances2.append(np.abs(start_distance - end_distance))
            
            gap_differences.append(np.abs(gap_distances1[count] - gap_distances2[count]))
            start_distance = ''
            end_distance = ''
            
            count += 1
        gap_all_levels.append(gap_differences)
    
    count = 0
    for level in gap_all_levels:
        print 'Level: %s; Mean: %s; Std. Dev.: %s; All Values: %s' % (levels[count], np.mean(level), np.std(level), level)
        count += 1
        

def calculate_flex_extend_error(subject_names, label_name1, calculate_nr_distance_only = False, measure_vertebrae_from = 'pmj'):
    
    start_distance = ''
    end_distance = ''
    
    levels = ['C3', 'C4', 'C5', 'C6', 'C7']
    neck_position = ['E', 'F']
    
    for position in neck_position:
        gap_all_levels = []
        for level in levels:
            count = 0
            gap_distances1 = []
            gap_distances2 = []
            gap_differences = []
            for name in subject_names: 
                try:
                    profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name1))
                except IOError:
                    profile = SubjectProfile(name)
                    
                regions = profile.get_all_regions()
                for region in regions:
                    if region[0].find(level + ' VB') != -1 and region[4] == measure_vertebrae_from:
                        print 'Distance to %s: %s' % (level + ' VB', region[2])
                        start_distance = region[2]
                        #start_distances.append([name, region[2]]) #Finds the average level of the vertebral body
                    
                    if region[0].find(level + ' NR') != -1:
                        print 'Distance to %s: %s' % (level + ' NR', region[2])
                        end_distance = region[2][0]
                        #end_distances.append([name, region[2]]) #Finds the average level of the nr segment
                
                if calculate_nr_distance_only: start_distance = 0
                print 'Name: %s; Label: %s; Gap Distance: %0.4f' % (name, label_name1, np.abs(start_distance - end_distance))
                gap_distances1.append(np.abs(start_distance - end_distance))
                start_distance = ''
                end_distance = ''
                
                try:
                    profile = load_pickle('analysis', '%s%s-profile-%s' % (name, position, label_name1))
                except IOError:
                    profile = SubjectProfile(name)
                    
                regions = profile.get_all_regions()
                for region in regions:
                    if region[0].find(level + ' VB') != -1 and region[4] == measure_vertebrae_from:
                        print 'Distance to %s: %s' % (level + ' VB', region[2])
                        start_distance = region[2]
                        #start_distances.append([name, region[2]]) #Finds the average level of the vertebral body
                    
                    if region[0].find(level + ' NR') != -1:
                        print 'Distance to %s: %s' % (level + ' NR', region[2])
                        end_distance = region[2][0]
                        #end_distances.append([name, region[2]]) #Finds the average level of the nr segment
                
                if calculate_nr_distance_only: start_distance = 0
                print 'Name: %s; Label: %s; Gap Distance: %0.4f' % (name, label_name1, np.abs(start_distance - end_distance))
                gap_distances2.append(np.abs(start_distance - end_distance))
                
                gap_differences.append(np.abs(gap_distances1[count] - gap_distances2[count]))
                start_distance = ''
                end_distance = ''
                
                count += 1
            gap_all_levels.append(gap_differences)
    
        count = 0
        print 'Position Type: %s' % position
        if calculate_nr_distance_only: print 'WARNING: CALCULATION OF NR DISTANCE ONLY, NOT DIFFERENCE BETWEEN NR AND VB'
        for level in gap_all_levels:
            print 'Level: %s; Mean: %s; Std. Dev.: %s; All Values: %s' % (levels[count], np.mean(level), np.std(level), level)
            count += 1

    

def aggregate_spinal_cord_illustration(subject_names, label_name, constant_level = 'C3', measure_vertebrae_from = 'pmj'):
    num_columns = len(subject_names)
    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
    width = 0.7
    plots = []
    subject_labels = []
    gap_distances = []
    
    start_distances = []
    for name in subject_names: 
        try:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name))
        except IOError:
            profile = SubjectProfile(name)
            
        regions = profile.get_all_regions()
        for region in regions:
            if region[0].find(constant_level + ' VB') != -1 and region[4] == measure_vertebrae_from:
                print 'Distance to %s: %s' % (constant_level + ' VB', region[2])
                start_distances.append([name, region[2]]) #Finds the average level of the vertebral body
    
 
    plt.figure(figsize=(10, 10), dpi=150)
    subject_counter = 1
    for name in subject_names: 
        subject_labels.append(name)
        profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name))
        regions = profile.get_all_regions()
        for start_dist in start_distances:
            if start_dist[0] == name:
                curr_start_dist = start_dist[1]
        
        count = 0
        for region in regions:
            if region[0].find('NR') != -1:
                gap = curr_start_dist - region[3] #Subtract the average distance from the PMJ to the landmark from the start position
                gap_distances.append(gap)
                plots.append(plt.bar(left = subject_counter - 1, height = (region[3] - region[1]), width = width, bottom = gap, color = colors[count % len(colors)]))
                count += 1
                
        subject_counter += 1
        
   
    x_values = np.arange(0, num_columns, 1)
    upper_y = round_partial(np.max(gap_distances), 10)
    lower_y = round_partial(np.min(gap_distances), 10)
    
    y_values = np.arange(lower_y - 10, upper_y + 20,5)
    #y_values = np.arange(-120,-40,10)
    plt.yticks(y_values)
    plt.xticks(x_values + width / 2, subject_labels, rotation = 45)
    plt.title('Nerve Rootlet Average Distance Above / (Below) Fixed Vertebral Level: %s' % constant_level)
    plt.ylabel('Distance from the %s vertebral level (mm)' % constant_level)
    plt.xlabel('Subject Number')
    plt.grid(True)
    plt.savefig('Illustrated NR Relative to Constant %s Vertebral Level' % constant_level)
    plt.show()
            

def round_partial(value, resolution):
        return round (value / resolution) * resolution
    

def update_subject_profiles(subject_names, label_name_1, label_name_2, show_graph = False, analyze_rootlets = True):
    
    cluster_names = ['C3 NR','C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR', 'T1 NR']
    vertebrae_names = ['C3 VB','C4 VB', 'C5 VB', 'C6 VB', 'C7 VB']
    subject_neck_heights = load_pickle('analysis', 'subject_neck_heights')
    subject_total_heights = load_pickle('analysis', 'subject_total_heights')
    subject_sex = load_pickle('analysis', 'subject_sex')
    subject_age = load_pickle('analysis', 'subject_age')
    
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        label_name = ''

        profile = SubjectProfile(name)
            
        """
        Analyze Rootlets
        """
        if analyze_rootlets:
            try:
                labelled_rootlets = load_pickle('analysis', '%s-rootlets_distribution-%s' % (name, label_name_1))
                label_name = label_name_1
            except:
                labelled_rootlets = load_pickle('analysis', '%s-rootlets_distribution-%s' % (name, label_name_2))
                label_name = label_name_2
                
            labelled_rootlets /= np.sum(labelled_rootlets) #Gives the % of the total labelled rootlets in each position of t and theta
            labelled_rootlets_d = np.sum(labelled_rootlets, axis=1) # gives the total % of the rootlets at each value of d
            
            r = np.round(labelled_rootlets_d * 100000).astype(np.int)
            points = np.repeat(spine_grid_d, r).reshape((-1, 1))
            
            if show_graph:
                plt.figure()
                plt.imshow(labelled_rootlets, origin='lower', aspect=spine_grid_theta[-1] / spine_grid_d[-1], extent=(spine_grid_theta[0], spine_grid_theta[-1],spine_grid_d[0], spine_grid_d[-1]))
                plt.title('Nerve Rootlet Distribution of Subject %s' % name)
                plt.xlabel('Relative Spinal Cord Rotation (rad)')
                plt.ylabel('Relative Spinal Cord Distance from PMJ (mm)')
                #plt.colorbar(format="%.1e")
                #plt.show()
                plt.savefig('%s NR Distribution.jpg' % name)
                
            
            #K_means_method, value of K is loaded from the "load_image_characteristics" function
            np.random.seed(0)
            k_means = KMeans(init = 'k-means++', n_clusters = K, n_init = 10) #k=number of clusters, n_init=number of times algo will run with different centroid seeds
            k_means.fit(points)
            k_means_labels_unique = np.unique(k_means.labels_)
            
            count = 0
            for i in np.argsort(k_means.cluster_centers_.squeeze()):
                p = points[k_means.labels_ == i]
                profile.add_region(cluster_names[count], np.min(p), k_means.cluster_centers_[i], np.max(p), note = 'kmeans')
                print 'K-Means %s; Min: %s; Average: %s; Max: %s' % (cluster_names[count], np.min(p), k_means.cluster_centers_[i], np.max(p))
                count += 1    
            
        #Find the subjects neck height in the database
        for subject in subject_neck_heights:
            if name == str(subject[0]):
                profile.add_neck_height(subject[1])
        
        #Find the subjects neck height in the database
        for subject in subject_total_heights:
            if name == str(subject[0]):
                profile.add_total_height(subject[1])
        
        #Find the subjects sex in the database
        for subject in subject_sex:
            if name == str(subject[0]):
                profile.add_sex(subject[1])
        
        #Find the subjects age in the database
        for subject in subject_age:
            if name == str(subject[0]):
                profile.add_age(subject[1])
        
        """
        Analyze Vertebrae
        """
        #if label_name == '': #Run this in case the nerve rootlets weren't analyzed yet
        try:
            distribution = load_pickle('analysis', '%s-vertebrae_distribution_from_pmj-%s' % (name, label_name_1))
            label_name = label_name_1
        except:
            distribution = load_pickle('analysis', '%s-vertebrae_distribution_from_pmj-%s' % (name, label_name_2))
            label_name = label_name_2
        
        count = 0
        for level in distribution:
            print 'Level: %s; Mean: %s' % (vertebrae_names[count], level[1])
            profile.add_region(vertebrae_names[count], level[0], level[1], level[2], note = 'pmj')
            count += 1
        
        distribution = load_pickle('analysis', '%s-vertebrae_distribution_from_basion-%s' % (name, label_name))
        
        count = 0
        for level in distribution:
            profile.add_region(vertebrae_names[count], level[0], level[1], level[2], note = 'basion')
            count += 1
        
        #Sort the data points
        profile.sort_data()
        #profile.illustrate_profile()
        
        save_pickle(profile, 'analysis', '%s-profile-%s' % (name, label_name))
    
    
def display_full_profiles(subject_names, label_name_1, label_name_2, save_results = True, output_name = ''):
    plt.figure(figsize=(13, 10), dpi=100)
    ax = plt.subplot(111)
    plt.legend()
    
    #width = 0.9
    width = 4
    total_width_per_subject = 18
    x_values = []
    
    count = 0
    for name in subject_names:
        try:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
        except:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
        
        if count == 0:
            profile.illustrate_profile(ax, count, width, total_width_per_subject, measure_vertebrae_from = 'pmj', place_labels = True) #measure from works for vertebrae only
        else:
            profile.illustrate_profile(ax, count, width, total_width_per_subject, measure_vertebrae_from = 'pmj')
        #x_values.append(name)
        x_values.append('%s' % (count + 1))
        for a in range(total_width_per_subject-1):
            x_values.append('')
        
        count += 1
    
    plt.title('Vertebrae and Nerve Rootlet Layout (N = %s Subjects)' % len(subject_names))
    plt.ylabel('Distance from PMJ (PMJ = 0mm)')
    plt.xlabel('Vertebrae / Nerve Rootlets')
    x_ticks = np.arange(0, len(subject_names) * total_width_per_subject, 1)
    plt.xticks(x_ticks)
    y_values = np.arange(-160,-40,10)
    plt.yticks(y_values)
    plt.xticks(x_ticks + width / 2, x_values, rotation = 45)
    legend = ax.legend(loc='lower right', bbox_to_anchor=(0.955, 0.0), shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('small')
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    if save_results: plt.savefig('Figure 5 - Illustrated VB and NR Subject of %s Subjects_%s' % (len(subject_names), output_name))
    plt.show()
    plt.close()


def display_rootlet_or_vertebrae_lengths(subject_names, level_type, label_name_1, label_name_2, save_results = True, measure_vertebrae_from = 'pmj'):
    
    if level_type == 'Vertebrae':
        spine_levels = ['C3 VB', 'C4 VB', 'C5 VB', 'C6 VB', 'C7 VB']
    elif level_type == 'Nerve Rootlets':
        spine_levels = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
    else:
        print 'Incorrect entry for level_type: %s. Please try again.' % level_type
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
    f.set_size_inches(20,15)
    
    means = []
    variances = []
    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
    height = 0.3
    
    count = len(spine_levels)
    count_up = 0
    all_lengths = []
    for level in spine_levels:
        temp_lengths = []
        
        for name in subject_names:
            print name
            try:
                profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
            except:
                profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
            
            regions = profile.get_all_regions()
            for region in regions:
                if region[0].find(level) != -1 and (region[4] == measure_vertebrae_from if level_type == 'Vertebrae' else True):
                    temp_lengths.append(region[3] - region[1])
                    all_lengths.append(region[3] - region[1])
        
        means.append(np.mean(temp_lengths))
        variances.append(np.sqrt(np.var(temp_lengths)))      
        
        ax2.bar(left = np.min(temp_lengths), height = height, width = (np.max(temp_lengths) - np.min(temp_lengths)), bottom = count - height / 2, color = colors[count_up % len(colors)])
        print count
        count -= 1
        count_up += 1            
    
    print 'Number of Samples: %s; Average Length: %s; St. Dev: %s; Min: %s; Max: %s' % (len(all_lengths), np.mean(all_lengths), np.sqrt(np.var(all_lengths)), np.min(all_lengths), np.max(all_lengths))
    x_range = np.arange(0,25,1)
    plot_gaussians(x_range, means, variances, spine_levels, ax1, linewidth=2, linestyle = '-')
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Length of %s (mm)' % level_type)
    ax1.set_title('Figure 2a: Average Length (+/- Standard Deviation) of %s - %s Subjects' % (level_type, len(subject_names)))
    ax1.grid(True)
    ax1.legend(loc='upper right')
    ax2.set_title('Figure 2b: Range of %s Lengths - %s Subjects' % (level_type, len(subject_names)))
    spine_levels.append('')
    spine_levels.reverse()
    y_ticks = [a for a in range(len(spine_levels))]
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(spine_levels)
    #ax2.set_xticks(spine_grid_d)
    ax2.set_xlabel('Length (mm)')
    ax2.grid(True)
    plt.show()
    if save_results: 
        if level_type == 'Vertebrae': plt.savefig('Figure x - %s Lengths.jpg' % level_type, dpi = 200) 
        elif level_type == 'Nerve Rootlets': plt.savefig('Figure x - %s Lengths.jpg' % level_type, dpi = 200)
    

def display_spine_distribution(subject_names, level_type, label_name_1, label_name_2, save_results = True, output_name = '', measure_vertebrae_from = 'pmj'):
    
    if level_type == 'Vertebrae':
        v_levels = ['C3 VB', 'C4 VB', 'C5 VB', 'C6 VB', 'C7 VB']
        nr_levels = []
    elif level_type == 'Nerve Rootlets':
        nr_levels = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
        v_levels = []
    elif level_type == 'both': 
        v_levels = ['C3 VB', 'C4 VB', 'C5 VB', 'C6 VB', 'C7 VB']
        nr_levels = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
    else:
        print 'Incorrect entry for level_type: %s. Please try again.' % level_type
        
     
    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
    height = 0.3
    height_output = []
    height_err = []
    
    #f, (ax1, ax2) = plt.subplots(2,1, sharex = False)
    f, ax1 = plt.subplots(1,1, sharex = False)
    f.set_size_inches(20,10)
    
    for a in range(2):
        if a == 0:
            spine_levels = v_levels
        elif a == 1:
            spine_levels = nr_levels
        
        count = len(spine_levels)
        count_up = 0
        min_location = []
        max_location = []
        means = []
        variances = []
        
        for level in spine_levels:
            temp_means = []
            min_location_temp = []
            max_location_temp = []
            temp_lengths = []
            for name in subject_names:
                try:
                    profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
                except:
                    profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
                
                regions = profile.get_all_regions()
                for region in regions:
                    if region[0].find(level) != -1 and (region[4] == measure_vertebrae_from if level.find('VB') != -1 else True):
                        min_location_temp.append(region[1])
                        max_location_temp.append(region[3])
                        temp_means.append(region[2])
                        temp_lengths.append(region[3] - region[1])
                        if level == 'C3 VB':
                            print 'Name: %s; Level: %s; Distance: %s' % (name, level, region[2])
                        
            means.append(np.mean(temp_means))
            variances.append(np.sqrt(np.var(temp_means)))
            height_output.append(np.mean(temp_lengths))
            height_err.append(np.sqrt(np.var(temp_lengths)))
            min_location.append(np.min(min_location_temp))
            max_location.append(np.max(max_location_temp))
            
            #ax2.errorbar(count_up + 0.5, np.mean(temp_lengths), yerr = np.sqrt(np.var(temp_lengths)), fmt = 'x', color = colors[count_up % len(colors)], linewidth = 2, label = '%s %.2f (+/- %.2f)' % (level, np.mean(temp_lengths), np.sqrt(np.var(temp_lengths))))
            
            count -= 1
            count_up += 1
        
        x_values = np.arange(0.5, len(spine_levels) + 0.5, 1)
        
        #ax2.errorbar(x_values, height_output, yerr = height_err, fmt = 'o', linewidth = 2)
        print means
        
        if a == 0:
            plot_gaussians_with_range(spine_grid_d, means, variances, min_location, max_location, spine_levels, ax1, linewidth=3, linestyle = '--')
        elif a == 1:
            plot_gaussians_with_range(spine_grid_d, means, variances, min_location, max_location, spine_levels, ax1, linewidth=3, linestyle = '-')
        
        count = 0
        for level in spine_levels:
            print '%s: Min: %s; Max: %s' % (level, min_location[count], max_location[count])
            count += 1
    
    ax1.set_ylabel('Probability', fontsize = 20)
    ax1.set_xlabel('Distance from PMJ (mm)', fontsize = 20)
    ax1.set_title('Average Distance (+/- Standard Deviation, Minimum, Maximum) from the PMJ (N = %s Subjects)' % (len(subject_names)), fontsize = 20)
    ax1.grid(True)
    ax1.legend(loc='upper right')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    ax1.legend(loc=1,prop={'size':26})
    
    
    if save_results: 
        if level_type == 'Vertebrae': plt.savefig('Figure 2 - %s Distribution%s.jpg' % (level_type, ' ' + output_name), dpi = 200) 
        elif level_type == 'Nerve Rootlets': plt.savefig('Figure 3 - %s Distribution%s.jpg' % (level_type, ' ' + output_name), dpi = 200)
        elif level_type == 'both': plt.savefig('Figure 3 - %s Distribution%s.jpg' % ('Spinal Cord', ' ' + output_name), dpi = 200)
    else:
        plt.show()


def display_nerve_rootlet_offset(subject_names, label_name_1, label_name_2, v1_level = 'C4', v2_level = 'C5', n1_level = 'C5', save_results = True, measure_vertebrae_from = 'pmj'):
    
    v_levels = [v1_level, v2_level]
    v_centers_1 = []
    v_centers_2 = []
    v_lengths_1 = []
    v_lengths_2 = []
    overlaps = []
    v_normalized = 10
    fig, ax1 = plt.subplots()
    fig.set_size_inches(15,10)
    
    count = 0
    for counter in range(len(v_levels)):
        overlap = 0
        total_length = 0
        
        for name in subject_names:
            try:
                profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
            except:
                profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
            
            regions = profile.get_all_regions()
            
            
            for region in regions:
                    if region[0].find(v_levels[counter]) != -1 and region[4] == measure_vertebrae_from:
                        min_location_v = region[1]
                        max_location_v = region[3]
                        mean_location_v = region[2]
                        
                        if counter == 0:
                            v_centers_1.append(mean_location_v)
                            v_lengths_1.append(max_location_v - min_location_v)
                        elif counter ==1:
                            v_centers_2.append(mean_location_v)
                            v_lengths_2.append(max_location_v - min_location_v)
                    elif region[0].find(n1_level + ' NR') != -1:
                        min_location_n = region[1]
                        max_location_n = region[3]
                        mean_location_n = region[2]
            
            curr_length = (max_location_n - min_location_n)
            total_length += curr_length
            
            if min_location_n < min_location_v and max_location_n < max_location_v and max_location_n > min_location_v:
                curr_overlap = (max_location_n - min_location_v)
                print '%s: Top Overlap: %d%%' % (name, (100.0 * curr_overlap / curr_length))
                overlap += (max_location_n - min_location_v)
            elif min_location_n > min_location_v and max_location_n > max_location_v and min_location_n < max_location_v:
                curr_overlap = (max_location_v - min_location_n)
                print '%s: Bottom Overlap: %d%%' % (name, (100.0 * curr_overlap / curr_length))
                overlap += (max_location_v - min_location_n)
            elif min_location_n > min_location_v and max_location_n < max_location_v:
                curr_overlap = (max_location_n - min_location_n)
                print '%s: Engulfed: %d%%' % (name, (100.0 * curr_overlap / curr_length))
                overlap += (max_location_n - min_location_n)
            elif min_location_n < min_location_v and max_location_n > max_location_v:
                curr_overlap = (max_location_v - min_location_v)
                print '%s: Full Vertebrae Overlap: %d%%' % (name, (100.0 * curr_overlap / curr_length))
                overlap += (max_location_v - min_location_v)
            else:
                print '%s: No Overlap' % (name)
            
            
            if counter == 0:
                scale_factor_length = (max_location_n - min_location_n) / (max_location_v - min_location_v)
                scale_factor_center = (mean_location_v - mean_location_n) / (max_location_v - min_location_v) 
                relative_center_n = v_normalized * scale_factor_center
                relative_length_n = v_normalized * scale_factor_length
                print '%s; Vertebrae Length: %s; NR Length: %s; Difference Between Vertebrae and NR Centers: %s' % (name, (max_location_v - min_location_v), (max_location_n - min_location_n), (mean_location_v - mean_location_n))
                print '%s; Scale Factor Length: %s; Relative NR Center: %s; Scaled NR Length: %s' % (name, scale_factor_length, relative_center_n, relative_length_n)
                
                ax1.bar(left = count + 0.25, height = relative_length_n, width = 0.5, bottom = relative_center_n - relative_length_n / 2, alpha = 1, color = 'r')
                count += 1
        
        overlap_percent = overlap * 1.0 / total_length * 100
        overlaps.append(overlap_percent)
        
    v_center_scale_factors = []
    v_length_scale_factors = []
    for a in range(len(v_centers_1)):
        v_center_scale_factors.append((v_centers_1[a] - v_centers_2[a]) / v_lengths_2[a])
        v_length_scale_factors.append(v_lengths_1[a] / v_lengths_2[a])
        
    v_scale_factor_length = np.mean(v_length_scale_factors)
    v_scale_factor_center = np.mean(v_center_scale_factors)
    relative_v1_length = v_normalized * v_scale_factor_length
    relative_v1_center =  v_normalized * v_scale_factor_center
    
    ax1.bar(left = 0, height = v_normalized, width = count, bottom = - v_normalized / 2, alpha = 0.2, color = 'y')
    #ax1.bar(left = 1.25, height = v_normalized, width = 0.5, bottom = - v_normalized / 2, color = 'g')
    ax1.bar(left = 0, height = relative_v1_length, width = count, bottom = relative_v1_center - relative_v1_length / 2, alpha = 0.2, color = 'y')
    #ax1.bar(left = 0.25, height = relative_v1_length, width = 0.5, bottom = relative_v1_center - relative_v1_length / 2, color = 'g')
    ax1.text(count - 3, 0, '%s Vertebrae Area' % v1_level, color = 'k', fontsize = 26)
    ax1.text(count - 3, relative_v1_center, '%s Vertebrae Area' % v2_level, color = 'k', fontsize = 26)
    #ax1.annotate('%s Vertebrae' % v1_level, xy=(count, 0), xycoords='data', xytext=(count + 0.1, 0), textcoords='data', arrowprops=dict(arrowstyle="-[", connectionstyle="arc", shrinkA = 1.0))
    
    #y1_range = np.arange(-10, 25, 5)
    #ax1.set_yticks(y1_range)
    plt.grid(True)
    
    plt.title('Scaled Placement of %s Nerve Segment (Individual Bars)\nand %s / %s Vertebrae (Solid Wide Bars) for %s Subjects - Total Overlap w/ %s / %s Vertebrae: %2d%% / %2d%%' % (n1_level, v1_level, v2_level, len(subject_names), v1_level, v2_level, overlaps[0], overlaps[1]), fontsize = 24)
    plt.ylabel('Scaled Relative Distance from the \nCenter of the %s Vertebrae (mm)' % v1_level, fontsize = 24)
    #subject_names.insert(0, '%s Vertebrae' % v2_level)
    #subject_names.insert(0, '%s Vertebrae' % v1_level)
    x_values = np.arange(0, len(subject_names), 1)
    #plt.xticks(x_values + 0.5, subject_names, rotation = 45)
    plt.xticks(x_values + 0.5, np.arange(1, len(subject_names) + 1, 1), rotation = 45)
    plt.xlabel('Subject Number', fontsize = 24)
    
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(24)
    
    if save_results: 
        plt.savefig('Figure 5 - Overlap Between %s-%s Vertebrae and %s Spinal Cord Segment.jpg' % (v1_level, v2_level, n1_level), dpi = 150)
    else:
        plt.show() 
          


def calculate_nerve_rootlet_characteristics(subject_names, label_name = 'label', subject_neck_heights = []):
    
    C3 = []
    C4 = []
    C5 = []
    C6 = []
    C7 = []
    C8 = []
    T1 = []
    individual_clusters = []
    individual_clusters.append(C3)
    individual_clusters.append(C4)
    individual_clusters.append(C5)
    individual_clusters.append(C6)
    individual_clusters.append(C7)
    individual_clusters.append(C8)
    individual_clusters.append(T1)
    
    C3_nr_minus_vertebrae_distance = []
    C3_vertebrae_locations = load_pickle("analysis", 'C3_vertebrae_locations-%s' % label_name)        
    
    max_rootlet_sets = 7
    lengths = np.zeros((len(subject_names), max_rootlet_sets), dtype = np.float32)
    nr_spaces = np.zeros((len(subject_names), max_rootlet_sets - 1), dtype = np.float32)

    C3_position = []
    neck_height = []
    subject_counter = 0
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        last_max = 0
        
        labelled_rootlets = load_pickle('analysis', '%s-rootlets_distribution-%s' % (name, label_name))
        labelled_rootlets /= np.sum(labelled_rootlets) #Gives the % of the total labelled rootlets in each position of t and theta
        labelled_rootlets_d = np.sum(labelled_rootlets, axis=1) # gives the total % of the rootlets at each value of d
        labelled_rootlets_theta = np.sum(labelled_rootlets, axis=0) # gives the total % of the rootlets at each value of theta
        
        r = np.round(labelled_rootlets_d * 100000).astype(np.int)
        points = np.repeat(spine_grid_d, r).reshape((-1, 1))
        
        #Determine which NR cluster each point belongs to
        non_zero_points = np.zeros((points.shape), dtype = np.float32) #Identifies the NR cluster that each point belongs to
        curr_NR = 0
        index = 0
        last_point = 0
        for point in points:
            #print 'point:', point
            if (point - last_point) > 1:
                curr_NR += 1
            non_zero_points[index] = curr_NR
            last_point = point
            index += 1
        
        
        #Calculate the space in between each NR cluster, the neck height corellation, and the distance between the C3 NR and C3 vertebrae
        count = 0
        print 'New Clusters: ', np.max(non_zero_points)
        for i in range(np.max(non_zero_points)):
            p = points[non_zero_points == (i + 1)]
            lengths[subject_counter, count] = np.max(p) - np.min(p) 
            if last_max != 0:
                print 'Name: %s, np.min(p): %s, last_max: %s' % (name, np.min(p), last_max)
                nr_spaces[subject_counter, count - 1] = np.min(p) - last_max 
            last_max = np.max(p)
            
            individual_clusters[count].append(p) #Saves all of the points in each NR cluster across all subjects
            if count == 0:
                for subject in subject_neck_heights:
                    if name == str(subject[0]):
                        C3_position.append(np.mean(p))
                        neck_height.append(subject[1])
                for record in C3_vertebrae_locations:
                    #print 'Record Name: %s, Name: %s' % (record[0], name)
                    if record[0] == name: #Checks to make sure the names of the records match
                        #print 'Name is %s, C3 record found' % name
                        C3_nr_minus_vertebrae_distance.append(record[2] - np.mean(p)) #C3 caudal average position less the average C3 nerve rootlet position
                        
            count += 1
        
        subject_counter += 1
    
    
    """
    Calculate Mean and Variance of NRs in Individual Subjects
    """
    means = []
    variances = []
    N_sizes = []
    cluster_names = ['C3','C4', 'C5', 'C6', 'C7', 'C8', 'T1']
    count = 0
    for cluster in individual_clusters:
        combined = []
        for sub_cluster in cluster:
            #print 'len(sub_cluster): ', len(sub_cluster)
            for item in sub_cluster:
                combined.append(item)
        
        means.append(np.mean(combined))
        variances.append(np.sqrt(np.var(combined)))
        N_sizes.append(len(cluster))
        if len(combined) > 1:
            print 'Cluster: %s, Num Subjects: %s, Min: %s, Max: %s, Mean: %s, Median: %s' % (cluster_names[count], len(cluster), np.min(combined), np.max(combined), np.mean(combined), np.median(combined))
        count += 1
    
    """
    Calculate the variance in the vertebrae distribution
    """
    distribution = load_pickle('analysis', 'vertebrae_distribution-%s' % label_name)
    print 'Vertebrae Distribution=', distribution
    vertebrae_names = ['C3','C4', 'C5', 'C6', 'C7']
    means2 = np.zeros((distribution.shape[1],1))
    variances2 = np.zeros((distribution.shape[1],1))
    N_sizes2 = []
    vertebrae_ranges = []
    for n in range(distribution.shape[1]):
        p = distribution[:,n]
        means2[n] = np.average(p)
        variances2[n] = np.sqrt(np.var(p)) 
        N_sizes2.append(distribution.shape[0]) #THIS ALWAYS SETS THE SAME SIZE - LIKELY THE CASE, BUT NEED TO MAKE IT FLEXIBLE
        vertebrae_ranges.append([vertebrae_names[n], np.min(p), np.max(p)])
        print 'Vertebrae: %s, Num Subjects: %s, Min: %s, Max: %s, Mean: %s, Median: %s' % (vertebrae_names[n], len(p), np.min(p), np.max(p), np.mean(p), np.median(p))
    
   
    
    """
    Plot the distribution of NRs and vertebrae along the spine
    """
    fig = plt.figure()
    plt.hold(True) 
    NR_labels = ['C3 Nerve Rootlets (N = %s)' % N_sizes[0], 'C4 Nerve Rootlets (N = %s)' % N_sizes[1], 'C5 Nerve Rootlets (N = %s)' % N_sizes[2], 'C6 Nerve Rootlets (N = %s)' % N_sizes[3], 'C7 Nerve Rootlets (N = %s)' % N_sizes[4], 'C8 Nerve Rootlets (N = %s)' % N_sizes[5], 'T1 Nerve Rootlets (N = %s)' % N_sizes[6]]
    plot_gaussians(spine_grid_d, means, variances, NR_labels, fig, linewidth=2, linestyle = '-')
    vertebrae_labels = ['C3 Vertebrae (N = %s)' % N_sizes2[0], 'C4 Vertebrae (N = %s)' % N_sizes2[1], 'C5 Vertebrae (N = %s)' % N_sizes2[2], 'C6 Vertebrae (N = %s)' % N_sizes2[3], 'C7 Vertebrae (N = %s)' % N_sizes2[4], '', '', '', '', '']
    plot_gaussians(spine_grid_d, means2, variances2, vertebrae_labels, fig, linewidth=4, linestyle = '--')
    plt.grid(True)
    plt.title('Gaussian Distribution of Nerve Rootlets / Vertebrae Distribution of %i Subjects Along the Spine (Source: %s)' % (len(subject_names), label_name))
    plt.xlabel('Relative Position from the PMJ (for nerve rootlets) and from the Basion (for vertebrae) [mm]')
    plt.ylabel('Probability')
    majorLocator = ticker.MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = ticker.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    
    
    """
    Calculate distance between NR regions and length of each NR segment
    """ 
    means_length = []
    variances_length = []
    means_spaces = []
    variances_spaces = []
    
    means_length_individual = []
    variances_length_individual = []
    means_spaces_individual = []
    variances_spaces_individual = []
        
    
        
    """
    Calculate the length and spaces between the NR clusters
    """
    for c in range(lengths.shape[1]):
        temp_values = lengths[:,c]
        means_length.append(np.mean(temp_values[np.nonzero(temp_values)]))
        variances_length.append(np.sqrt(np.var(temp_values[np.nonzero(temp_values)])))
        
    for d in range(nr_spaces.shape[1]):
        temp_values = nr_spaces[:,d]
        means_spaces.append(np.mean(temp_values[np.nonzero(temp_values)]))
        variances_spaces.append(np.sqrt(np.var(temp_values[np.nonzero(temp_values)])))
    
    for e in range(lengths.shape[0]):
        temp_values = lengths[e,:]
        #print 'NR Lengths: ', temp_values
        means_length_individual.append(np.mean(temp_values[np.nonzero(temp_values)]))
        variances_length_individual.append(np.sqrt(np.var(temp_values[np.nonzero(temp_values)])))
        
    for f in range(lengths.shape[0]):
        temp_values = nr_spaces[f,:]
        #print 'NR Spaces: ', temp_values
        means_spaces_individual.append(np.mean(temp_values[np.nonzero(temp_values)]))
        variances_spaces_individual.append(np.sqrt(np.var(temp_values[np.nonzero(temp_values)])))
        
    
    """
    Graph the variance in NR length and spacing amongst the various subjects
    """
    fig = plt.figure()
    plt.hold(True)
    x_values = np.arange(0, np.round(np.max(means_length) + 10), 0.1)
    plot_gaussians(x_values, means_length, variances_length, NR_labels, fig, linewidth=2, linestyle = '-')
    inter_spacing_names = ['C3-C4 Inter Rootlet Spacing', 'C4-C5 Inter Rootlet Spacing', 'C5-C6 Inter Rootlet Spacing', 'C6-C7 Inter Rootlet Spacing', 'C7-C8 Inter Rootlet Spacing', 'C8-T1 Inter Rootlet Spacing', '', '', '', '']
    plot_gaussians(x_values, means_spaces, variances_spaces, inter_spacing_names, fig, linewidth=2, linestyle = '--')
    
    plt.grid(True)
    plt.title('Length of Nerve Rootlet Segments and Inter-Nerve Rootlet Segment Spacing of %i Subjects Along the Spine' % len(subject_names))
    plt.xlabel('Length of Nerve Rootlet Segment / Spacing [mm]')
    plt.ylabel('Probability')
    
    """
    Graph the variance in NR length and spacing in each individual
    """ 
    fig = plt.figure()
    plt.hold(True)
    ind_subject_names_length = ['Subject 1 Nerve Rootlet Length', 'Subject 2 Nerve Rootlet Length', 'Subject 3 Nerve Rootlet Length', 'Subject 4 Nerve Rootlet Length', 'Subject 5 Nerve Rootlet Length', 'Subject 6 Nerve Rootlet Length', 'Subject 7 Nerve Rootlet Length', 'Subject 8 Nerve Rootlet Length', 'Subject 9 Nerve Rootlet Length', 'Subject 10 Nerve Rootlet Length', 'Subject 11 Nerve Rootlet Length', 'Subject 12 Nerve Rootlet Length', 'Subject 13 Nerve Rootlet Length', 'Subject 14 Nerve Rootlet Length', 'Subject 15 Nerve Rootlet Length', 'Subject 16 Nerve Rootlet Length', 'Subject 17 Nerve Rootlet Length', 'Subject 18 Nerve Rootlet Length', 'Subject 19 Nerve Rootlet Length', 'Subject 20 Nerve Rootlet Length']
    x_values = np.arange(0, np.round(np.max(means_length_individual) + 10), 0.1)
    plot_gaussians(x_values, means_length_individual, variances_length_individual, ind_subject_names_length, fig, linewidth=2, linestyle = '-')
    ind_subject_names_spaces = ['Subject 1 Inter Rootlet Spacing', 'Subject 2 Inter Rootlet Spacing', 'Subject 3 Inter Rootlet Spacing', 'Subject 4 Inter Rootlet Spacing', 'Subject 5 Inter Rootlet Spacing', 'Subject 6 Inter Rootlet Spacing', 'Subject 7 Inter Rootlet Spacing', 'Subject 8 Inter Rootlet Spacing', 'Subject 9 Inter Rootlet Spacing', 'Subject 10 Inter Rootlet Spacing', 'Subject 11 Inter Rootlet Spacing', 'Subject 12 Inter Rootlet Spacing', 'Subject 13 Inter Rootlet Spacing', 'Subject 14 Inter Rootlet Spacing', 'Subject 15 Inter Rootlet Spacing', 'Subject 16 Inter Rootlet Spacing', 'Subject 17 Inter Rootlet Spacing', 'Subject 18 Inter Rootlet Spacing', 'Subject 19 Inter Rootlet Spacing', 'Subject 20 Inter Rootlet Spacing']
    plot_gaussians(x_values, means_spaces_individual, variances_spaces_individual, ind_subject_names_spaces, fig, linewidth=2, linestyle = '--')
    
    plt.grid(True)
    plt.title('Length of Nerve Rootlet Segments and Inter-Nerve Rootlet Segment Spacing of %i Subjects Along the Spine' % len(subject_names))
    plt.xlabel('Length of Nerve Rootlet Segment / Spacing [mm]')
    plt.ylabel('Probability')
    
    plt.show()
    
    
def gaussian(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var ** 2) * np.exp(-(x - mean) ** 2 / 2 / var ** 2) * (x[1] - x[0])

def plot_gaussians(x, means, variances, labels, fig = None, **kargs):
    if fig == None:
        fig = plt.figure()

    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
    
    plt.hold(True)
    for n in range(len(means)):
        fig.plot(x, gaussian(x, means[n], variances[n]),
                 color=colors[n % len(colors)], figure=fig,
                 label="%s %.2f (+/- %.2f)" % (labels[n], means[n], variances[n]), **kargs)
        

    plt.legend()


def plot_gaussians_with_range(x, means, variances, mins, maxs, labels, fig = None, **kargs):
    if fig == None:
        fig = plt.figure()

    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
    
    plt.hold(True)
    for n in range(len(means)):
        fig.plot(x, gaussian(x, means[n], variances[n]),
                 color=colors[n % len(colors)], figure=fig,
                 label="%s %.1f (+/- %.1f, %0.1f, %0.1f)" % (labels[n], means[n], variances[n], mins[n], maxs[n]), **kargs)
                 
        

    plt.legend()
    

def create_flattened_spine(subject_names):
    
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        image_array = load_image_array(name, label = False, sub_name = 'smoothing')
        print image_array.dtype
        print image_array.shape
            
        CP = load_pickle(name, 'spline_control_points_centered')
        
        s = spline.CRSpline()
        s.add_spline_points(CP.astype(np.float32))
        
        t, theta = generate_spline_coords(100, 180) 
        radialPoints = 32
        sagitalView = 0
        
        print 'Starting flattening process'
        output_array = s.create_flattened_spine(t, theta, volume_spacing_mm, image_orientation, radialPoints, sagitalView, image_array)
        print output_array.dtype
        print output_array.shape
        temp_image = itk.PyBuffer[InternalImageType2D].GetImageFromArray(output_array)
        save_data_2D(temp_image, name, 'flattened_spine')
        misc.imsave('spine_profile.jpg', output_array)
        print 'Done outputting 2-D image'


def height_correlation(subject_names, label_name_1, label_name_2, type = 'Neck', save_results = True):
    
    """
    Plot the distribution of neck height versus average C3 NR position
    """
    
    fig = plt.figure(figsize = (18, 10))
    count = 0
    x = []
    y = []
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)        
        
        try:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
        except:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
        
        if type == 'Neck':
            height = profile.neck_height
        elif type == 'Total':
            height = profile.total_height
        
        if height != 0:
            #print profile.get_region('C3 NR'),', ', height
            plt.scatter(profile.get_region('C3 NR')[2], height, marker = 'x', color = 'r', s = 100)
            x.append(profile.get_region('C3 NR')[2])
            y.append(height)
            count += 1
    
    
    clf = linear_model.LinearRegression()
    clf.fit(x,y)
    regress_coefs = clf.coef_
    #print regress_coefs
    regress_intercept = clf.intercept_
    #print regress_intercept
    r_squared = clf.score(x, y)
    #print 'R^2=', r_squared
    
    x_min = np.min(x)
    x_max = np.max(x)
    new_x = np.arange(x_min, x_max, 0.1)
    new_y = regress_coefs[0] * new_x + regress_intercept
        
    #for x_value, y_value in zip(x,y):
    #    print '%s, %s' % (x_value, y_value) 
    plt.plot(new_x, new_y, c = 'b')
    plt.title('Corellation of %s Height to C3 NR Center Position for %i Subjects - R^2 = %.4f' % (type, count, r_squared))
    plt.xlabel('C3 Nerve Rootlet Center Position from PMJ (mm)')
    plt.ylabel('%s Height (cm)' % type)
    majorLocator = ticker.MultipleLocator(2)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = ticker.MultipleLocator(0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    #plt.show()
    if save_results: plt.savefig('Methods - %s Height Correlation.png' % type, dpi = 150)
    

def calculate_area(subject_names, volume_name='surface_spine_interpolated', label_name_1 = 'label', label_name_2 = 'label', spline_input = 'centered', output_name = '', specific_area = '', save_results = False):
    """
    Given a location defined by the vertebral landmarks, returns the area of a slice of the spinal cord, calculated by summing triangles in the spine
    """
    #nr_names = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'T1']
    if specific_area == '':
        nr_regions = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
    else:
        nr_regions = [specific_area]
    
    axial_slicing = 0
    
    output_area = []
    output_table = []
    
    sub_num = 0
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)        
        
        try:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
            print 'Name: %s; Using: %s' % (name, label_name_1)
        except:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
            print 'Name: %s; Using: %s' % (name, label_name_2)
        
        try:
            CP = load_pickle(name, 'spline_control_points_%s' % spline_input)
            print 'Name: %s; Using: %s' % (name, spline_input)
        except IOError:
            CP = load_pickle(name, 'spline_control_points_initial')
            print 'Name: %s; Using: %s' % (name, 'initial')
        
        s = spline.CRSpline()
        s.add_spline_points(CP.astype(np.float32))
        N = 1000
        t, theta = generate_spline_coords(N, M_curr)
        #nr_avg_distance = np.zeros((1), dtype = np.float32)
        empty_array = np.zeros((1), dtype = np.float32)
        
        #data = load_image_array(name, label = False)
        try:
            segmented_spine = load_image_array(name, label = True, sub_name = volume_name)
        except:
            print 'Error: No file found for %s / %s' % (name, volume_name)
            continue
    
        count = 0 
        #for min_max in nr_min_max:
        for region_name in nr_regions:
            temp_region = profile.get_region(region_name)
                                         
            mm_start = empty_array + temp_region[1]
            mm_end = empty_array + temp_region[3]
            
            t_start = int(s.get_relative_spline_distance(mm_start, volume_spacing_mm) * N)
            t_end = int(s.get_relative_spline_distance(mm_end, volume_spacing_mm) * N)
                                                                            
            t_start2 = t_start * 1.0 / N
            t_end2 = t_end * 1.0 / N
            
            actual_mm_start = s.get_interpolated_spline_distance(t_start2, volume_spacing_mm)
            actual_mm_end = s.get_interpolated_spline_distance(t_end2, volume_spacing_mm)
            
            #print 'mm start (initial / actual): %s / %s' % (mm_start, actual_mm_start)
            #print 'mm end (initial / actual): %s / %s' % (mm_end, actual_mm_end)
            
            roi_output = np.zeros(segmented_spine.shape, dtype = np.ushort)
            
            num_voxels = s.get_segment_voxels(t, theta, axial_slicing, image_orientation, t_start, t_end, volume_spacing_mm, segmented_spine, roi_output)
            if nr_regions[count] == 'C5 NR':
                temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(roi_output) 
                save_label(temp_image, name, 'nr_segment_%s_new' % nr_regions[count])
            full_voxels = int(num_voxels[0])
            volume = full_voxels * volume_spacing_mm[0] * volume_spacing_mm[1] * volume_spacing_mm[2]
            area = volume / (actual_mm_end - actual_mm_start) 
            print 'Name: %s; Rootlet: %s; Voxels: %s; Volume: %s; Area: %s; Thickness of Slice: %s' % (name, nr_regions[count], full_voxels, volume, area, (actual_mm_end - actual_mm_start))
            output_area.append([region_name, area])
            if sub_num == 0:
                output_table.append([region_name, area])
            else:
                output_table[count].append(area)
            
            count += 1    
        
        sub_num += 1
        #Clears the variables from memory
        segmented_spine = None
        roi_output = None
        temp_image = None
        
    
    print output_table
    
    for a in range(len(output_table)):
        print 'Region: %s; Mean Area: %s St. Dev: %s' % (output_table[a][0], np.mean(output_table[a][1:]), np.std(output_table[a][1:]))
        
        
    if specific_area != '':
        return output_area #Exits if we're only looking for the specific area in one spot
    
    
    """
    Calculate and Plot the Mean and Variance of Individual NR Areas
    """
    
    plt.figure(figsize = (10,15))
    ax = plt.subplot(111)
    
    means = []
    stddev = []
    y_error = []
    N_sizes = []
    count = 0
    for region in nr_regions:
        combined = []
        for temp_area in output_area:
            if temp_area[0] == region:
                combined.append(temp_area[1])
        
        means.append(np.mean(combined))
        stddev.append(np.sqrt(np.var(combined)))
        N_sizes.append(len(combined))
        
        if len(combined) - 1 == 15:
            t_value = 2.131
        elif len(combined) - 1 == 14:
            t_value = 2.145
        elif len(combined) - 1 == 13:
            t_value = 2.160
        elif len(combined) - 1 == 12:
            t_value = 2.179
        elif len(combined) - 1 == 16:
            t_value = 2.120
        elif len(combined) - 1 == 17:
            t_value = 2.110
        elif len(combined) - 1 == 18:
            t_value = 2.101
        elif len(combined) - 1 == 19:
            t_value = 2.093
        elif len(combined) - 1 == 20:
            t_value = 2.086
        else:
            print 'WARNING: NEED TO LOOK UP NEW T-VALUE FOR THE GIVEN DEGREES OF FREEDOM'
            return
        
        confidence_95p = t_value * np.sqrt(np.var(combined)) / np.sqrt(len(combined)) 
        y_error.append(confidence_95p)
        
        ax.errorbar(np.mean(combined), len(nr_regions) - count, xerr = confidence_95p, fmt = 'o', linewidth = 3, label = '%s %.2f (+/- %.2f)' % (region, means[-1], stddev[-1]))
        
        if len(combined) > 1:
            print '%s, Num Subjects: %s, Average: %s, Median: %s, Min: %s, Max: %s, Std. Dev.: %s, 95p Confidence: %s' % (nr_regions[count], len(combined), np.mean(combined), np.median(combined), np.min(combined), np.max(combined), np.sqrt(np.var(combined)), confidence_95p)
        count += 1
    
    print 'Means: %s' % means
    print 'Standard Deviation: %s' % stddev
    print 'Y-Error / 95p Confidence: %s' % y_error
    
    x = np.arange(0, len(nr_regions),1) 
    
    """
    Plot the distribution of NR Areas
    """
    
    #NR_labels = ['C3 Nerve Rootlet Region (N = %s)' % N_sizes[0], 'C4 Nerve Rootlet Region (N = %s)' % N_sizes[1], 'C5 Nerve Rootlet Region (N = %s)' % N_sizes[2], 'C6 Nerve Rootlet Region (N = %s)' % N_sizes[3], 'C7 Nerve Rootlet Region (N = %s)' % N_sizes[4], 'C8 Nerve Rootlet Region (N = %s)' % N_sizes[5], 'T1 Nerve Rootlet Region (N = %s)' % N_sizes[6]]
    print nr_regions
    count = 0
    x_naming = []
    for region in nr_regions:
        x_naming.append(region[0:2])
    #x_naming = nr_regions
    x_naming.append('')
    print x_naming
    x_naming.reverse()
    print x_naming
    y_max = len(x_naming)
    x_min = min(means) - 15
    x_max = max(means) + 15
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0, y_max])
    ax.grid(True)                                        
    ax.set_title('Area and 95%% Confidence Bars of Nerve Rootlets Regions (N = %s Subjects)' % (len(subject_names)))
    ax.set_ylabel('Spinal Cord Segment')
    ax.set_xlabel('Average Spinal Cord Cross-Sectional Area, mm ^ 2')
    ax.legend(loc = 'lower right')

    
    ax.set_yticklabels(x_naming)
    #plt.show()
    if save_results:
        plt.savefig('Figure 4 - Area of Spinal Cord%s.jpg' % output_name, dpi = 150)
    else:
        plt.show()
    
    print 'Done calculating area'
        
    
def draw_radial_slices(subject_names, spline_input = 'initial'):
    """
    Creates a label map to show the radial coordinates at the PMJ and one other slice down the spine
    """
    
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
        N = 1000
        
        t, theta = generate_spline_coords(N, M_curr)
        
        #This image is loaded simply to get the dimensions to create a new overlay image
        image_array = load_image_array(name, label = False)
        
        spline_name = 'spline_control_points_' + spline_input
        
        CP = load_pickle(name, spline_name)
        s = spline.CRSpline()
        s.add_spline_points(CP.astype(np.float32))
        
        new_spine_label = np.zeros(image_array.shape, dtype = np.ushort)
        slice_num = int(0.3 * N)  #an arbitrary number, the index of the slice location, starting at PMJ = 0
        t_slices = np.array([0.3 * N, 0.3 * N + (N / 5) * 1, 0.3 * N + (N / 5) * 2, 0.3 * N + (N / 5) * 3])
        t_slices /= N
        print t_slices
        t_slices = t_slices.astype(np.float32)
        t_mm = s.get_interpolated_spline_distance(t_slices, volume_spacing_mm)
        
        radius = 5 #The number of radial voxels
        
        s.create_coordinate_system_slices(t, theta, volume_spacing_mm, image_orientation, radius, slice_num, new_spine_label)
        
        temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(new_spine_label)
        save_label(temp_image, name, 'coordinate_display')        
        

def load_image_characteristics(name):
    image_data = load_pickle('analysis', 'image_characteristics')
    for subject in image_data:
        #print 'Subject: %s' % subject 
        if name == subject[0]: 
            print 'Name: %s; Spacing: %s; Orientation: %s; Num Clusters: %s' % (subject[0], subject[1], subject[2], subject[3])
            return np.array(subject[1], dtype=np.float32), subject[2], subject[3] #Returns volume_spacing_mm and image orientation


def load_mask_characteristics(name):
    image_data = load_pickle('analysis', 'mask_characteristics')
    for subject in image_data: 
        if name == subject[0]: 
            print 'Name: %s; Starting Rotation: %s; Ending Rotation: %s' % (subject[0], subject[1], subject[2])
            return subject[1], subject[2]


def graph_flexion_extension(subject_names, zero_region = 'C7 VB', spline_input = 'centered', label_name = 'label-DC', save_results = True, measure_vertebrae_from = 'pmj'):
    
    relative_region = 'C3 VB'
    
    for name in subject_names:   
        N = 200
        M = 180
        t, theta = generate_spline_coords(N, M)
        
        spline_name = 'spline_control_points_' + spline_input

        f, (ax1, ax2) = plt.subplots(2, 1)
        f.set_size_inches(22,10)
        
        N_mm = 300
        total_mm = 150
        distance_mm = np.arange(0, total_mm, 1.0 * total_mm / N_mm).astype(np.float32)
        #distance_t = np.arange(0, 1, (1.0 / N)).astype(np.float32)
        
        for extension in ['F', 'E']:
            profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name))
            volume_spacing_mm, image_orientation, K = load_image_characteristics(name + extension)
            
            regions = profile.get_all_regions()
        
            for region in regions:
                if region[0].find(relative_region) != -1 and region[4] == measure_vertebrae_from:
                    average_specific_region_distance = region[2]
            
            CP = load_pickle(name + extension, spline_name)
            s = spline.CRSpline()
            s.add_spline_points(CP.astype(np.float32))
            #distance_mm = s.get_interpolated_spline_distance(distance_t, volume_spacing_mm)
            distance_t = s.get_relative_spline_distance(distance_mm, volume_spacing_mm)
        
            empty_array = np.zeros((1), dtype = np.float32)
            temp_region = profile.get_region(zero_region)
            mm_start = empty_array + temp_region[1]
            mm_end = empty_array + temp_region[3]
            average_mm = 1.0 * (mm_start + mm_end) / 2
            
            t_start = int(s.get_relative_spline_distance(mm_start, volume_spacing_mm) * N)
            t_end = int(s.get_relative_spline_distance(mm_end, volume_spacing_mm) * N)
            
            zero_mm_location = distance_mm[int(average_mm * 1.0 / total_mm * N_mm)]
            
            temp_angles = s.get_spine_flexion_extension(distance_t, image_orientation)
            #find the index of average_t (the zero point we want to measure other angles against
            #print 'average_t (index): %s;' % (average_t) 
            zero_angle = temp_angles[int(average_mm * 1.0 / total_mm * N_mm)]
            if extension == 'F': 
                flexion_angles = temp_angles - zero_angle
                specific_region_angle_f = flexion_angles[int(average_specific_region_distance / total_mm * N_mm)]
                #ax1.plot(distance_mm, temp_angles, color = 'b', linewidth = 2, label = name + 'F: Unadjusted')
                ax1.plot(distance_mm, flexion_angles, color = 'r', linewidth = 2, label = 'Flexion: Relative to %s @ %s mm; %s Angle: (%0.1f deg @ %0.1f mm)' % (zero_region, zero_mm_location, relative_region, specific_region_angle_f, average_specific_region_distance))
            elif extension == 'E': 
                extension_angles = temp_angles - zero_angle
                specific_region_angle_e = extension_angles[int(average_specific_region_distance / total_mm * N_mm)]
                #ax2.plot(distance_mm, temp_angles, color = 'b', linewidth = 2, label = name + 'E: Unadjusted')
                ax1.plot(distance_mm, extension_angles, color = 'b', linewidth = 2, label = 'Extension: Relative to %s @ %s mm; %s Angle: (%0.1f deg @ %0.1f mm)' % (zero_region, zero_mm_location, relative_region, specific_region_angle_e, average_specific_region_distance))
                   
        ax2.plot(distance_mm, (flexion_angles - extension_angles), color = 'g', linewidth = 2, label = 'Difference in angles; %0.1f deg diff @ %s' % ((specific_region_angle_f - specific_region_angle_e), relative_region))
        
        ax1.grid(True)
        ax2.grid(True)
        ax1.set_title('Flexion Angles / Extension Angles Relative to %s' % zero_region)
        ax1.set_xlabel('Distance from the PMJ (mm)')
        ax1.set_ylabel('Angle (deg)')
        ax2. set_title('Difference Between Flexion and Extension Angles Relative to %s (Flexion minus Extension)' % zero_region)
        ax2.set_xlabel('Distance from the PMJ (mm)')
        ax2.set_ylabel('Difference in Angle (deg)')
        ax1.legend(loc = 'upper right')
        ax2.legend(loc = 'upper right')
        #plt.show()
        if save_results: plt.savefig('Methods Section - Flexion Extension Exhibit - %s.png' % name, dpi = 150)
        

def inspect_templates():
    template, index_values, distance_values = load_pickle('analysis', 'template_database')
    
    plt.figure()
    
    x_range = np.arange(0, template.shape[1], 1) 
    for a in range(template.shape[0]):
        plt.plot(x_range, template[a,:])
        plt.show()
    
    
def test_edge_vs_mm(name, axial_slice = True, spline_input = 'centered'):

    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    spline_name = 'spline_control_points_' + spline_input
    CP = load_pickle(name, spline_name)
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))

    N = 100
    M = 180
    
    t, theta = generate_spline_coords(N, M)
    segmented_spine = load_image_array(name, label = True, sub_name = 'surface_spine_interpolated') #This is only used to pass the dimensions on to c++

    try:
        newEdge_index = load_pickle('analysis', '%s-EdgeIndex_N%s_M%s_P%s' %(name, N, M, int(95)))
    except:
        print 'Index values do not exist!'
        return
    
    
    newEdge_index_original_smoothed = smoothing_filter_1D(newEdge_index, axial_smoothing = True, smoothing_type = 'median', kernelSize = 15)
    
    newEdge_mm = np.zeros((newEdge_index.shape), dtype = np.float32)
    s.convert_edge_index_distances_to_mm_distances(t, theta, volume_spacing_mm, image_orientation, axial_slice, segmented_spine, newEdge_index, newEdge_mm)
    
    newEdge_mm_smoothed = smoothing_filter_1D(newEdge_mm, axial_smoothing = True, smoothing_type = 'median', kernelSize = 15)
    
    
    newEdge_index_new = np.zeros((newEdge_index.shape), dtype = np.int32)
    s.convert_edge_mm_distances_to_index_distances(t, theta, volume_spacing_mm, image_orientation, axial_slice, segmented_spine, newEdge_mm_smoothed, newEdge_index_new)
    
    plt.figure()
    x_range = np.arange(0, newEdge_index.shape[1], 1)
    for a in range(newEdge_index.shape[0]):
        plt.plot(x_range, newEdge_index[a,:], c = 'r', linewidth = 2, label = 'Old Index Values')
        plt.plot(x_range, newEdge_mm[a,:], c = 'b', linewidth = 2, label = 'New mm Values')
        plt.plot(x_range, newEdge_mm_smoothed[a,:], c = 'g', linewidth = 2, label = 'New Smoothed mm Values')
        plt.plot(x_range, newEdge_index_new[a,:], c = 'k', linewidth = 2, label = 'New Smoothed Index Values')
        plt.plot(x_range, newEdge_index_original_smoothed[a,:], c = 'c', linewidth = 2, label = 'Original Smoothed Index Values')
        plt.grid(True)
        plt.legend()
        plt.show()
    

def convert_index_to_mm(name, edge_index, axial_slice = True, spline_input = 'initial'):
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    spline_name = 'spline_control_points_' + spline_input
    CP = load_pickle(name, spline_name)
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))

    N = edge_index.shape[0]
    M = edge_index.shape[1]
    
    t, theta = generate_spline_coords(N, M)
    spine = load_image_array(name, label = False, sub_name = 'smoothing') #This is only used to pass the dimensions on to c++
    fake_array = np.zeros((spine.shape), dtype = np.ushort)
    spine = None

    newEdge_mm = np.zeros((edge_index.shape), dtype = np.float32)
    s.convert_edge_index_distances_to_mm_distances(t, theta, volume_spacing_mm, image_orientation, axial_slice, fake_array, edge_index, newEdge_mm)
    
    
    return newEdge_mm


def convert_mm_to_index(name, edge_mm, axial_slice = True, spline_input = 'initial'):
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    spline_name = 'spline_control_points_' + spline_input
    CP = load_pickle(name, spline_name)
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))

    N = edge_mm.shape[0]
    M = edge_mm.shape[1]
    
    t, theta = generate_spline_coords(N, M)
    spine = load_image_array(name, label = False, sub_name = 'smoothing') #This is only used to pass the dimensions on to c++
    fake_array = np.zeros((spine.shape), dtype = np.ushort)
    spine = None

    edge_index = np.zeros((edge_mm.shape), dtype = np.int32)
    s.convert_edge_mm_distances_to_index_distances(t, theta, volume_spacing_mm, image_orientation, axial_slice, fake_array, edge_mm, edge_index)
    
    return edge_index
    

def calculate_area_error(subject_names, fill1_name = 'surface_spine_interpolated', fill2_name = 'C5_filled', specific_area = 'C5 NR', label_name_1 = 'label', label_name_2 = 'label2'):
    
    fill1_areas = calculate_area(subject_names, volume_name = fill1_name, label_name_1 = label_name_1, label_name_2 = label_name_2, specific_area = specific_area)
    fill2_areas = calculate_area(subject_names, volume_name = fill2_name, label_name_1 = label_name_1, label_name_2 = label_name_2, specific_area = specific_area)
    
    print fill1_areas
    print fill2_areas
    
    print len(fill1_areas)
    print len(fill2_areas)
    
    differences = []
    for a in range(len(fill1_areas)):
        percentage_difference = (fill1_areas[a][1] - fill2_areas[a][1]) / ((fill1_areas[a][1] + fill2_areas[a][1]) / 2)
        print 'Name: %s; P Difference: %s' % (subject_names[a], percentage_difference)
        differences.append(percentage_difference)
    
    #range = np.max(differences) - np.min(differences)
    mean = np.mean(differences)
    st_dev = np.sqrt(np.var(differences))
    
    print 'Min: %s; Max: %s; Mean: %s; Std Dev: %s' % (np.min(differences), np.max(differences), mean, st_dev)
        

           

def analyze_segmented_array(name, fig, count, array_in, template_index, mm_start, mm_end, spline_input, axial = False, longitudinal_angle = 90):
    
    template, index_values, distance_values, edge_3d_points = load_pickle('analysis', 'template_database')
    
    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)        
        
    CP = load_pickle(name, 'spline_control_points_%s' % spline_input)
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    N = array_in.shape[0]
    
    empty_array = np.zeros((1), dtype = np.float32)
    
    mm_start = empty_array + mm_start
    mm_end = empty_array + mm_end
    
    t_start = int(s.get_relative_spline_distance(mm_start, volume_spacing_mm) * N)
    t_end = int(s.get_relative_spline_distance(mm_end, volume_spacing_mm) * N)
    
    plt.title('Distance from Center of Spinal Cord to Cord / CSF Boundary')
    plt.ylabel('Distance to Edge (mm)')
    if axial == False:
        x_values = np.arange(0, array_in.shape[0], 1)
        if count == 3:
            plt.plot(x_values, array_in[:, longitudinal_angle], color = colors[count % len(colors)], linewidth = 4)
        else:
            plt.plot(x_values, array_in[:, longitudinal_angle], color = colors[count % len(colors)], linewidth = 1)
        
        plt.xlabel('Index of Rostral / Caudal Location Along the Spinal Cord (0 = Rostral)')
        
    else:
        x_values = np.arange(0, array_in.shape[1], 1)
        for a in range(array_in.shape[0]):
            if a > t_start and a < t_end:
                if count == 3:
                    plt.plot(x_values, array_in[a, :], color = colors[count % len(colors)], linewidth = 4)
                else:
                    plt.plot(x_values, array_in[a, :], color = colors[count % len(colors)], linewidth = 1)
                for b in range(array_in.shape[1]):
                    matching_templates = template_index[a * array_in.shape[1] + b, :]
                    for c in range(matching_templates.shape[0]):
                        if matching_templates[c] != 0:
                            x = edge_3d_points[matching_templates[c], 0]
                            y = edge_3d_points[matching_templates[c], 1]
                            z = edge_3d_points[matching_templates[c], 2]
                            distance = distance_values[matching_templates[c]]
                            index = index_values[matching_templates[c]]
                            #print 'Slice %s; Radial: %s; Matching Template No.: %s; 3D Edge Point on Template: %0.1f / %0.1f / %0.1f; Distance / Index to Edge: %0.1f / %s' % (a, b, matching_templates[c], x, y, z, distance, index) 
        plt.xlabel('Index of Rotation Around the Center of the Spinal Cord in One Axial Plane at %0.2f mm Caudal to the PMJ' % (1.0 * (mm_start + mm_end) / 2))
    
    
        

def add_missing_participant_stats(subject_names, label_name_1, label_name_2):
    
    subject_age = load_pickle('analysis', 'subject_age')
    subject_sex = load_pickle('analysis', 'subject_sex')
    subject_neck_height = load_pickle('analysis', 'subject_neck_heights')
    subject_total_height = load_pickle('analysis', 'subject_total_heights')
    
    for name in subject_names:
        try:
            old_profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
            label_name = label_name_1
        except:
            old_profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
            label_name = label_name_2
            
        new_profile = SubjectProfile(name)
        new_profile.regions = old_profile.regions
        
        #Find the subjects neck height in the database
        if old_profile.neck_height == 0:
            for subject in subject_neck_height:
                if name == str(subject[0]):
                    new_profile.add_neck_height(subject[1])
        else:
            new_profile.neck_height = old_profile.neck_height
        
        #Find the subjects total height in the database
        if old_profile.total_height == 0:
            for subject in subject_total_height:
                if name == str(subject[0]):
                    new_profile.add_total_height(subject[1])
        else:
            new_profile.total_height = old_profile.total_height
        
        #Find the subjects sex in the database
        if old_profile.sex == '':
            for subject in subject_sex:
                if name == str(subject[0]):
                    new_profile.add_sex(subject[1])
        else:
            new_profile.sex = old_profile.sex
        
        #Find the subjects age in the database
        if old_profile.age == 0:
            for subject in subject_age:
                if name == str(subject[0]):
                    new_profile.add_age(subject[1])
        else:
            new_profile.age = old_profile.age
            
        #Sort the data points
        new_profile.sort_data()
        
        save_pickle(new_profile, 'analysis', '%s-profile-%s' % (name, label_name))


def get_profile_information(subject_names, label_name_1, label_name_2, output_name):
    output_filename = os.path.join(data_path, output_name)
    open(output_filename, 'w')
    
    count = 0
    for name in subject_names:
        try:
            old_profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
            label_name = label_name_1
        except:
            old_profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
            label_name = label_name_2
        
        #for region in old_profile.get_all_regions():
        #    print region
        
        open(output_filename, 'a').write('%s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s;' % 
                                         (count, old_profile.age, old_profile.sex, old_profile.total_height, old_profile.neck_height, 
                                          old_profile.get_region('C3 NR')[1],
                                          old_profile.get_region('C3 NR')[3],
                                          old_profile.get_region('C4 NR')[1],                                          
                                          old_profile.get_region('C4 NR')[3],
                                          old_profile.get_region('C5 NR')[1],   
                                          old_profile.get_region('C5 NR')[3],
                                          old_profile.get_region('C6 NR')[1],                                          
                                          old_profile.get_region('C6 NR')[3],
                                          old_profile.get_region('C7 NR')[1],                                          
                                          old_profile.get_region('C7 NR')[3],
                                          old_profile.get_region('C8 NR')[1],                                          
                                          old_profile.get_region('C8 NR')[3],
                                          old_profile.get_region('C3 VB', 'pmj')[1],                                          
                                          old_profile.get_region('C3 VB', 'pmj')[3],
                                          old_profile.get_region('C4 VB', 'pmj')[1],   
                                          old_profile.get_region('C4 VB', 'pmj')[3],
                                          old_profile.get_region('C5 VB', 'pmj')[1],                                          
                                          old_profile.get_region('C5 VB', 'pmj')[3],
                                          old_profile.get_region('C6 VB', 'pmj')[1],                                          
                                          old_profile.get_region('C6 VB', 'pmj')[3],
                                          old_profile.get_region('C7 VB', 'pmj')[1],
                                          old_profile.get_region('C7 VB', 'pmj')[3]) + '\n')
        count += 1


def predict_nr_locations(subject_names, input_markers):
    for name in subject_names:
        set_initial_spline_points(name, label_name = 'label-DC', output_name = 'spline_control_points_initial')
        
    learn_vertebrae_distribution(subject_names, label_name_1 = 'label-DC', label_name_2 = 'label-DC', spline_input = 'initial', save_results = True)
    update_subject_profiles(subject_names, label_name_1 = 'label-DC', label_name_2 = 'label-DC', show_graph = False, analyze_rootlets = False) 
    
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        profile = load_pickle('analysis', '%s-profile-%s' % (name, 'label-DC'))
        regions = profile.get_specific_regions(input_markers)
        #print regions
        estimated_nr = get_predicted_nr_set(regions)
        
        print estimated_nr
        
        CP = load_pickle(name, 'spline_control_points_initial')
        
        s = spline.CRSpline()
        s.add_spline_points(CP.astype(np.float32))
    
        t2 = s.get_relative_spline_distance(np.array(estimated_nr, dtype = np.float32), volume_spacing_mm)
        print t2
    
        points = s.get_interpolated_spline_point(t2.astype(np.float32).flatten()) #gets the interpolated spline points of 's' at the locations identified by 't', which I believe is the % distance down the spline from the PMJ
        print points
        

def compare_two_label_volumes(name, spline_input = 'spline_control_points_initial', label_name_1 = 'label-JW', label_name_2 = 'label-DC'):
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    image_array_1 = load_image_array(name, label = True, sub_name = 'manual_filled_spine')
    image_array_2 = load_image_array(name, label = True, sub_name = 'surface_spine_interpolated')
    
    spline_name = 'spline_control_points_' + spline_input
    CP = load_pickle(name, spline_name)
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    N = 1000
    t, theta = generate_spline_coords(N, M_curr)
    
    nr_regions = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
        
    #MIGHT NEED TO CREATE REGIONS BASED ON MANUAL SPLINE HERE
    try:
        profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_1))
        print 'Name: %s; Using: %s' % (name, label_name_1)
    except:
        profile = load_pickle('analysis', '%s-profile-%s' % (name, label_name_2))
        print 'Name: %s; Using: %s' % (name, label_name_2)
    
    
    empty_array = np.zeros((1), dtype = np.float32)
    
    for region_name in nr_regions:
        temp_region = profile.get_region(region_name)
                                     
        mm_start = empty_array + temp_region[1]
        mm_end = empty_array + temp_region[3]
        
        t_start = int(s.get_relative_spline_distance(mm_start, volume_spacing_mm) * N)
        t_end = int(s.get_relative_spline_distance(mm_end, volume_spacing_mm) * N)
                   
        
        distances = s.compare_two_volumes(t, theta, t_start, t_end, volume_spacing_mm, image_orientation, image_array_1, image_array_2)
        
        print 'Region: %s' % region_name
        print 'Avg. Over Estimate: %0.2f; %% of total: %0.2f%%' % (distances[1], (100.0 * distances[0] / (distances[4] + distances[0])))
        print 'Avg. Under Estimate: %0.2f; %% of total: %0.2f%%' % (distances[5], (100.0 * distances[4] / (distances[4] + distances[0])))
        print 'Max. Over Estimate: %0.2f' % distances[2]
        print 'Avg. Relative Diff. For Over Est.: %0.3f%%' % distances[3]
        print 'Max. Under Estimate: %0.2f' % distances[6]
        print 'Avg. Relative Diff. For Under Est.: %0.3f%%' % distances[7]
        calculate_area_error(subject_names, fill1_name = 'surface_spine_interpolated', fill2_name = 'manual_filled_spine', specific_area = '%s' % region_name, label_name_1 = label_name_1, label_name_2 = label_name_2)


def create_submask(name, region_start, region_stop, label_in, spline_input = 'centered', save_results = True, get_region_length = False):
    """
    Creates a sub_mask which is within the bounds given
    """
        
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)        
    
    try:
        profile = load_pickle('analysis', '%s-profile-%s' % (name, 'label-JW'))
    except:
        profile = load_pickle('analysis', '%s-profile-%s' % (name, 'label-DC'))
    
    try:
        CP = load_pickle(name, 'spline_control_points_%s' % spline_input)
    except IOError:
        print 'ERROR: SPLINE DOES NOT EXIST'
        return
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    N = 1000
    t, theta = generate_spline_coords(N, 180)
    
    print 'Max Spline Dist: %s' % s.get_interpolated_spline_distance(0.99, volume_spacing_mm)

    temp_region_1 = profile.get_region(region_start)
    temp_region_2 = profile.get_region(region_stop)
    
    print 'Start region: %s' % temp_region_1
    print 'Stop region: %s' % temp_region_2
    
    empty_array = np.zeros((1), dtype = np.float32)
    
    mm_start = empty_array + temp_region_1[1]
    mm_end = empty_array + temp_region_2[3]
    
    print mm_start
    print mm_end
    
    if get_region_length == True:
        return (mm_end - mm_start)
    
    t_start = int(s.get_relative_spline_distance(mm_start, volume_spacing_mm) * N)
    t_end = int(s.get_relative_spline_distance(mm_end, volume_spacing_mm) * N)
    
    print 'Start t: %s' % t_start
    print 'Stop t: %s' % t_end
                            
    original_mask = load_image_array(name, label = True, sub_name = label_in)
    roi_output = np.zeros(original_mask.shape, dtype = np.ushort)
    #s.create_submask(t, theta, image_orientation, t_start, t_end, volume_spacing_mm, original_mask, roi_output)
    
    plane_values = s.get_normal_to_plane(t, theta, image_orientation, t_start, t_end) #plane_values are an array of floats
    print plane_values
    print plane_values.dtype
    print plane_values.shape
    
    print 'Creating submask...'
    create_submask_between_planes(original_mask, roi_output, plane_values)
    

    if save_results: 
        temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(roi_output) 
        save_label(temp_image, name, '%s_%s %s' % (label_in, region_start, region_stop))
        
    
    
    
    
    

def create_spline_mask(name, region_start, region_stop, label_in = 'surface_spine_interpolated', spline_input = 'centered', save_results = True):
    """
    Creates a mask of the input spline
    """
        
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)        
    
    try:
        profile = load_pickle('analysis', '%s-profile-%s' % (name, 'label-JW'))
    except:
        profile = load_pickle('analysis', '%s-profile-%s' % (name, 'label-DC'))
    
    
    
    original_mask = load_image_array(name, label = True, sub_name = label_in)
    
    
    try:
        CP = load_pickle(name, 'spline_control_points_%s' % spline_input)
        #print CP
    except IOError:
        print 'ERROR: SPLINE DOES NOT EXIST'
        return
    
    s = spline.CRSpline()
    s.add_spline_points(CP.astype(np.float32))
    N = 1000
    t, theta = generate_spline_coords(N, M_curr)
    
    #print 'Max Spline Dist: %s' % s.get_interpolated_spline_distance(0.99, volume_spacing_mm)
    
    empty_array = np.zeros((1), dtype = np.float32)
    
    if region_start != '':
        temp_region_1 = profile.get_region(region_start)
        mm_start = empty_array + temp_region_1[1]
        t_start = int(s.get_relative_spline_distance(mm_start, volume_spacing_mm) * N)
        print temp_region_1
        print mm_start
        print t_start
    else:
        t_start = 0
    
    if region_stop != '':
        temp_region_2 = profile.get_region(region_stop)
        mm_end = empty_array + temp_region_2[3]
        t_end = int(s.get_relative_spline_distance(mm_end, volume_spacing_mm) * N)
        print temp_region_2
        print mm_end
        print t_end
    else:
        t_end = 1000
    
    roi_output = np.zeros(original_mask.shape, dtype = np.ushort)
    s.create_spline_mask(t, image_orientation, t_start, t_end, volume_spacing_mm, roi_output)
    
    if save_results: 
        temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(roi_output) 
        save_label(temp_image, name, 'spline_mask_%s' % spline_input)



def calculate_axial_center(name, label_in, spline_output, save_results = True):
    #Calculate new spline points from a mask
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    num_new_cps = 500 #used to be 1000
    
    original_mask = load_image_array(name, label = True, sub_name = label_in)
    roi_output = np.zeros(original_mask.shape, dtype = np.ushort)
    
    new_CP = centroid_calc(original_mask, roi_output)
    
    if save_results: 
        temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(roi_output) 
        save_label(temp_image, name, 'axial_spline_mask_%s' % spline_output)
        
        P0 = new_CP[1,:]
        #print 'P0:', P0
        CPNewTemp = np.vstack((P0, new_CP[2:new_CP[0,0]:4,:])).astype(np.float32)
        #CPNew = CPNewTemp[0::4,:]
        
        #print CPNewTemp
        s = spline.CRSpline()
        """ NOTE THAT THIS IS USING CPNEWTEMP NOT CPNEW """
        s.add_spline_points(CPNewTemp)
        
        """
        This section ensures that the CPs are spaced evenly down the spline
        """
        t = np.linspace(0, 1, num_new_cps).astype(np.float32)
        dist_mm = s.get_interpolated_spline_distance(t, volume_spacing_mm)
        #print 'dist_mm=', dist_mm
        max_dist = dist_mm[dist_mm.shape[0] - 1]
        mm_space = np.linspace(0, max_dist, num_new_cps).astype(np.float32)
        t2 = s.get_relative_spline_distance(mm_space, volume_spacing_mm)
        
        CPDistributed = s.get_interpolated_spline_point(t2.astype(np.float32).flatten()) #gets the interpolated spline points of 's' at the locations identified by 't', which I believe is the % distance down the spline from the PMJ
        
        #print 'CPOld = ', CP
        #print 'CPDistributed = ', CPDistributed
        output_name = 'spline_control_points_%s' % spline_output
        
        save_pickle(CPDistributed, name, output_name)
        
    
        

def calculate_area_new_method(subject_names, label_in, matching_template):
    
    nr_regions = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']    
    
    output_table = []
    
    subject_num = 0
    for name in subject_names:
        volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
        
        count = 0
        for region in nr_regions:
            if label_in == 'manual_filled_spine':
                region_length = create_submask(name, '%s' % region, '%s' % region, 'manual_filled_spine', spline_input = 'manual', save_results = True, get_region_length = True)
                original_mask = load_image_array(name, label = True, sub_name = '%s_%s %s' % (label_in, region, region))
            else:
                region_length = create_submask(name, '%s' % region, '%s' % region, '%s_t%s' % (label_in, matching_template), spline_input = 'centered_interpolated_t%s' % matching_template, save_results = True, get_region_length = True)
                original_mask = load_image_array(name, label = True, sub_name = '%s_t%s_%s %s' % (label_in, matching_template, region, region))
                
            
            filled_voxels = get_filled_voxels(original_mask)
            
            volume = 1.0 * filled_voxels * volume_spacing_mm[0] * volume_spacing_mm[1] * volume_spacing_mm[2] 
            area = volume / region_length
            print 'Name: %s; Region: %s; Volume: %s; Length: %s; Area: %s' % (name, region, volume, region_length, area)
            if subject_num == 0:
                output_table.append([region, area])
            else:
                output_table[count].append(area)
            count += 1 
            
            
            
        subject_num += 1
    
    print output_table
    
    for a in range(len(output_table)):
        print 'Region: %s; Mean Area: %s St. Dev: %s; Min: %s; Max: %s' % (output_table[a][0], np.mean(output_table[a][1:]), np.std(output_table[a][1:]), np.min(output_table[a][1:]), np.max(output_table[a][1:]))


def compare_total_areas(name, regions, label, spline_input):
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    cum_volume = 0
    for region in regions:
        original_mask = load_image_array(name, label = True, sub_name = '%s_%s %s' % (label, region, region))
        filled_voxels = get_filled_voxels(original_mask)
        cum_volume += 1.0 * filled_voxels * volume_spacing_mm[0] * volume_spacing_mm[1] * volume_spacing_mm[2] 
        
    #get volume for whole segment
    original_mask = load_image_array(name, label = True, sub_name = '%s_%s %s' % (label, regions[0], regions[-1]))
    filled_voxels = get_filled_voxels(original_mask)
    whole_volume = 1.0 * filled_voxels * volume_spacing_mm[0] * volume_spacing_mm[1] * volume_spacing_mm[2]
    
    return 'Volume Comparison for %s from %s to %s. Summed Volumes: %s; Total Single Volume: %s; %% difference: %s' % (name, regions[0], regions[-1], cum_volume, whole_volume, 100.0 * (whole_volume - cum_volume) / whole_volume)


def compare_two_areas_new_method(name, regions, label_in_1, label_in_2, matching_template_1, matching_template_2, spline_input, return_differences = False):
    
        
    #nr_regions = ['C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
    #print 'Comparing %s to %s' % (nr_regions[0], nr_regions[-1])
    output_differences = []
    
    subject_num = 0
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    count = 0
    out1 = []
    out2 = []
    for region in regions:
        region_length = create_submask(name, '%s' % region, '%s' % region, '%s' % label_in_1, spline_input = spline_input, save_results = True, get_region_length = True)
        original_mask = load_image_array(name, label = True, sub_name = '%s_%s %s' % (label_in_1, region, region))
        
        filled_voxels = get_filled_voxels(original_mask)
        
        volume = 1.0 * filled_voxels * volume_spacing_mm[0] * volume_spacing_mm[1] * volume_spacing_mm[2] 
        area_1 = volume / region_length[0]
        out1.append(area_1)
        del original_mask
        
        region_length = create_submask(name, '%s' % region, '%s' % region, '%s' % label_in_2, spline_input = spline_input, save_results = True, get_region_length = True)
        original_mask = load_image_array(name, label = True, sub_name = '%s_%s %s' % (label_in_2, region, region))
        
        filled_voxels = get_filled_voxels(original_mask)
        
        volume = 1.0 * filled_voxels * volume_spacing_mm[0] * volume_spacing_mm[1] * volume_spacing_mm[2] 
        area_2 = volume / region_length[0]
        out2.append(area_2)
        
        output_differences.append((area_2 - area_1) / area_1)
         
        del original_mask
        
    if return_differences == True:
        return output_differences
    else:
        return out1, out2


def get_filled_voxels_from_label(name, label_in_1):
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    original_mask = load_image_array(name, label = True, sub_name = '%s' % label_in_1)
    
    filled_voxels = get_filled_voxels(original_mask)
    
    area = 1.0 * filled_voxels * volume_spacing_mm[0] * volume_spacing_mm[1] 
    
    print 'Voxels: %s; Area: %s' % (filled_voxels, area)
    


def calculate_area_difference(name, y_idx_to_search, input1, input2):
    
    volume_spacing_mm, image_orientation, K = load_image_characteristics(name)
    
    try:
        test1_array = load_image_array(name, label = True, sub_name = input1)
    except:
        logger.info('File: %s does not exist.' % (input1))
        return
    
    try:
        test2_array = load_image_array(name, label = True, sub_name = input2)
    except:
        logger.info('File: %s does not exist.' % (input2))
        return
    
    roi_output = np.zeros(test1_array.shape, dtype = np.ushort)
    matchPercent = area_difference_output(y_idx_to_search, test1_array, test2_array, roi_output)
    temp_image = itk.PyBuffer[LabelImageType].GetImageFromArray(roi_output) 
    save_label(temp_image, name, 'area_difference-%s_%s' % (input1, input2))
    del test1_array
    del test2_array
    
    print 'Area difference between %s and %s at %s (y-idx) (in voxels): %s' % (input1, input2, y_idx_to_search, matchPercent) 
    

def output_metrics_by_template(templates, inputs, smoothing_profile):

    file_count = 0
    filenames = os.listdir(os.path.dirname(__file__))
    
    found = True
    while found == True:
        filename = '%s_%s.txt' % ('spine_data_for_excel_by_template', file_count)
        if filename not in filenames:
            found = False
        else:
            file_count += 1
    
    
    print 'Opening file: %s' % filename
    
    paper_data = load_pickle('paper_data')
    f = open('%s' % filename,'w') #a appends, w writes
    
    for template in templates:
        for a in range(len(inputs)):
            print 'a: %s' % a
            for b in range(a + 1, len(inputs)):
                print 'b: %s' % b
                input1 = inputs[a]
                input2 = inputs[b]
                
                if 'surface_spine' in input1:
                    input1 += '%s' % template
                
                if 'surface_spine' in input2:
                    input2 += '%s' % template
                
                print 'Template: %s' % template
                print 'Input 1: %s' % input1
                print 'Input 2: %s' % input2
                
                f.write('Template: %s\nInput1: %s\nInput2: %s\nSmoothing: %s\n' % (template, input1, input2, smoothing_profile))
            
                max_time = 0
                min_time = 10000000000
                for item in paper_data:
                    if item[1] == template and item[2] == input1 and item[3] == input2:
                        time = int(item[10].strftime('%s'))
                        if time > max_time: max_time = time
                        elif time < min_time: min_time = time
                
                try:
                    f.write('Min Time: %s; Max Time: %s; Time Diff(s): %s' % (datetime.datetime.fromtimestamp(min_time).strftime('%Y-%m-%d %H:%M:%S'), datetime.datetime.fromtimestamp(max_time).strftime('%Y-%m-%d %H:%M:%S'), max_time - min_time))
                except:
                    f.write('No Time Data Available')
                    
                f.write('\n')
                
                f.write('dice;')
                for item in paper_data:
                    if item[1] == template and item[2] == input1 and item[3] == input2:
                        f.write('%s;' % item[4])
                f.write('\n')
            
                f.write('md;')
                for item in paper_data:
                    if item[1] == template and item[2] == input1 and item[3] == input2:
                        f.write('%s;' % item[6])
                f.write('\n')
            
                f.write('haus;')
                for item in paper_data:
                    if item[1] == template and item[2] == input1 and item[3] == input2:
                        f.write('%s;' % item[5])
                f.write('\n')
            
                f.write('%s areas\n' % input1)
                for item in paper_data:
                    if item[1] == template and item[2] == input1 and item[3] == input2:
                        for area in item[8]:
                            f.write('%s;' % area)
                        f.write('\n')
            
                f.write('%s areas\n' % input2)
                for item in paper_data:
                    if item[1] == template and item[2] == input1 and item[3] == input2:
                        for area in item[9]:
                            f.write('%s;' % area)
                        f.write('\n')
                f.write('\n')
                f.write('\n')

    print 'Closing File'
    f.close()                    


def output_metrics_by_subject(subjects, inputs, smoothing_profile):

    file_count = 0
    filenames = os.listdir(os.path.dirname(__file__))
    
    found = True
    while found == True:
        filename = '%s_%s.txt' % ('spine_data_for_excel_by_subject', file_count)
        if filename not in filenames:
            found = False
        else:
            file_count += 1
    
    
    print 'Opening file: %s' % filename
    
    paper_data = load_pickle('paper_data')
    f = open('%s' % filename,'w') #a appends, w writes
    
    for subject in subjects:
        for a in range(len(inputs)):
            print 'a: %s' % a
            for b in range(a + 1, len(inputs)):
                print 'b: %s' % b
                input1 = inputs[a]
                input2 = inputs[b]
                
                print 'Subject: %s' % subject
                print 'Input 1: %s' % input1
                print 'Input 2: %s' % input2
                
                f.write('Subject: %s  Templates: ' % subject)
                for item in paper_data:
                    if item[0] == subject and item[2] == input1 and item[3] == input2:
                        f.write('%s,' % item[1])
                
                f.write('\nInput1: %s\nInput2: %s\nSmoothing: %s\n' % (subject, input1, input2, smoothing_profile))
            
                max_time = 0
                min_time = 10000000000
                for item in paper_data:
                    if item[0] == subject and item[2] == input1 and item[3] == input2:
                        time = int(item[10].strftime('%s'))
                        if time > max_time: max_time = time
                        elif time < min_time: min_time = time
                
                try:
                    f.write('Min Time: %s; Max Time: %s; Time Diff (s): %s' % (datetime.datetime.fromtimestamp(min_time).strftime('%Y-%m-%d %H:%M:%S'), datetime.datetime.fromtimestamp(max_time).strftime('%Y-%m-%d %H:%M:%S'), max_time - min_time))
                except:
                    f.write('No Time Data Available')
                    
                f.write('\n')
                
                f.write('dice;')
                for item in paper_data:
                    if item[0] == subject and item[2] == input1 and item[3] == input2:
                        f.write('%s;' % item[4])
                f.write('\n')
            
                f.write('md;')
                for item in paper_data:
                    if item[0] == subject and item[2] == input1 and item[3] == input2:
                        f.write('%s;' % item[6])
                f.write('\n')
            
                f.write('haus;')
                for item in paper_data:
                    if item[0] == subject and item[2] == input1 and item[3] == input2:
                        f.write('%s;' % item[5])
                f.write('\n')
            
                f.write('%s areas\n' % input1)
                for item in paper_data:
                    if item[0] == subject and item[2] == input1 and item[3] == input2:
                        for area in item[8]:
                            f.write('%s;' % area)
                        f.write('\n')
            
                f.write('%s areas\n' % input2)
                for item in paper_data:
                    if item[0] == subject and item[2] == input1 and item[3] == input2:
                        for area in item[9]:
                            f.write('%s;' % area)
                        f.write('\n')
                f.write('\n')
                f.write('\n')

    print 'Closing File'
    f.close()                    
    

def main_segmentation(subjectName, template, smooth = 0):

    print 'Processing: %s ' % subjectName
    if spline_input == 'initial':
        try:
            set_initial_spline_points(subjectName, label_name = 'label-JW', output_name = 'spline_control_points_initial')
        except:
            set_initial_spline_points(subjectName, label_name = 'label-DC', output_name = 'spline_control_points_initial')
        #draw_spline(subjectName, 200, N_curr, M_curr, spline_input = 'initial', image_outname = 'spline_image_initial')
    
    
    #recursive segmentation returns the number of pixels (not distance) away from the spline center point the edge lies
    print 'Starting recursive segmentation...'
    min_matches = 50
    match_rate_threshold = 90
    newSegmentationEdge, newMatchPercent, newMatchedTemplateIndex, match_rate = recursive_segmentation(subjectName, template, inputMatchThreshold, min_matches, match_rate_threshold, N_curr, M_curr, index_points, axial_slice = True, spline_input = spline_input)
    
    #NEW
    if smooth == 1:
        template += 's'
    
    
    """Keep track of the match rates"""
    try:
        match_rates = load_pickle_non_subject_related('match_rates')
    except:
        match_rates = []
    
    found = False
    for item in match_rates:
        if item[0] == subjectName:
            item[1] = match_rate
            found = True
    
    if found == False:
        match_rates.append([subjectName, match_rate])
    
    save_pickle_non_subject_related(match_rates, 'match_rates')
    """End of match rates tracking"""
    
    produce_new_edge_overlay(subjectName, newSegmentationEdge, index_type = True, axial_slice = True, subImageNum = imageNum, spline_input = spline_input, output_name = 'max_%s_radial_points_t%s' % (index_points, template))
                                                                                                                                                                                                                    
    
    axial_smoothing = [True, False, False, True]

    if smooth == 0:
        kernelSizes = [1, 1, 1, 1] #Last used to segment all images
    else:
        kernelSizes = [5, 9, 5, 5] #Last used to segment all images
    #kernelSizes = [1, 5,1, 5]
    smoothing_type = ['median', 'median', 'median', 'median']
    
    newEdge_mm = convert_index_to_mm(subjectName, newSegmentationEdge, axial_slice = True, spline_input = spline_input)
    
    inputGrid = newEdge_mm
    count = 0
    for kernel, axial, smooth_type in zip(kernelSizes, axial_smoothing, smoothing_type):
        print 'Smoothing: Axial: %s, Kernel: %s' % (axial, kernel)
        #if count == 0:
        #analyze_segmented_array(subjectName, ax, count, inputGrid, newMatchedTemplateIndex, 2, 4, spline_input, axial = False, longitudinal_angle = 90)
        analyze_segmented_array(subjectName, ax, count, inputGrid, newMatchedTemplateIndex, 110, 112, spline_input, axial = True, longitudinal_angle = 90)
        outputGrid = smoothing_filter_1D_c(inputGrid, axial_smoothing = axial, smoothing_type = smooth_type, kernelSize = kernel)
        inputGrid = outputGrid
        
        #outputGrid = convert_mm_to_index(subjectName, outputGrid, axial_slice = True, spline_input = 'initial')
        #produce_new_edge_overlay(subjectName, outputGrid, index_type = True, axial_slice = True, subImageNum = imageNum, spline_input = 'initial', output_name = 'smoothed_%s' % count)
        count += 1
    
    plt.grid(True)
    #plt.show()
    
    outputGrid = convert_mm_to_index(subjectName, outputGrid, axial_slice = True, spline_input = spline_input)
    produce_new_edge_overlay(subjectName, outputGrid, index_type = True, axial_slice = True, subImageNum = imageNum, spline_input = spline_input, output_name = 'smoothed_t%s' % template)
    

    #calculate_new_center_from_edge_segmentation(subjectName, outputGrid, newMatchPercent, N_curr, M_curr, axial_slice = True, spline_input = spline_input, spline_output = spline_output)
    calculate_axial_center(subjectName, label_in = 'surface_spine_smoothed_t%s' % template, spline_output = 'centered_smoothed_t%s' % template)
    #draw_spline(subjectName, 100, N_curr, M_curr, spline_input = spline_output, image_outname = 'spline_image_refined_2')
    
    #Convert back to mm before interpolation
    outputGrid = convert_index_to_mm(subjectName, outputGrid, axial_slice = True, spline_input = spline_input)
    
    outputGrid = interpolate_to_larger_grid(outputGrid, 6, 2) #z-multiplier / theta-multiplier
    outputGrid = outputGrid.astype(np.float32)
    
    axial_smoothing = [False, True]
    #kernelSizes = [7, 5]
    #kernelSizes = [5, 1]
    if smooth == 0:
        kernelSizes = [1, 1]
    else:
        kernelSizes = [7, 5]
    smoothing_type = ['median', 'median']
    
    inputGrid = outputGrid
    for kernel, axial, smooth_type in zip(kernelSizes, axial_smoothing, smoothing_type):
        print 'Smoothing: Axial: %s, Kernel: %s' % (axial, kernel)
        outputGrid = smoothing_filter_1D_c(inputGrid, axial_smoothing = axial, smoothing_type = smooth_type, kernelSize = kernel)
        inputGrid = outputGrid
    
    outputGrid = convert_mm_to_index(subjectName, outputGrid, axial_slice = True, spline_input = spline_input)
    
    produce_new_edge_overlay(subjectName, outputGrid, index_type = True, axial_slice = True, subImageNum = imageNum, output_name = 'interpolated_t%s' % template)
    calculate_axial_center(subjectName, label_in = 'surface_spine_interpolated_t%s' % template, spline_output = 'centered_interpolated_t%s' % template)
    print 'Done drawing refined'
    
    name = multiprocessing.current_process().name
    print 'DONE: %s' % name
   


def main_segmentation_parent(template_subjects, subject_names, smooth, use_multiple_templates = False):
    """
    Need to calculate the center of the manual fill before computing templates from it
    """
    
    if use_multiple_templates == False:
        for template in template_subjects:
            jobs = []
            if len(subject_names) > 1:
                print 'It is best to run this when there are multiple templates segmenting a single subject. EXITING.'
                sleep(1000000)
            for subjectName in subject_names:
                p = multiprocessing.Process(target=main_segmentation, args = (subjectName, template, smooth, ))
                jobs.append(p)
                p.start()
    
    else:
        template = ''
        for subject in template_subjects:
            template += '%s' % (subject)
        print "Template: %s" % template
        
        jobs = []
        for subjectName in subject_names:
            p = multiprocessing.Process(target=main_segmentation, args = (subjectName, template, smooth, ))
            jobs.append(p)
            p.start()

    
    
   

def comparison_analysis_parent(subject_names, input1, input2, template_subjects, smooth, create_masks, calculate_metrics, use_multiple_templates = True):
    if use_multiple_templates == True:
        template = ''
        for subject in template_subjects:
            template += '%s' % (subject)
        print "Template: %s" % template
        
        if smooth == 1:
            template += 's'
        
        if 'surface_spine_interpolated' in input1:
            input1 += '_t%s' % template
            
        
        if 'surface_spine_interpolated' in input2:
            input2 += '_t%s' % template
        
        
            
        pool_arg_inputs = []
        for subjectName in subject_names:
            pool_arg_inputs.append([subjectName, input1, input2, template, create_masks, calculate_metrics, smooth])
        pool = multiprocessing.Pool(processes=len(subject_names))
        results = pool.map(comparison_analysis, pool_arg_inputs)
    
    else:
        for template in template_subjects:
            if smooth == 1:
                template += 's'
    
            pool_arg_inputs = []
            for subjectName in subject_names:
                pool_arg_inputs.append([subjectName, input1, input2, template, create_masks, calculate_metrics, smooth])
        
        pool = multiprocessing.Pool(processes=(len(subject_names) * len(template_subjects)))
        results = pool.map(comparison_analysis, pool_arg_inputs)
    
    #Output the data to the saved table
    try:
        paper_data = load_pickle('paper_data')
    except:
        paper_data = []
    
    if calculate_metrics == 1:
        for result in results:
            #result = [subjectName, template, input1, input2, dice, haus, md, temp1, temp2, datetime.datetime.now()]
            subjectName = result[0]
            dice = result[4]
            haus = result[5]
            md = result[6]
            nr_regions = result[7]
            temp1 = result[8]
            temp2 = result[9]
        
            found_data = False
            for data_row in paper_data:
                if data_row[0] == subjectName and data_row[1] == template and data_row[2] == input1 and data_row[3] == input2:
                    data_row[4] = dice
                    data_row[5] = haus
                    data_row[6] = md
                    data_row[7] = nr_regions
                    data_row[8] = temp1
                    data_row[9] = temp2
                    data_row[10] = datetime.datetime.now()
                    found_data = True
            
            if found_data == False:
                paper_data.append([subjectName, template, input1, input2, dice, haus, md, nr_regions, temp1, temp2, datetime.datetime.now()])
        
    save_pickle(paper_data, 'paper_data')




def comparison_analysis(input):
    
    print 'Input: %s' % input
    sleep(5)
    subjectName = input[0]
    input1 = input[1]
    input2 = input[2]
    template = input[3]
    create_masks = input[4]
    calculate_metrics = input[5]
    
    area_diff = []
    start = 'C3 NR'
    stop = 'C8 NR'

    area_matching_template_1 = template
    area_matching_template_2 = template
    input1_areas = []
    input2_areas = []
    nr_regions = ['C3 NR', 'C4 NR', 'C5 NR', 'C6 NR', 'C7 NR', 'C8 NR']
    #nr_regions = ['C8 NR']
    for region in nr_regions:
        area_diff.append([])
    
    compare_area_error = []
   
        
    if create_masks == 1: #The try and excepts allow you to only enter either input 1 or 2, if you don't want both calculated.
        #Calculate_axial_center calculates a new centroid and outputs a spline and a mask
        #Create_spline_mask only creates a mask from an existing spline
        
        #This creates the files with the names 'axial_spline_mask_'
        
        try:
            calculate_axial_center(subjectName, label_in = input1, spline_output = input1)
        except:
            pass
        try:
            calculate_axial_center(subjectName, label_in = input2, spline_output = input2)
        except:
            pass
        #create_spline_mask(subjectName, region_start = '', region_stop = '', label_in = 'manual_filled_spine', spline_input = 'manual_filled_spine', save_results = True)
        
        
        if input1 != '':
            try:
                #create_submask(subjectName, start, stop, input1, spline_input = 'centered_interpolated_t%s' % template, save_results = True)
                create_submask(subjectName, start, stop, input1, spline_input = 'manual_filled_spine', save_results = True)
                for region in nr_regions:
                    #create_submask(subjectName, '%s' % region, '%s' % region, input1, spline_input = 'centered_interpolated_t%s' % template, save_results = True)
                    create_submask(subjectName, '%s' % region, '%s' % region, input1, spline_input = 'manual_filled_spine', save_results = True)
                #create_submask(subjectName, start, stop, 'axial_spline_mask_%s' % input1b, spline_input = 'centered_interpolated_t%s' % template, save_results = True)
                create_submask(subjectName, start, stop, 'axial_spline_mask_%s' % input1, spline_input = 'manual_filled_spine', save_results = True)
            except:
                pass
        
       
        if input2 != '':
            try:
                #create_submask(subjectName, start, stop, input2, spline_input = 'centered_interpolated_t%s' % template, save_results = True)
                create_submask(subjectName, start, stop, input2, spline_input = 'manual_filled_spine', save_results = True)
                for region in nr_regions:
                    create_submask(subjectName, '%s' % region, '%s' % region, input2, spline_input = 'manual_filled_spine', save_results = True)
                #create_submask(subjectName, start, stop, input2, spline_input = 'initial', save_results = True)
                #create_submask(subjectName, start, stop, 'axial_spline_mask_%s' % input2b, spline_input = 'centered_interpolated_t%s' % template, save_results = True)
                create_submask(subjectName, start, stop, 'axial_spline_mask_%s' % input2, spline_input = 'manual_filled_spine', save_results = True)
            except:
                pass
        
        
    
    if calculate_metrics == 1:
        dice = compare_segmentations('dice', subjectName, '%s_%s %s' % (input1, start, stop), '%s_%s %s' % (input2, start, stop)) #use dice or jaccard or hausdorff
        haus = compare_segmentations('hausdorff', subjectName, '%s_%s %s' % (input1, start, stop), '%s_%s %s' % (input2, start, stop)) #use dice or jaccard or hausdorff   
        md = compare_segmentations('hausdorff-mean', subjectName, 'axial_spline_mask_%s_%s %s' % (input1, start, stop), 'axial_spline_mask_%s_%s %s' % (input2, start, stop)) #use dice or jaccard or hausdorff
        
        #UNCOMMENT THESE
        #temp1, temp2 = compare_two_areas_new_method(subjectName, nr_regions, input1, input2, matching_template_1 = area_matching_template_1, matching_template_2 = area_matching_template_1, spline_input = 'centered_interpolated_t%s' % template)
        #returns an array of the areas for each of the inputs
        temp1, temp2 = compare_two_areas_new_method(subjectName, nr_regions, input1, input2, matching_template_1 = area_matching_template_1, matching_template_2 = area_matching_template_1, spline_input = 'manual_filled_spine')
        #input1_areas.append(temp1)
        #input2_areas.append(temp2)
        
        #compare_area_error.append(compare_total_areas(subjectName, nr_regions, input1, spline_input = 'centered_interpolated_t%s' % template))
        compare_area_error.append(compare_total_areas(subjectName, nr_regions, input1, spline_input = 'manual_filled_spine'))

        #print 'Subject: %s' % subjectName
        print 'Dice: %s' % dice
        print 'Haus: %s' % haus
        print 'MD: %s' % md
        print 'Area Differences: %s' % area_diff
        print 'Mean Distance Between Splines Mean/St dev: %s / %s' % (np.mean(md), np.std(md))
        #print 'Area Difference Mean / St. Dev: %s / %s' % (np.mean(area_diff), np.std(area_diff))
        
        #UNCOMMENT THESE
        print input1
        for area in input1_areas:
            print area
        print input2 
        for area in input2_areas:
            print area 
        
        y_idx_to_search = 296
        calculate_area_difference(subjectName, y_idx_to_search, '%s_%s %s' % (input1, start, stop), '%s_%s %s' % (input2, start, stop)) #use dice or jaccard or hausdorff
        print 'Note that area difference is the average of the average segmental differences for all subjects'
        
        for item in compare_area_error:
            print item    

        return [subjectName, template, input1, input2, dice, haus, md, nr_regions, temp1, temp2, datetime.datetime.now()]
            

def output_metrics_by_template_and_subject(templates, subjects, input1in, input2in, smoothing_profile, regions, outputname = ''):
#outputs data arrays with subjects along the x axis and templates along the y axis

    
    if outputname != '':
        filename = '%s_%s_%s_%s.txt' % ('spine_data_for_excel_matrix', input1in, input2in, outputname)
    else:
        file_count = 0
        filenames = os.listdir(os.path.dirname(__file__))
        
        found = True
        while found == True:
            filename = '%s_%s_%s_%s.txt' % ('spine_data_for_excel_matrix', input1in, input2in, file_count)
            if filename not in filenames:
                found = False
            else:
                file_count += 1
    
    
    print 'Opening file: %s' % filename
    
    paper_data = load_pickle('paper_data')
    f = open('%s' % filename,'w') #a appends, w writes
    f.write('%s\n' % filename)
    f.write('Input1: %s, Input2: %s, Smoothing: %s\n' % (input1in, input2in, smoothing_profile))
    #Write the dice coefficients
    f.write('y=Template, x=Subject;')
    for subject in subjects:
        f.write('%s;' % subject)
    f.write('\n')
        
    f.write('Dice\n')
    for template in templates:
            if smoothing_profile == 1:
                template += 's'
            
            if 'surface_spine' in input1in:
                input1 = '%s_t%s' % (input1in, template)
            else:
                input1 = input1in
                
            if 'surface_spine' in input2in:
                input2 = '%s_t%s' % (input2in, template)
            else:
                input2 = input2in

            f.write('%s;' % template)
            for subject in subjects:                
                    for item in paper_data:
                        if item[0] == subject and item[1] == template and item[2] == input1 and item[3] == input2:
                            if subject[:5] == template[:5] and len(subject) < 9 and len(template) < 9:
                                f.write(';')
                            else:
                                f.write('%s;' % item[4])
            f.write('\n')


    #Write the Haus
    f.write('\nHaus\n')
    for template in templates:
            if smoothing_profile == 1:
                template += 's'
        
            if 'surface_spine' in input1in:
                input1 = '%s_t%s' % (input1in, template)
            else:
                input1 = input1in
                
            if 'surface_spine' in input2in:
                input2 = '%s_t%s' % (input2in, template)
            else:
                input2 = input2in
                
            f.write('%s;' % template)
            for subject in subjects:                
                    for item in paper_data:
                        if item[0] == subject and item[1] == template and item[2] == input1 and item[3] == input2:
                            if subject[:5] == template[:5] and len(subject) < 9 and len(template) < 9:
                                f.write(';')
                            else:
                                f.write('%s;' % item[5])
            f.write('\n')

    #Write the MD
    f.write('\nMD\n')
    for template in templates:
            if smoothing_profile == 1:
                template += 's'
                
            if 'surface_spine' in input1in:
                input1 = '%s_t%s' % (input1in, template)
            else:
                input1 = input1in
                
            if 'surface_spine' in input2in:
                input2 = '%s_t%s' % (input2in, template)
            else:
                input2 = input2in
            
            f.write('%s;' % template)
            for subject in subjects:                
                    for item in paper_data:
                        if item[0] == subject and item[1] == template and item[2] == input1 and item[3] == input2:
                            if subject[:5] == template[:5] and len(subject) < 9 and len(template) < 9:
                                f.write(';')
                            else:
                                f.write('%s;' % item[6])
            f.write('\n')
    f.write('\n')
    
    #Write the region areas for input 1
    region_count = 0
    for region in regions:
    
        f.write('%s %s Areas\n' % (input1, region))
        for template in templates:
                if smoothing_profile == 1:
                    template += 's'
                    
                if 'surface_spine' in input1in:
                    input1 = '%s_t%s' % (input1in, template)
                else:
                    input1 = input1in
                    
                if 'surface_spine' in input2in:
                    input2 = '%s_t%s' % (input2in, template)
                else:
                    input2 = input2in
                
                f.write('%s;' % template)
            
                for subject in subjects:
                    for item in paper_data:
                            if item[0] == subject and item[1] == template and item[2] == input1 and item[3] == input2:
                                if subject[:5] == template[:5] and len(subject) < 9 and len(template) < 9:
                                    f.write(';')
                                else:
                                    f.write('%s;' % item[8][region_count])
                
                f.write('\n')
        f.write('\n')
        region_count += 1

    #Write the region areas for input 2
    region_count = 0
    for region in regions:
    
        f.write('%s %s Areas\n' % (input2, region))
        for template in templates:
                if smoothing_profile == 1:
                    template += 's'
                    
                if 'surface_spine' in input1in:
                    input1 = '%s_t%s' % (input1in, template)
                else:
                    input1 = input1in
                    
                if 'surface_spine' in input2in:
                    input2 = '%s_t%s' % (input2in, template)
                else:
                    input2 = input2in
                
                f.write('%s;' % template)
            
                for subject in subjects:
                    for item in paper_data:
                            if item[0] == subject and item[1] == template and item[2] == input1 and item[3] == input2:
                                if subject[:5] == template[:5] and len(subject) < 9 and len(template) < 9:
                                    f.write(';')
                                else:
                                    f.write('%s;' % item[9][region_count])
                
                f.write('\n')
        f.write('\n')
        region_count += 1

    print 'Closing File'
    f.close()                    





#MAIN PROGRAM
spacing_1 = [0.39060, 0.39060, 0.30011]
spacing_2 = [0.4000, 0.3906, 0.3906]
spacing_3 = [0.3906, 0.3906, 0.4000]
spacing_4 = [0.3000, 0.5700, 0.5700]
spacing_5 = [0.4297, 0.4297, 0.30004]
spacing_6 = [1.0000, 1.0000, 1.0000]
spacing_7 = [1.0938, 1.0938, 1.0000]

#[subject_name, voxel size, image acquisition oritentation (1 = 'coronal', 0 = 'sagital'), number of marked clusters (0 = unknown)] 
image_characteristics = []
image_characteristics.append(["13696", spacing_1, 0, 7])
image_characteristics.append(["13697", spacing_1, 0, 7])
image_characteristics.append(["13746", spacing_1, 0, 7])
image_characteristics.append(["13747", spacing_1, 0, 7])
image_characteristics.append(["13755", spacing_1, 0, 7])
image_characteristics.append(["14350", spacing_1, 0, 7])
image_characteristics.append(["14406", spacing_1, 0, 7])
image_characteristics.append(["14411", spacing_1, 0, 7])
image_characteristics.append(["14493", spacing_1, 0, 7])
image_characteristics.append(["14693", spacing_1, 0, 7])

save_pickle(image_characteristics, 'analysis', 'image_characteristics')

spine_mask_templates = []
spine_mask_templates.append(['14693', 0, 360])
spine_mask_templates.append(['13696', 0, 360])
spine_mask_templates.append(['13697', 0, 360])
spine_mask_templates.append(['13746', 0, 360])
spine_mask_templates.append(['14411', 0, 360])
spine_mask_templates.append(['13747', 125, 235])
spine_mask_templates.append(['19525', 125, 235])

save_pickle(spine_mask_templates, 'analysis', 'mask_characteristics')


N_curr = 100
M_curr = 180
inputMatchThreshold = 1.00
index_points = 40

imageNum = '' #Use '' for anatomical images
startTime = time.time()


name = "14693"
label_name_1 = 'label-JW'
label_name_2 = 'label-DC'

#Number 1
spline_input = 'initial'
spline_output = 'centered'

ax = plt.subplot(111)

template_subjects = ["13755"]
template_subjects = ["13697"]
template_subjects = ["13696"]
template_subjects = ["13746"]
template_subjects = ["14350"]
template_subjects = ["14693"]
template_subjects = ["14493"]
template_subjects = ["14411"]
template_subjects = ["13747"]
template_subjects = ["14406"]


subject_names = ["14406", "13747", "14411", "14493", "14693", "14350", "13746", "13696", "13697", "13755"] # done segmenting
smooth = 0

use_multiple_templates = True  
for subjectName in template_subjects:
    calculate_axial_center(subjectName, label_in = 'manual_filled_spine', spline_output = 'manual')

if use_multiple_templates == True:
    compute_templates_from_spine_mask(template_subjects, index_points, overwrite = True, axial_slice = True, label_name = '', save_results = True) #when label_name = '', it uses the manual fill spline points
else:
    for template in template_subjects:
        compute_templates_from_spine_mask([template], index_points, overwrite = True, axial_slice = True, label_name = '', save_results = True) #when label_name = '', it uses the manual fill spline points


main_segmentation_parent(template_subjects, subject_names, smooth, use_multiple_templates = True) #SET THIS TO TRUE WHEN YOU ARE RUNNING A NORMAL SEGMENTATION WITH 1 OR MORE TEMPLATES
endTime = time.time()
print 'Elapsed Time(ms):', (endTime - startTime) * 1000, ', (s):', (endTime - startTime),', (m):', (endTime - startTime)/60
#sleep(100000)

input1 = 'manual_filled_spine'
#input1 = 'manual_filled_spine-DC_autofill'
#input1 = ''
#input2 = 'surface_spine_interpolated'
input2 = 'manual_filled_spine-DC_autofill'
create_masks = 0
calculate_metrics = 1

comparison_analysis_parent(subject_names, input1, input2, template_subjects, smooth, create_masks, calculate_metrics, use_multiple_templates = True)

endTime = time.time()
print 'Elapsed Time(ms):', (endTime - startTime) * 1000, ', (s):', (endTime - startTime),', (m):', (endTime - startTime)/60


#learn_nerve_rootlets_distribution(subject_names, label_name_1 = 'label-JW', label_name_2 = 'label-DC', spline_input = 'manual', save_results=True)
#learn_vertebrae_distribution(subject_names, label_name_1 = 'label-JW', label_name_2 = 'label-DC', spline_input = 'manual', save_results = True)
#learn_nerve_rootlets_distribution(subject_names, label_name_1 = 'label-JW', label_name_2 = 'label-DC', spline_input = 'initial', save_results=True)
#learn_vertebrae_distribution(subject_names, label_name_1 = 'label-JW', label_name_2 = 'label-DC', spline_input = 'initial', save_results = True)

#update_subject_profiles(subject_names, label_name_1 = 'label-JW', label_name_2 = 'label-DC', show_graph = True)

#Figure 3
#display_spine_distribution(subject_names, 'both', label_name_1 = 'label-JW', label_name_2 = 'label-DC', save_results = True, output_name = 'newest_Jan 2015')
#display_rootlet_or_vertebrae_lengths(subject_names, 'Vertebrae', label_name_1 = 'label-JW', label_name_2 = 'label-DC', save_results = True)
#display_rootlet_or_vertebrae_lengths(subject_names, 'Nerve Rootlets', label_name_1 = 'label-JW', label_name_2 = 'label-DC', save_results = True)
#display_spine_distribution(subject_names, 'Vertebrae', label_name = 'label-DC', save_results = True, output_name = 'new')
#display_spine_distribution(subject_names, 'Nerve Rootlets', label_name = 'label-DC', save_results = True, output_name = 'new')

#for name in subject_names:
#    profile = load_pickle('analysis', '%s-profile-%s' % (name, 'label-DC'))
#    profile.display_profile()

#Figure 4
#calculate_area(subject_names, volume_name='surface_spine_interpolated', label_name_1 = 'label-JW', label_name_2 = 'label-DC', output_name = '11012013', save_results = True)

#Figure 5
#This is a smaller set of subjects, arbitrarily picked to create a more compact display
subject_names_10 = ["13747", "13696", "14407","13757","14493","13755","19432", "19433","19481","19525"]
#subject_numbers_for_graph = [1,4,5,9,10,12,15,16,18,20]
#display_full_profiles(subject_names_10, label_name_1 = 'label-JW', label_name_2 = 'label-DC', save_results = True, output_name = 'newest_Jan 2015')
#display_nerve_rootlet_offset(subject_names_10, label_name_1 = 'label-JW', label_name_2 = 'label-DC', v1_level = 'C6', v2_level = 'C7', n1_level = 'C7', save_results = False)
