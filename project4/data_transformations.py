'''data_transformations.py
Dean Hickman
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Fall 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    min = np.min(data, axis = 0)
    max = np.max(data, axis = 0)
    return (data - min)/(max - min)


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    means = np.mean(data, axis = 0)
    return data - means


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    rad = np.deg2rad(degrees)
    ro = np.zeros((3,3))
    if(axis == 'x'):
        ro[0,0] = 1 
        ro[1,1] = np.cos(rad)
        ro[1,2] = -np.sin(rad)
        ro[2,1] = np.sin(rad)
        ro[2,2] = np.cos(rad)
    elif(axis == 'z'):
        ro[2,2] = 1 
        ro[0,0] = np.cos(rad)
        ro[0,1] = -np.sin(rad)
        ro[1,0] = np.sin(rad)
        ro[1,1] = np.cos(rad)
    elif(axis == 'y'):
        ro[1,1] = 0
        ro[0,0] = np.cos(rad)
        ro[0,2] = np.sin(rad)
        ro[2,0] = -np.sin(rad)
        ro[2,2] = np.cos(rad)
    return ro
   
