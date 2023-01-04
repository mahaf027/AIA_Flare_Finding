'''
This file contains the global constants and little functions that aid all the processes in the
project.
'''

#Modules
import pickle
import os
import sys
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt

#Setting global constants
MAX_DOWNLOAD_ATTEMPTS = 3
DATA_DIR = './DATA/'
FITS_DIR_FORMAT = DATA_DIR + '{date}/fits/'
LIGHTCURVES_DIR_FORMAT = DATA_DIR + '{date}/lightcurves/'
IMAGES_DIR_FORMAT = DATA_DIR + '{date}/images/'
TIME_STR_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
FITS_FILE_NAME_FORMAT = 'aia_{date}_{time}_{wavelength:04d}_image_lev1.fits'
sys.path.append(os.getcwd())
MAP_X_LABEL = 'X (arcseconds)'
MAP_Y_LABEL = 'Y (arcseconds)'
COLORMAP = {94:'red', 131:'red', 171:'cyan', 193:'cyan', 211:'cyan', 304:'cyan', 335:'red',
            1600:'red', 1700:'cyan'}
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 8

def ins_sort_func(ins):
    '''
    Used to generate a sort key for a list of Instant objects. Instants will be sorted by time,
    from earliest to latest.
    -----------
    Parameters:
    -----------
    ins : Instant
        The Instant object being checked
    '''
    return ins.timestamp

def cluster_sort_func(cluster):
    '''
    Used to generate a sort key for a list of Cluster objects. Clusters will be sorted by time,
    from earliest to latest.
    -----------
    Parameters:
    -----------
    cluster : Cluster
        The Cluster object being checked
    '''
    #Assumes all Macropixels in the Cluster are simultaneous, which they always should be
    return cluster.elements[0].timestamp

def adjust_n(num_data, N, bool_report=False):
    '''
    Adjust the boxcar width based on the amount of available data. The boxcar width is adjusted if
    num_data < 2N.
    -----------
    Parameters:
    -----------
    num_data : int
        The number of available data points.
    N : int
        The proposed boxcar width.
    bool_report : Boolean
        Specifies whether the finalized value of N is printed.
    --------
    Returns:
    --------
    N : int
        The new boxcar width.
        May be equal to the original value if left unchanged.
    '''
    if num_data < 2*N:
        N_old = N
        N = max(num_data//10, 3)
        if N%2 == 0:
            N += 1
        if bool_report:
            print('Insufficient data points for given boxcar width N=' + str(N_old) +
                    '. Defaulting to N=' + str(N))
    else:
        if bool_report:
            print(f'Proceeding with given boxcar width N={N}')
    return N

def moving_average(arr, N, bool_report):
    '''
    Performs a very fast moving average with width N.
    -----------
    Parameters:
    -----------
    arr : numpy ndarray
        The data over which we wish to take the average (along axis 0).
    N : int
        The width of the boxcar to be used for the moving average.
    bool_report : Boolean
        Specifies whether to print a statement signaling that the function is being performed and
        reporting the boxcar width being used.
    --------
    Returns:
    --------
        An ndarray containing the moving average results
    '''
    N = adjust_n(len(arr), N, bool_report)
    if bool_report:
        print('Applying fast moving average. Provided N=' + str(N))
    #Taking advantage of array indexing to do a fast process equivalent to a moving average
    ret = np.cumsum(arr, axis=0, dtype=float)
    ret[N:] = ret[N:] - ret[:-N]
    return ret[N-1:] / N

def make_directories(date):
    '''
    Create the necessary directories for storing the FITS files, the lightcurve CSVs, and the
    images. The created directories are: ./fits, ./lightcurves/region, ./lightcurves/pixels,
    and ./images.
    -----------
    Parameters:
    -----------
    date : str
        A formatted string representing a date. Formatted as YYYYMMDD
    '''
    dirs = [
        DATA_DIR,
        DATA_DIR + date + '/',
        FITS_DIR_FORMAT.format(date=date),
        DATA_DIR + date + '/Instants/',
        DATA_DIR + date + '/Intervals/',
        IMAGES_DIR_FORMAT.format(date=date),
        DATA_DIR + date + '/movies/',
        DATA_DIR + date + '/figures/'
    ]
    #Making the directories
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)

def epoch_to_datetime(t):
    '''
    Convert an epoch time to a datetime object.
    -----------
    Parameters:
    -----------
    t : float
        A timestamp in epoch time.
    --------
    Returns:
    --------
        A datetime object corresponding to the input time. It is timezone aware and in UTC.
    '''
    return datetime.fromtimestamp(t, tz=timezone.utc)

def str_to_epoch(s):
    '''
    Convert a date-time string to epoch time. The string must be formatted as
    '%Y-%m-%dT%H:%M:%S.%f'.
    -----------
    Parameters:
    -----------
    s : str
        A string containing the date and time. Formatted as stated above.
    --------
    Returns:
    --------
        A float of the epoch time corresponding to the provided string.
    '''
    dt = datetime.strptime(s, TIME_STR_FORMAT).replace(tzinfo=timezone.utc)

    return datetime_to_epoch(dt)

def datetime_to_epoch(dt):
    '''
    Convert a datetime object to an epoch timestamp.
    -----------
    Parameters:
    -----------
    dt : datetime object
        The input datetime object. This must be timezone aware.
    --------
    Returns:
    --------
        A float of the epoch time corresponding to the provided string.
    '''
    epoch =  datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds()

def epoch_to_str(t):
    '''
    Convert an epoch time to a string format. The output format is formatted as
    '%Y-%m-%dT%H:%M:%S.%f'.
    -----------
    Parameters:
    -----------
    t : float
        A timestamp in epoch time.
    --------
    Returns:
    --------
        A string corresponding to the input time.
    '''
    return epoch_to_datetime(t).strftime(TIME_STR_FORMAT)

def read_instant_from_file(fname):
    '''
    Uses the pickle module to load a previously saved Instant object into a usable Instant.
    -----------
    Parameters:
    -----------
    fname : str
        The relative path to the binary file containing the Instnat object.
    --------
    Returns:
    --------
    instant : Instant object
        The desired Instant object.
    '''
    f = open(fname, 'rb')
    instant = pickle.load(f)
    f.close()
    return instant

def read_interval_from_file(fname):
    '''
    Uses the pickle module to load a previously saved Interval object into a usable Interval.
    -----------
    Parameters:
    -----------
    fname : str
        The relative path to the binary file containing the Interval object.
    --------
    Returns:
    --------
    interval : Interval object
        The desired Interval object.
    '''
    f = open(fname, 'rb')
    interval = pickle.load(f)
    f.close()
    return interval
