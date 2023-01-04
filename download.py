'''
This file contains the method used to download AIA FITS files from JSOC.
'''

#Modules
import platform
import sys
import os
import http.client
from datetime import datetime, timedelta, timezone
import astropy.units as u
from sunpy.net import Fido, attrs as a
import requests

#Other files
from helpers import FITS_DIR_FORMAT, make_directories

def get_fits(desired_time, wavelength):
    '''
    Downloads an AIA level 1 fits file using FIDO at a time within 6 seconds of the desired time.
    Download will be attempted 3 times before declaring failure.
    -----------
    Parameters:
    -----------
    desired_time : float
        The unix timestamp of the time for which data is queried. AIA samples every 12 seconds, so
        the data obtained will be within 6 seconds of the input time.
    wavelength : int
        The AIA channel for which data is to be queried.
    --------
    Returns:
    --------
    fname : str
        The path to the file that was downloaded.
    '''
    dt = datetime.fromtimestamp(desired_time, tz=timezone.utc)
    start = dt - timedelta(seconds=6)
    end = dt + timedelta(seconds=6)
    #Checking to see if the desired FITS file has already been downloaded
    date_str = dt.strftime('%Y%m%d')
    for i in range(12):
        time_check = start + timedelta(seconds=i)
        fname_check = FITS_DIR_FORMAT.format(date=date_str) + f"aia_{date_str}_" + \
                        f"{time_check.strftime('%H%M%S')}_{wavelength:04}_image_lev1.fits"
        if os.path.exists(fname_check):
            return fname_check
    #Making the necessary directories and fetching the FITS file
    make_directories(dt.strftime('%Y%m%d'))
    attempt_number = 1
    while attempt_number <= 3:
        print(f'Attempting download for {wavelength} A, {start} - {end}')
        try:
            result = Fido.search(a.Time(start, end), a.Instrument('aia'),
                                    a.Wavelength(wavelength * u.angstrom), a.Sample(12*u.second))
            date = result[0][0][0].value.split(' ')[0].replace('-', '')
            path = FITS_DIR_FORMAT.format(date=date)
            fname = Fido.fetch(result[0][0], site='ROB', path=path, progress = False)[0]
            break
        except (http.client.RemoteDisconnected, requests.exceptions.ConnectionError) as error:
            print(f'Exception on attempt number {attempt_number}/3.\n{error}')
            attempt_number += 1
            continue
    #Quitting if we tried unsuccessfully 3 times
    if attempt_number > 3:
        print('Exceeded maximum number of attempts. Exitting.')
        sys.exit(1)
    #Adjusting fname for potential OS differences
    if platform.system() == 'Windows':
        fname = fname.replace('\\', '/')
    return fname
