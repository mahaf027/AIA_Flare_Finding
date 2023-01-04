'''
This file contains definitions of the various classes employed in flare finding. Instant contains
the image data for one wavelength at one specific timestamp. Subsequent snapshots of processing
steps are stored in parallel in the same Instant object. An Interval object is an ordered
collection of Instant objects. This class allows for the interval-wide processing steps like boxcar
and sdev calculation. SignificantPixel objects contain the data for an individual pixel that has
been identified as significant. These pixel objects are grouped spatially into Clusters, and
Clusters that at least partially overlap in adjacent times are grouped into an Event.
'''

#Modules
from datetime import datetime, timezone, timedelta
import pickle
import os
import numpy as np
import sunpy.map
import astropy.units as u
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
import imageio

#Other Files
from helpers import MAP_X_LABEL, MAP_Y_LABEL, TIME_STR_FORMAT, DATA_DIR
from helpers import ins_sort_func, cluster_sort_func, epoch_to_str, adjust_n, moving_average
from download import get_fits

class Macropixel:
    '''
    Contains the data for one 3x3 macropixel at one time. Instances of this class are the building
    blocks for Instant and Cluster objects.
    -----------
    Attributes:
    -----------
    timestamp : float
        The Unix timestamp of the image the macropixel is taken from.
    wavelength : int
        The AIA channel from which the data came.
    indices : (int, int)
        The indices of the macropixel in the Macropixel attribute of the Instant for which it was
        created.
    brightness : int
        The average brightness of the 9 pixels.
    processed : float
        The macropixel's value after detrending and normalization.
    '''

    def __init__(self, timestamp, wavelength, indices, brightness):
        '''
        Creates a SignificantPixel object with the provided attributes.
        -----------
        Parameters:
        -----------
            Described in the class docstring. 'processed' is given a value after initialization.
        '''
        self.timestamp = timestamp
        self.wavelength = wavelength
        self.indices = indices
        self.brightness = brightness
        self.processed = None

    def __str__(self):
        '''
        Produces a String representation of the instance.
        --------
        Returns:
        --------
        self_str : String
            The String representation.
        '''
        d_t = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        time_str = d_t.strftime('%Y/%m/%d %H:%M:%S')
        self_str = f'{time_str} {self.wavelength}A Macropixel {self.indices[0]}, {self.indices[1]}'
        return self_str

    def mp_fill_processed(self, value):
        '''
        Helper function that assigns a value to the processed attribute, describing the macropixels
        value after detrending and normalization.
        -----------
        Parameters:
        -----------
        value : float
            The value to be assigned to self.processed.
        '''
        self.processed = value

    def get_pixel_locations(self):
        '''
        Helper function that returns the locations of all 9 pixels used to create the Macropixel.
        --------
        Returns:
        --------
        pixel_locations : list of (int, int) tuples
            The locations of each pixel.
        '''
        pixel_locations = []
        center_x = 3 * self.indices[0] + 1
        center_y = 3 * self.indices[1] + 1
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                pixel_locations.append((center_x + i, center_y + j))
        return pixel_locations

class Cluster:
    ''''
    Arranges spatially adjacent potentially significant Macropixel instances for one timestamp.
    -----------
    Attributes:
    -----------
    elements : list of Macropixels
        The Macropixels that make up the spatial extent of the identified event.
    timestamp : float
        The Unix timestamp of the Cluster.
    wavelength : int
        The AIA channel from which the data came.
    '''

    def __init__(self, seed):
        '''
        Creates an instance of the Cluster class.
        -----------
        Parameters:
        -----------
        seed : Macropixel
            The Macropixel that was identified and led to the creation of this Cluster.
        timestamp : float
            The Unix timestamp of the Cluster.
        wavelength : int
            The AIA channel from which the data came.
        '''
        self.elements = []
        self.elements.append(seed)
        self.timestamp = seed.timestamp
        self.wavelength = seed.wavelength

    def __str__(self):
        '''
        Produces a String representation of the instance.
        --------
        Returns:
        --------
        self_str : String
            The String representation.
        '''
        self_str = f'Cluster around {self.elements[0]}'
        return self_str

    def add_macropixel(self, new):
        '''
        Appends a Macropixel to the Cluster.
        -----------
        Parameters:
        -----------
        new : Macropixel
            The Macropixel to be added to the Cluster.
        '''
        self.elements.append(new)

    def get_cluster_brightness(self):
        '''
        Returns the combined brightness of all of the pixels contained in the Cluster.
        --------
        Returns:
        --------
        total_brightness : int
            The sum of the brightnesses of all pixels in the Cluster.
        '''
        total_brightness = 0
        for macropixel in self.elements:
            #Macropixel brightness is the mean brightness of 9 pixels, we want the total brightness
            total_brightness += int(macropixel.brightness * 9)
        return total_brightness

    def get_all_pixel_locations(self):
        '''
        Returns a list specifying the locations of all pixels contained in the Macropixels making
        up this Cluster.
        --------
        Returns:
        --------
        locations_list : list of (int, int) tuples
            A list containing the locations of all pixels.
        '''
        locations_list = []
        for macropixel in self.elements:
            locations_list += macropixel.get_pixel_locations()
        return locations_list

    def check_overlap(self, other):
        '''
        Determines whether self shares any Macropixels with the input Cluster.
        -----------
        Parameters:
        -----------
        other : Cluster
            The Cluster with which we want to compare.
        --------
        Returns:
        --------
        bool_overlapping : Boolean
            True if self and other share any Macropixels. False otherwise.
        '''
        self_indices = [mp.indices for mp in self.elements]
        other_indices = [mp.indices for mp in other.elements]
        for indices in self_indices:
            if indices in other_indices:
                return True
        return False

class Event:
    '''
    An object representing a brightening determined to be significant. Consists of a list of
    spatially overlapping, adjacent in time Cluster objects. Fully describes the spatial and
    temporal extent of a flare.
    -----------
    Attributes:
    -----------
    clusters : list of Cluster objects
        The instantaneous Clusters making up the Event, ordered in time.
    '''

    def __init__(self, clusters):
        '''
        Creates an Event instance from an input list of Clusters.
        -----------
        Parameters:
        -----------
        clusters : list of Cluster of objects
            The Clusters from which to create the event.
        '''
        self.clusters = clusters
        self.clusters.sort(key = cluster_sort_func)

    def __str__(self):
        '''
        Produces a String representation of the instance.
        --------
        Returns:
        --------
        self_str : String
            The String representation.
        '''
        start_str = epoch_to_str(self.clusters[0].timestamp)
        end_str = epoch_to_str(self.clusters[-1].timestamp)
        wavelength = self.clusters[0].wavelength
        self_str = f'Significant Event from {start_str} to {end_str} ({wavelength} A)'
        return self_str

    def add_clusters(self, new_list):
        '''
        Adds a list of  Clusters to the Event and ensures that it is ordered in time.
        -----------
        Parameters:
        -----------
        new_list : list of Cluster objects.
            The Clusters to be added.
        '''
        self.clusters += new_list
        self.clusters.sort(key = cluster_sort_func)

    def get_cluster_at_t(self, timestamp):
        '''
        Returns the Cluster with the given timestamp if such a Cluster exists.
        -----------
        Parameters:
        -----------
        timestamp : float
            The Unix timestamp of the Cluster we wish to acccess.
        --------
        Returns:
        --------
        cluster : Cluster
            The Cluster at the the timestamp, if it exists. Returns 0 if it does not exist.
        '''
        for cluster in self.clusters:
            if cluster.timestamp == timestamp:
                return cluster
        #Returning None if no Cluster was found at the given timestamp
        return None

class Instant:
    '''
    Contains the image data for one specific time, wavelength, and region of AIA data.
    -----------
    Attributes:
    -----------
    timestamp : float
        The unix timestamp of the data.
    dt : datetime.datetime
        The datetime object (UTC) corresponding to timestamp of the fits file from which the object
        was created.
    timestr : str
        A formatted time string used in plotting and string representation.
    wavelength : int
        The AIA channel from which the data came.
    center : (int, int)
        The coordinates (x, y), in arcseconds, of the center of the region.
    radius : int
        The radius, in arcseconds, of the region.
    fname : str
        The filepath of the FITS file used to generate the specific Instant in question.
    macropixels : 2D array of Macropixel objects
        The image data stored in Macropixel objects
    clusters : list of Cluster objects
        The list of Clusters describing significantly bright regions
    brightness : 2D N x N array of ints
        The raw brightness data acquired by FIDO from AIA level 1 data.
    flares : 2D N x N array of ints
        The results of brightening identification. Regions with significant brightenings are marked
        by 1s.
    q_d : int
        The detection threshold last used to identify brightenings in the Instant.
        (see the identify_brightenings method)
    q_c : int
        The connection threshold last used to identify brightenings in the Instant.
        (see the identify_brightenings method)
    '''

    def __init__(self, desired_time, wavelength, center, radius):
        '''
        Creates an Instant object by downloading the necessary FITS file and storing the brightness
        data for the region specified by center and radius in the brightness 2D array.
        -----------
        Parameters:
        -----------
        desired_time : float
            The unix timestamp of the approximate time for which we want to query data.
        wavelength : int
            The AIA channel from which these data come.
        center : (int, int)
            The center, in arcseconds, of the region from which the data came.
        radius : int
            The radius, in arcseconds, of the region.
        '''
        self.fname = get_fits(desired_time, wavelength)
        dtstr = self.fname.split('/')[-1].split('_')[1] + self.fname.split('/')[-1].split('_')[2]
        self.dt = datetime.strptime(dtstr, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        self.timestamp = self.dt.timestamp()
        self.timestr = self.dt.strftime(TIME_STR_FORMAT)
        self.wavelength = wavelength
        self.center = center
        self.radius = radius
        #Getting the FITS file at fname for the specified subregion
        m = sunpy.map.Map(self.fname)
        #Creating the region specified by center and radius
        bl_x, bl_y = center[0] - radius, center[1] - radius
        bl = SkyCoord(bl_x*u.arcsecond, bl_y*u.arcsecond, frame=m.coordinate_frame)
        tr_x, tr_y = center[0] + radius, center[1] + radius
        tr = SkyCoord(tr_x*u.arcsecond, tr_y*u.arcsecond, frame=m.coordinate_frame)
        subm = m.submap(bottom_left=bl, top_right=tr)
        reg = CircleSkyRegion(SkyCoord(*center, unit='arcsecond', frame=subm.coordinate_frame),
                                radius*u.arcsecond)
        mask = (reg.to_pixel(subm.wcs)).to_mask()
        #Getting the values from the FITS file
        self.brightness = []
        for xcoord in range(mask.data.shape[0]):
            self.brightness.append([])
            for ycoord in range(mask.data.shape[1]):
                self.brightness[xcoord].append(subm.data[xcoord, ycoord])
        #Averaging the FITS data for 3x3 squares to make Macropixels
        self.macropixels = []
        for i in range(len(self.brightness) // 3):
            self.macropixels.append([])
            for j in range(len(self.brightness) // 3):
                #Floor division ensures no indices are out of bounds
                #The last 1 or 2 pixels in either direction may be lost
                x_range = [3 * i, 3 * i + 1, 3 * i + 2]
                y_range = [3 * j, 3 * j + 1, 3 * j + 2]
                pixel_values = []
                for x in x_range:
                    pixel_values += [self.brightness[x][y] for y in y_range]
                avg_brightness = np.mean(pixel_values)
                #Creating a Macropixel object with the average brightnes of the 3x3 array of pixels
                new_mp = Macropixel(self.timestamp, self.wavelength, (i, j), avg_brightness)
                self.macropixels[i].append(new_mp)
        #Setting clusters and flares to empty arrays, for now
        self.clusters = []
        self.flares = []
        #Setting qd and qc to None. Their values are updated when identify_brightenings is run
        self.q_d = None
        self.q_c = None

    def __str__(self):
        '''
        Produces a String representation of the instance.
        --------
        Returns:
        --------
        self_str : String
            The String representation.
        '''
        self_str = f'{self.wavelength}A {self.timestr} brightness'
        if self.macropixels[0][0].processed is not None:
            self_str += ', processed'
            if len(self.flares) > 0:
                self_str += ', and flares '
        else:
            if len(self.flares) > 0:
                self_str += ' and flares '
        self_str += ' data'
        return self_str

    def get_mp_brightness(self):
        '''
        A helper function that returns the brightness data for the Macropixels making up the
        Instant.
        --------
        Returns:
        --------
        brightness : 2D array of floats
            The brightness data for the Macropixels in this Instant, organized according to the
            organization of self.macropixels
        '''
        brightness = []
        for i, row in enumerate(self.macropixels):
            brightness.append([])
            for macropixel in row:
                brightness[i].append(macropixel.brightness)
        return brightness

    def ins_fill_processed(self, data):
        '''
        A helper function that sets the processed value of each Macropixel making up the Instant.
        -----------
        Parameters:
        -----------
        data : 2D array of floats
            The brightness data of this Instant's Macropixels after being boxcarred and normalized
            over some interval.
        '''
        for i, row in enumerate(self.macropixels):
            for j, macropixel in enumerate(row):
                macropixel.mp_fill_processed(data[i][j])

    def reset_clusters(self):
        '''
        Reinitializes self.clusters to an empty array.
        '''
        self.clusters = []

    def add_cluster(self, cluster):
        '''
        Adds the input Cluster object to self.clusters
        -----------
        Parameters:
        -----------
        cluster : Cluster
            The Cluster to be added.
        '''
        self.clusters.append(cluster)

    def fill_flares(self):
        '''
        Marks the locations of significant brightenings in a 2D array self.flares. 1s denote pixels
        in a Cluster, 0s denote nonsignificant pixels. Uses the locations of the Macropixels
        contained in Clusters to determine the significant locations.
        '''
        self.flares = []
        cluster_locations = []
        for cluster in self.clusters:
            cluster_locations += cluster.get_all_pixel_locations()
        for i, row in enumerate(self.brightness):
            self.flares.append([])
            for j, _ in enumerate(row):
                if (i, j) in cluster_locations:
                    self.flares[i].append(1)
                else:
                    self.flares[i].append(0)

    def identify_brightenings(self, q_d, q_c):
        '''
        Identifies the coordinates of all significant brightenings in data, as determined by
        thresholds q_d and q_c. q_d is the detection threshold a pixel must meet to be identified
        as significant, and q_c is the connection threshold for a pixel to be identified as
        significant when it has an already significant neighbor. Creates Clusters around each
        detected Macropixel and adds them to self.clusters.
        -----------
        Parameters:
        -----------
        q_d : int
            The number of standard deviations above the mean brightness for a certain pixel to be
            DETECTED as a brightening at a certain time.
        q_c : int
            The number of standard deviations above the mean brightness for a certain pixel to be
            CONNECTED to an already identified neighboring brightening event.
        '''
        #Checking that the brightness data has been preprocessed for brightening identification
        if self.macropixels[0][0].processed is None:
            print('Ensure data has been processed. Exitting.')
            return 0
        self.clusters = []
        self.q_d = q_d
        self.q_c = q_c
        #Iterating through the macropixels array to find Macropixels that are above q_d
        for row in self.macropixels:
            for macropixel in row:
                if macropixel.processed >= q_d:
                    #Making sure that this Macropixel isn't already in a Cluster
                    bool_in_cluster = False
                    for cluster in self.clusters:
                        if macropixel in cluster.elements:
                            bool_in_cluster = True
                    if not bool_in_cluster:
                        #Creating a new Cluster and recursively checking neighboring Macropixels
                        new_cluster = Cluster(macropixel)
                        significant_neighbors = self.check_connection(macropixel, q_c, [macropixel])
                        for new_macropixel in significant_neighbors:
                            new_cluster.add_macropixel(new_macropixel)
                        #Only adding Clusters that are 2 Macropixels or larger
                        if len(new_cluster.elements) >= 2:
                            self.clusters.append(new_cluster)

    def check_connection(self, macropixel, q_c, already_connected):
        '''
        Checks the Macropixels directly adjacent to the input Macropixel against the input
        connection threshold to determine if they are significantly bright or not. Recursively
        checks Macropixels adjacent to any connected Macropixel.
        -----------
        Parameters:
        -----------
        macropixel : Macropixel
            A Macropixel detected as significant, whose neighbors we wish to check.
        q_c : int
            The connection threshold against which Macropixel processed values are checked.
        already_connected : list of Macropixels
            A list of the Macropixels already connected. Passed in to stop infinite recursion
        --------
        Returns:
        --------
        connected_list : list of Macropixel objects
            The Macropixel objects making up the significantly bright region surrounding the input
            Macropixel
        '''
        connected_list = already_connected
        #Getting the indices of all adjacent Macropixels
        center = macropixel.indices
        left = (center[0] - 1, center[1])
        right = (center[0] + 1, center[1])
        below = (center[0], center[1] - 1)
        above = (center[0], center[1] + 1)
        indices_to_check = [left, right, below, above]
        #Checking each neighboring Macropixel if it is not out of bounds
        for indices in indices_to_check:
            if indices[0] in range(len(self.macropixels)):
                if indices[1] in range(len(self.macropixels[0])):
                    macropixel_to_check = self.macropixels[indices[0]][indices[1]]
                    if macropixel_to_check.processed >= q_c:
                        #Ensuring we haven't already counted the Macropixel being checked
                        if macropixel_to_check not in connected_list:
                            connected_list.append(macropixel_to_check)
                            #Checking the Macropixels adjacent to a new significant Macropixel
                            next_neighbors = self.check_connection(macropixel_to_check, q_c, connected_list)
                            for new_macropixel in next_neighbors:
                                if new_macropixel not in connected_list:
                                    connected_list.append(new_macropixel)
        return connected_list

    def plot_brightness_map(self, bool_flare_id, bool_show_fig, bool_save_fig):
        '''
        Produces a sunpy image of the brightness data. Can be told to include an outline of the
        identified brightenings, if possible. Shows the figure, and it can be saved.
        -----------
        Parameters:
        -----------
        bool_flare_id : Boolean
            Specifies whether or not to outline the identified significant brightenings, if
            identify_brightenings has been run and self.flares is full of data.
        bool_show_fig : Boolean
            Specifies whether or not to show the plot when it is created.
        bool_save_fig : Boolean
            Specifies whether the plot should be saved and its path returned as a String.
        --------
        Returns:
        --------
        fname : str
            The relative path of the saved plot, if requested.
        '''
        #Creating a submap for the region of this Instant
        m = sunpy.map.Map(self.fname)
        bl_x, bl_y = self.center[0] - self.radius, self.center[1] - self.radius
        bl = SkyCoord(bl_x*u.arcsecond, bl_y*u.arcsecond, frame=m.coordinate_frame)
        tr_x, tr_y = self.center[0] + self.radius, self.center[1] + self.radius
        tr = SkyCoord(tr_x*u.arcsecond, tr_y*u.arcsecond, frame=m.coordinate_frame)
        subm = m.submap(bottom_left=bl, top_right=tr)

        #Plotting
        plt.ioff()
        _, ax = plt.subplots(figsize=(12,12), subplot_kw={'projection':subm})
        #If want reverse colormap, re-add parameter 'cmap=m.cmap.reversed()'
        subm.plot(axes=ax)
        subm.draw_limb(color='black')
        ax.tick_params(which='major', direction='in')
        ax.grid(False)
        ax.set(xlabel=MAP_X_LABEL, ylabel=MAP_Y_LABEL)
        (ax.coords[0]).display_minor_ticks(True)
        (ax.coords[0]).set_minor_frequency(5)
        (ax.coords[1]).display_minor_ticks(True)
        (ax.coords[1]).set_minor_frequency(5)
        ax.tick_params(which='minor', length=1.5)
        plt.colorbar(fraction=0.046, pad=0.02)
        #Adding contour lines for significant brightenings, if requested
        if bool_flare_id:
            self.fill_flares()
            ax.contour(self.flares, [0,1])
        #Generating a filename and saving the figure, if requested
        if bool_save_fig:
            fname = f"{DATA_DIR}{self.fname.split('/')[-1].split('_')[1]}/images/" +\
                    f"{self.fname.split('/')[-1].split('_')[2]}_submap"
            if bool_flare_id:
                fname += f'_flare-id_d{self.q_d}c{self.q_c}'
            fname += '.png'
            plt.savefig(fname)
        if bool_show_fig:
            print('Showing figure:')
            plt.show()
        plt.close()
        if bool_save_fig:
            return fname

    def write_to_file(self):
        '''
        Uses the pickle module to write an Instant object to a binary file.
        --------
        Returns:
        --------
        fname : str
            The relative path to the file.
        '''
        fname = f'{DATA_DIR}{self.dt.strftime("%Y%m%d")}/Instants/{self.dt.strftime("%H%M%S")}' +\
                f'_{self.wavelength:04d}A_brightness'
        if self.macropixels[0][0].processed is not None:
            fname += '_processed'
        if len(self.flares) > 0:
            fname += '_flares'
        fname += '.dat'
        f = open(fname, 'wb')
        pickle.dump(self, f)
        f.close()
        return fname

class Interval:
    '''
    Contains a sequence of Instant objects from the same channel and subregion, making up a time
    series of AIA image data.
    -----------
    Attributes:
    -----------
    time_series : list of Instant objects
        The ordered list of Instant objects contained in this interval.
    events : list of Event objects
        A list of the significant Events present in the Interval.
    '''

    def __init__(self, ins_list):
        '''
        Creates an instance of the Interval class.
        -----------
        Parameters:
        -----------
        time_series : list of Instant objects or single Instant object
            The collection (does not need to be ordered) of Instant objects from which the Interval
            is to be created.
        '''
        self.time_series = []
        if ins_list is Instant:
            self.time_series.append(Instant)
        else:
            for ins in ins_list:
                #Checking to see that it makes sense to combine all the Instants in ins_list
                #By default, all Instants are checked against the first entry of ins_list
                same_wavelength = True
                same_subregion = True
                if ins.wavelength != ins_list[0].wavelength:
                    print('Ensure all Instants being combined are from the same channel: ' +
                            f'{ins_list[0].wavelength} A')
                    same_wavelength = False
                if ins.radius != ins_list[0].radius or ins.center != ins_list[0].center:
                    print('Ensure all Instants being combined described the same subregion: ' +
                            f'Center {ins_list[0].center} with radius {ins_list[0].radius}')
                    same_subregion = False
                if same_wavelength and same_subregion:
                    self.time_series.append(ins)
        #Sorting the Instants from earliest to latest
        self.time_series.sort(key = ins_sort_func)
        #Getting rid of any duplicate (same time) Instants
        self.time_series = [self.time_series[0]] +\
                            [self.time_series[i+1] for i in range(len(self.time_series)-1)
                            if self.time_series[i+1].timestamp != self.time_series[i].timestamp]
        #Initializing events to an empty array
        self.events = []

    def __str__(self):
        '''
        Produces a String representation of the instance.
        --------
        Returns:
        --------
        self_str : String
            The String representation.
        '''
        if len(self.time_series) == 1:
            return self.time_series[0]
        else:
            start_str = epoch_to_str(self.time_series[0].timestamp)
            end_str = epoch_to_str(self.time_series[-1].timestamp)
            self_str = f'{self.time_series[0].wavelength} A {start_str} to {end_str}. ' +\
                        f'{len(self.time_series)} Instants'
            return self_str

    def __add__(self, other):
        '''
        Method for concatenating Interval objects with the same wavelength and subregion.
        Individual Instance objects can also be added; they are treated as an Interval with one
        element.
        -----------
        Parameters:
        -----------
        self : Interval
            'self'-explanatory
        other : Interval or Instance
            The object to be combined with self.
        --------
        Returns:
        --------
        comb : Interval
            The combination of self and other after being sorted in time with duplicates removed.
        '''
        if other is Instant:
            other = Interval(Instant)
        #Checking that it makes sense to combine self and other
        if self.time_series[0].wavelength != other.time_series[0].wavelength:
            print('Ensure both Intervals describe the same channel ' +\
                    f'({self.time_series[0].wavelength} A). Exitting.')
            return 0
        if self.time_series[0].center != other.time_series[0].center:
            print('Ensure both Intervals describe the same subregion (Center ' +\
                    f'{self.time_series[0].center} with radius {self.time_series[0].radius}). ' +\
                    'Exitting.')
            return 0
        if self.time_series[0].radius != other.time_series[0].radius:
            print('Ensure both Intervals describe the same subregion (Center ' +\
                    f'{self.time_series[0].center} with radius {self.time_series[0].radius}). ' +\
                    'Exitting.')
            return 0
        #Creating a list of all Instants in self and other, and making an Interval from it
        instants = []
        for ins in self.time_series:
            instants.append(ins)
        for ins in other.time_series:
            instants.append(ins)
        comb = Interval(instants)
        return comb

    def boxcar_and_normalize(self, N, bool_report):
        '''
        Performs a boxcar average on each single-pixel lightcurve in the subregion, over the time
        period for which there are data in the Interval. A constant sample spacing is assumed. The
        boxcar is subtracted from the brightness data, and the resulting detrened lightcurves are
        mean-subtracted and normalized by their standard deviation. The resulting processed data is
        stored in the 'processed' 2D array attribute of each Instant object in the Interval.
        -----------
        Parameters:
        -----------
        N : int
            The desired boxcar width. Will be adjusted if too large. Must be odd.
        bool_report : Boolean
            Specifies whether to print what N is being used when the boxcar method starts to run.
        '''
        #Calculating the correct boxcar width
        N = adjust_n(len(self.time_series), N, bool_report)
        #Creating a 3D array to store the 2D brightness data of each Instant in order
        brightness_time_series = []
        for ins in self.time_series:
            brightness_time_series.append(ins.get_mp_brightness())
        #Calculating and subtracting off the moving average with width N
        avg = moving_average(brightness_time_series, N, bool_report)
        raw = brightness_time_series[int((N-1)/2):int(-1*(N-1)/2)]
        residuals = raw - avg
        #Creating the skeleton of the final processed data array
        processed = []
        for t in range(len(residuals)):
            processed.append([])
            for x in range(len(residuals[0])):
                processed[t].append([])
                for y in range(len(residuals[0][0])):
                    processed[t][x].append(0)
        #Processing each pixel's lightcurve by subtracting its mean and normalizing by its st. dev.
        for x in range(len(residuals[0])):
            for y in range(len(residuals[0][0])):
                pixel_lc = [residuals[t][x][y] for t in range(len(residuals))]
                std = np.std(pixel_lc)
                mean = np.mean(pixel_lc)
                #If the st. dev. of a pixel's lightcurve is 0, the pixel's processed values are 0
                if std != 0:
                    for t, processed_ins in enumerate(processed):
                        processed_ins[x][y] = (pixel_lc[t] - mean) / std
        #Removing the unprocessed Instants on either end of the Interval
        self.time_series = self.time_series[int((N-1)/2):int(-1*(N-1)/2)]
        #Populating the processed data for all of the Macropixels in each Instant
        for i, ins in enumerate(self.time_series):
            ins.ins_fill_processed(processed[i])

    def get_next_overlapping_clusters(self, index, cluster):
        '''
        Used to create an Event object. Recursively looks forward in time to find Clusters that
        overlap with the input Cluster.
        -----------
        Parameters:
        -----------
        index : int
            The index of the Instant in self.time_series in which the input Cluster is found.
        cluster : Cluster
            The Cluster used to check overlap of subsequent Clusters.
        --------
        Returns:
        --------
        clusters_list : list of Cluster objects
            The list of overlapping, temporally adjacent Clusters to be added to an Event with the
            input Cluster.
        '''
        clusters_list = []
        #Looking through the Clusters in the Instant at index+1 to check overlap
        if index + 1 < len(self.time_series):
            for c in self.time_series[index + 1].clusters:
                if cluster.check_overlap(c):
                    clusters_list.append(c)
                    clusters_list += self.get_next_overlapping_clusters(index + 1, c)
        return clusters_list


    def identify_brightenings_all(self, q_d, q_c):
        '''
        Performs identify_brightenings() on all Instants in self.time_series subject to the given
        detection and connection thresholds. Uses the Clusters created in each Instant to construct
        Events.
        -----------
        Parameters:
        -----------
        q_d : int
            Detection threshold as described in identify_brightenings.
        q_c : int
            Connection threshold as described in identify_brightenings.
        '''
        self.events = []
        #Creating Clusters in each Instant
        for ins in self.time_series:
            ins.identify_brightenings(q_d, q_c)
        #Creating Events from the produced Clusters
        for i, ins in enumerate(self.time_series[:-1]):
            for cluster1 in ins.clusters:
                for cluster2 in self.time_series[i+1].clusters:
                    #Making Events out of regions that are significantly bright in at least two
                    #temporally adjacent Instants
                    if cluster1.check_overlap(cluster2):
                        new_event = Event([cluster1, cluster2])
                        #Looking forward in time to get the rest of the Event
                        subsequent_clusters = self.get_next_overlapping_clusters(i+1, cluster2)
                        new_event.add_clusters(subsequent_clusters)
                        self.events.append(new_event)
        #Updating each Instant to get rid of insignificant Clusters
        for ins in self.time_series:
            ins.reset_clusters()
            significant_clusters = [event.get_cluster_at_t(ins.timestamp) for event in self.events]
            for cluster in significant_clusters:
                if cluster is not None:
                    ins.add_cluster(cluster)
            ins.fill_flares()


    def make_movie(self, bool_flare_id):
        '''
        Creates a .mp4 video file out of the brightness maps of each Instant in the Interval. Video
        is saved and the relative path is returned by the method.
        -----------
        Parameters:
        -----------
        bool_flare_id : Boolean
            Specifies whether identified flares are outlined in the images.
        --------
        Returns:
        --------
        mp4_fname : str
            The relative path of the video file created.
        '''
        #Saving the images for each Instant and storing the filenames in a list
        paths = [ins.plot_brightness_map(bool_flare_id, False, True) for ins in self.time_series]
        #Generating the movie filename
        mp4_fname = f'{DATA_DIR}{self.time_series[0].dt.strftime("%Y%m%d")}/movies/'
        start_time = self.time_series[0].dt.strftime("%H%M%S")
        end_time = self.time_series[-1].dt.strftime("%H%M%S")
        wavelength = self.time_series[0].wavelength
        mp4_fname += f'{start_time}-{end_time}_{wavelength:04d}A_movie'
        #Only denoting flare identification in the movie fname if it was actually performed
        if bool_flare_id and len(self.time_series[0].flares) != 0:
            qd_used = self.time_series[0].q_d
            qc_used = self.time_series[0].q_c
            mp4_fname += f'_flare-id_d{qd_used}c{qc_used}'
        mp4_fname += '.mp4'
        #If a movie already exists at fname, it is deleted so that the new movie can be saved
        if os.path.exists(mp4_fname):
            os.remove(mp4_fname)
        #Making the movie
        writer = imageio.get_writer(mp4_fname, fps=10)
        for path in paths:
            writer.append_data(imageio.imread(path))
        writer.close()
        return mp4_fname

    def write_to_file(self):
        '''
        Uses the pickle module to write an Interval object to a binary file. If the Interval is too
        large for the system Memory, it is recursively split in half until the Intervals being
        pickled are of an acceptable size.
        --------
        Returns:
        --------
        fname_list : list of str
            The relative path(s) to the file(s) saved as a result of this process.
        '''
        fname_list = []
        fname = f'{DATA_DIR}{self.time_series[0].dt.strftime("%Y%m%d")}/Intervals/'
        start_time = self.time_series[0].dt.strftime("%H%M%S")
        end_time = self.time_series[-1].dt.strftime("%H%M%S")
        wavelength = self.time_series[0].wavelength
        fname += f'{start_time}-{end_time}_{wavelength:04d}_brightness'
        if len(self.time_series[0].processed) > 0:
            fname += '_processed'
        if len(self.time_series[0].flares) > 0:
            q_d = self.time_series[0].q_d
            q_c = self.time_series[0].q_c
            fname += f'_flares-d{q_d}c{q_c}'
        fname += '.dat'
        try:
            f = open(fname, 'wb')
            pickle.dump(self, f)
            f.close()
            fname_list.append(fname)
            return fname_list
        except MemoryError:
            #If the Interval is too large to save, it is recursively divided in half
            first_half = Interval(self.time_series[:len(self.time_series)//2])
            second_half = Interval(self.time_series[len(self.time_series)//2:])
            fname_list += first_half.write_to_file()
            fname_list += second_half.write_to_file()
            return fname_list

def make_interval(start_time, end_time, wavelength, center, radius):
    '''
    Creates an Interval object with the given specifications. Handles iterative creation of
    Instants to make interactive computing more streamlined. In classes.py instead of helpers.py
    to avoid circular import issues.
    -----------
    Parameters:
    -----------
    start_time : Datetime
        A datetime object representing the desired start of the Interval (UTC).
    end_time : Datetime
        A datetime object representing the desired end of the Interval (UTC).
    wavelength : int
        The desired wavelength (in Angstrom).
    center : (int, int)
        The center coordinates (x, y), in arcseconds, of the region we want to analyze.
    radius : int
        The radius, in arcseconds, of the region we want to analyze.
    --------
    Returns:
    --------
    interval : Interval object
        The created Interval object.
    '''
    time_i = start_time
    ins_list = []
    while time_i <= end_time:
        ins_list.append(Instant(time_i.timestamp(), wavelength, center, radius))
        time_i += timedelta(seconds=12)
    interval = Interval(ins_list)
    return interval
