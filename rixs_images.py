import os
import time
import h5py
import numpy as np
# from scipy.signal import fftconvolve
from scipy.signal import medfilt2d, correlate
from scipy.ndimage import median_filter
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
# from scipy import optimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from cmcrameri import cm
from abc import ABC, abstractmethod
import fabio
from nexusformat.nexus import *
import re


class RIXS_Image(ABC):

    def __init__(self):
        self.imgs_processed = None
        self.normalization_factor = None

    @abstractmethod
    def _get_raw_data(self):
        pass

    @abstractmethod
    def _get_run_number(self):
        pass

    @abstractmethod
    def _get_normalization_factor(self):
        pass

    @abstractmethod
    def _get_attributes(self):
        pass

    @abstractmethod
    def _get_energy(self):
        pass

    def process_imgs(self,
                     use_spc,
                     spc_parameters={},
                     no_spc_parameters={},
                     ):
        
        if use_spc:
            self.single_photon_counting(**spc_parameters)
        else:
            self._remove_bkg_and_filter(**no_spc_parameters)

        return self.imgs_processed, self.normalization_factor


    @abstractmethod
    def _remove_bkg_and_filter(self,
                               **kwargs):
        pass

    def single_photon_counting(
            self, 
            curve_a, curve_b=0,
            roi_x=(0,2048),roi_y=(0,2048),
            roi_x_for_dark=(1600,1800), roi_y_for_dark=(250,1800),
            subdivide_bins_factor_x=1, subdivide_bins_factor_y=2.7,
            factor_ADC=0.55,
            vertical_shift = 0,
            subtract_background_from_img=False,
            subtract_background_from_corner=False,
            bkg=0,
            dark_img=None,
            plot_raw_image=False):
        """
        Process the raw image data using single photon counting technique.
        Uses the centroid function to identify and locate photon hits.
        
        Parameters
        ----------
        curve_a : float
            Coefficient that represents the linear (first-order) coefficient of the slope
        curve_b : float
            Coefficient that represents the quadratic (2nd-order) coefficient of the slope
        roi_x : tuple of int
            Region of interest along the x-axis
        roi_y : tuple of int
            Region of interest along the y-axis
        roi_x_for_dark : tuple of int
            Region of interest along the x-axis for dark image subtraction
        roi_y_for_dark : tuple of int
            Region of interest along the y-axis for dark image subtraction
        factor_ADC : float
            Factor to convert ADU to electrons
        subdivide_bins_factor_x : float, optional
            Factor by which to subdivide the bins along the x-axis
        subdivide_bins_factor_y : float, optional
            Factor by which to subdivide the bins along the y-axis
        vertical_shifts : float
            vertical shifts to be subtracted from y-coordinates, coming from 
            cross-correlation. leave empty if cross-corr has not been done yet.
        subtract_background_from_img : bool, optional
            If True, subtract background from a pre-processed dark image
        dark_img : numpy.ndarray, optional
            Dark image to be subtracted from the raw image data
            
        Returns
        -------
        numpy.ndarray
            2D histogram array containing the photon counts in each bin
        """
        # Make sure we have raw data to process
        if self.raw_data is None:
            raise ValueError("No raw data available for processing")
        
        if self.raw_data.ndim < 3:
                self.raw_data = np.expand_dims(self.raw_data, axis=0)

        # Crop the image according to roi_x and roi_y
        cropped_data = np.array([img[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]] for img in self.raw_data])
            
        # Use the centroid function to process the image
        if self.res is None:
            
            if subtract_background_from_corner:
                # Used for TPS
                bkg = self.raw_data[:,50:-50,:70:100].mean()
                dark_img_cropped = 0
                mean_factors_dark = np.zeros(self.raw_data.shape[0])

            elif subtract_background_from_img:
                # Used for TPS, possibly
                if dark_img is None:
                    raise ValueError("Dark image must be provided")
                bkg = 0
                mean_factors_dark = [np.mean(raw[roi_y_for_dark[0]:roi_y_for_dark[1], roi_x_for_dark[0]:roi_x_for_dark[1]]) - 
                                     np.mean(dark_img[roi_y_for_dark[0]:roi_y_for_dark[1], roi_x_for_dark[0]:roi_x_for_dark[1]]) for raw in self.raw_data]
                dark_img_cropped = dark_img[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]] 
            else:
                #used for ESRF
                dark_img_cropped = 0
                mean_factors_dark = np.zeros(self.raw_data.shape[0])

            self.raw_data = cropped_data

            for raw, norm_factor_dark in zip(self.raw_data, mean_factors_dark):
                res_now, _ = self._centroid(
                    img=raw - dark_img_cropped + norm_factor_dark,
                    energy=self.attributes["energy"],
                    bkg_mean=bkg,
                    factor_ADC=factor_ADC,
                    avoid_double=False,
                    curve_a=curve_a, 
                    curve_b=curve_b,
                )
                self.res = res_now if self.res is None else self.res + res_now              

        #bin into a grid
        self.imgs_processed, _ = self._bin(self.res, 
                            image_size_h=self.raw_data.shape[-1], 
                            image_size_v=self.raw_data.shape[-2], 
                            subdivide_bins_factor_x=subdivide_bins_factor_x, 
                            subdivide_bins_factor_y=subdivide_bins_factor_y,
                            vertical_shift = vertical_shift)


        # Plot the raw image if requested
        if plot_raw_image:
            self.plot_raw_image()

        return self.imgs_processed, self.normalization_factor
    

    def plot_raw_image(self, roi_y=(0,2048), roi_x=(0,2048)):
        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 2])

        # Plot the raw image
        ax2 = fig.add_subplot(gs[1, 1])
        img = self.raw_data[:,roi_y[0]:roi_y[1],roi_x[0]:roi_x[1]].sum(axis=0) \
            if self.raw_data.ndim == 3 else self.raw_data[roi_y[0]:roi_y[1],roi_x[0]:roi_x[1]]
        vmin = img.mean() - 3*img.std()
        vmax = img.mean() + 3*img.std()
        im = ax2.imshow(img, cmap=cm.lapaz, vmin=vmin, vmax=vmax)
        # fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Raw Image')
        ax2.set_aspect('auto')

        # Plot the vertical average
        ax1 = fig.add_subplot(gs[0, 1])
        vertical_avg = img.mean(axis=0)
        ax1.plot(vertical_avg)
        ax1.set_title('Vertical Average')
        ax1.set_ylim([vertical_avg.min()*0.9, vertical_avg.max()*1.1])
        # ax1.set_ylim([0, len(vertical_avg)])
        # ax1.invert_yaxis()

        # Plot the horizontal average
        ax3 = fig.add_subplot(gs[1, 0])
        horizontal_avg = img.mean(axis=1)
        ax3.plot(horizontal_avg, np.arange(len(horizontal_avg)))
        ax3.set_title('Horizontal Average')
        ax3.set_xlim([horizontal_avg.min(), horizontal_avg.max()])
        ax3.set_ylim([0, len(horizontal_avg)])
        ax3.invert_yaxis()
        ax3.invert_xaxis()

        plt.tight_layout()
        plt.show()

        return fig, (ax1, ax2, ax3)
    
    @staticmethod
    def _centroid(
        img,
        energy,
        bkg_mean=0,
        factor_ADC=1.2,
        factor_for_ghost_clouds = 0,
        avoid_double=False,
        curve_a=0,
        curve_b=0,
    ):
        """
        Parameters
        ----------
        img : ndarray
            2D detector image array containing photon hit data
        energy : float
            Energy of the incident photons in eV
        bkg_mean : float, optional
            Mean value of the flat background to subtract from image, default 300.52
        factor_ADC : float, optional
            Factor to convert ADU to electrons, default 0.56 (TPS uses electron-multiplied CCD)
        factor_for_ghost_clouds : float, optional
            Factor to discard ghost clouds, only present at TPS. If present, should be 200. For, ESRF, 0.
        factor_SpotLOW : float, optional
            Multiplication factor for low threshold (i.e. selects electron clouds whose sum is above this limit), default 0.4.
            This factor is 0.4 at ESRF, but here we need to discard this strange clouds and need to be set above 200
        avoid_double : bool, optional
            If True, ignore double photon events. If False or None, include them
        image_size : tuple of int
            Size of the 2D detector image as (height, width)
        subdivide_bins_factor_x : float, optional
            Factor by which to subdivide each pixel along x-axis for sub-pixel resolution, default 1.0
        subdivide_bins_factor_y : float, optional
            Factor by which to subdivide each pixel along y-axis for sub-pixel resolution, default 1.0

        Returns
        -------
        tuple
            (hist_p, photon_count) where:
            - hist_p is the 2D histogram of photon positions with sub-pixel resolution
            - photon_count is the total number of detected photons
        """

        SpotLOW = max(0.4 * energy / 3.6 / factor_ADC, factor_for_ghost_clouds)  # Multiplication factor * ADU/photon
        SpotHIGH = 1.5 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon
        low_th_px = 0.2 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon
        high_th_px = 1 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon

        if avoid_double == True:
            SpotHIGH = 100000
            print(
                "The double events are not taken into account, the double event threshold is set to "
            )
            print(SpotHIGH)


        img = img - bkg_mean

        gs = 2
        cp = np.argwhere(
            (img[gs // 2 : -gs // 2, gs // 2 : -gs // 2] > low_th_px)
            * (img[gs // 2 : -gs // 2, gs // 2 : -gs // 2] < high_th_px)
        ) + np.array([gs // 2, gs // 2])


        res = []
        double = []

        for cy, cx in cp:
            spot = img[cy - gs // 2 : cy + gs // 2 + 1, cx - gs // 2 : cx + gs // 2 + 1]
            spot[spot < 0] = 0
            if (spot > img[cy, cx]).sum() == 0:
                mx = np.average(
                    np.arange(cx - gs // 2, cx + gs // 2 + 1), weights=spot.sum(axis=0)
                )
                my = np.average(
                    np.arange(cy - gs // 2, cy + gs // 2 + 1), weights=spot.sum(axis=1)
                )
                my -= (curve_a + curve_b * mx) * mx
                if (spot.sum() > SpotLOW) * (spot.sum() <= SpotHIGH):
                    res.append((my, mx))
                elif spot.sum() > SpotHIGH:
                    res.append((my, mx))
                    res.append((my, mx))
                    double.append((my, mx))

        res_shifted = [(my + 0.5, mx + 0.5) for (my, mx) in res]
        double_shifted = [(my + 0.5, mx + 0.5) for (my, mx) in double]
        
        return res_shifted, double_shifted
    
    @staticmethod
    def _bin(
        p_pos_list, 
        image_size_h, 
        image_size_v, 
        subdivide_bins_factor_x, 
        subdivide_bins_factor_y,
        vertical_shift = 0):
        """
        p_pos_list : list
            A list containing the (x, y) coordinates of photons.
        image_size_h : int
            The height of the 2D image.
        image_size_v : int
            The width of the 2D image.
        subdivide_bins_factor_x : float
            The number of sub-pixels in which each pixel is divided along the x-axis.
        subdivide_bins_factor_y : float
            The number of sub-pixels in which each pixel is divided along the y-axis.
        vertical_shifts : float, optional
            A float number representing the vertical shifts to be applied to the y-coordinates of the photon positions. 
            These shifts are typically derived from cross-correlation analysis.
            If cross-correlation has not been performed, this parameter can be left empty. 
        """
        # Check if p_pos_list is empty
        if not p_pos_list:
            # If empty, set hist_p to an array of zeros with shape (image_size_v, image_size_h)
            hist_p = np.zeros(
                (
                    int(image_size_v * subdivide_bins_factor_y),
                    int(image_size_h * subdivide_bins_factor_x),
                )
            )
            photon_count = 0
        else:
            # Convert p_pos_list to a NumPy array
            p_pos_array = np.array(p_pos_list)
            p_pos_array[:, 0] -= vertical_shift/subdivide_bins_factor_y

            # Define bin edges
            x_edges_pht = list(np.arange(0, image_size_h, 1 / subdivide_bins_factor_x)) + [
                image_size_h
            ]
            y_edges_pht = list(np.arange(0, image_size_v, 1 / subdivide_bins_factor_y)) + [
                image_size_v
            ]

            # Create histogram
            hist_p, _, _ = np.histogram2d(
                p_pos_array[:, 0], p_pos_array[:, 1], bins=(y_edges_pht, x_edges_pht)
            )
            photon_count = len(p_pos_array)

        return hist_p.astype(np.float32), photon_count


class EDF_Image(RIXS_Image):
    def __init__(self, file_path):
        """
        Initialize an EDF image object from a file path.

        Parameters
        ----------
        file_path : str
            Path to the EDF image file
        """

        super().__init__()
        self.file_path = file_path
        
        self.attributes = {}

        self.raw_data = self._get_raw_data()
        self.n_images = self.raw_data.shape[0]
        self._get_run_number()
        self._get_image_number()
        self._get_absolute_image_number()
        self._get_normalization_factor()
        self._get_attributes()

    def _get_raw_data(self):
        file = fabio.open(self.file_path)
        self.raw_data = np.flipud(file.data)
        self.header = self._reorganize_header(file.header)

        if self.raw_data.ndim == 2:
            self.raw_data = np.expand_dims(self.raw_data, axis=0)

        return self.raw_data


    def _get_run_number(self):
        """
        Get the run number from the header
        """
        self.run_number = int(self.header["scan_no"])
        return self.run_number
    
    def _get_image_number(self):
        """
        Get the image number from the header
        """
        self.image_number = int(self.header["point_no"])
        return self.image_number
    
    def _get_absolute_image_number(self):
        """
        Get the absolute image number from the header
        """
        # self.absolute_image_number = int(self.header["run"])
        self.absolute_image_number = int(self.file_path[-8:-4])
        return self.absolute_image_number

    def _get_normalization_factor(self):
        """
        Get the normalization factor from the header
        """

        self.normalization_factor = self.header['mir'] / 1E6

        return self.normalization_factor
    
    def _get_energy(self):
        """
        Get the energy from the header
        """
        if "energy" not in self.attributes:
            raise ValueError("No energy found in the header")
        else:
            self.energy = self.attributes["energy"]
        
        return self.energy
    
    def _get_attributes(self):
        """
        Get the attributes from the header and map them using metadata_name_mapping.
        """

        metadata_name_mapping = {
            "th": "th",
            "chi": "chi",
            "phi": "phi",
            "tth": "rtth",
            "energy": "energy",
            "x": "xsam",
            "y": "ysam",
            "z": "zsam",
            "T": "tstage",
        }
        self.attributes = {}

        # Map the attributes using metadata_name_mapping
        for key, mapped_key in metadata_name_mapping.items():
            if key in self.header:
                self.attributes[key] = self.header[mapped_key]

        # Retrieve polarization motor values if present
        # Try to get polarization motor values from either HU88AP/HU88CP or HU70AP/HU70CP
        if "HU88AP" in self.header:
            ap = float(self.header["HU88AP"])
            polarization = self._determine_polarization(ap, ap)
        elif "HU70AP" in self.header:
            ap = float(self.header["HU70AP"])
            polarization = self._determine_polarization(ap, ap)
        else:
            print("Warning: Polarization motor values not found. Polarization could not be determined.")
            polarization = "Unknown"

        self.attributes["polarization"] = polarization
        
        return self.attributes
    
    def single_photon_counting(self, 
            curve_a, curve_b=0,
            roi_x=(0,2048),roi_y=(0,2048),
            roi_x_for_dark=(1600,1800), roi_y_for_dark=(250,1800),
            subdivide_bins_factor_x=1, subdivide_bins_factor_y=2.7,
            factor_ADC=1.2,
            vertical_shift = 0,
            subtract_background_from_img=False,
            subtract_background_from_corner=False,
            dark_img=None,
            plot_raw_image=False):
        
        if subtract_background_from_img:
            print("Warning: ESRF images do NOT need a dark image. Not using it. Setting flat background to 300.52...")

        if factor_ADC != 1.2:
            print("Warning: factor_ADC set 1.2, standard for ESRF.")
        
        return super().single_photon_counting(curve_a=curve_a, 
                curve_b=curve_b,
                roi_x=roi_x,
                roi_y=roi_y,
                subdivide_bins_factor_x=subdivide_bins_factor_x, 
                subdivide_bins_factor_y=subdivide_bins_factor_y,
                factor_ADC=1.2,
                vertical_shift=vertical_shift,
                subtract_background_from_img=False,
                subtract_background_from_corner=False,
                bkg=300.52,
                plot_raw_image=plot_raw_image)
    
    @staticmethod
    def _reorganize_header(header):
        # Keys that need special handling (splitting into lists)
        keys_to_split = ['counter_mne', 'motor_mne']
        values_to_split = ['counter_pos', 'motor_pos']
        # Extract all header items as strings, except for the keys that need splitting
        header_dict = {k: str(v) for k, v in header.items() if k not in keys_to_split and k not in values_to_split}

        # Add the split keys separately as lists
        names = []
        for key in keys_to_split:
            if key in header:
                names += str(header[key]).split()

        values = []
        for key in values_to_split:
            if key in header:
                values += [float(x) for x in header[key].split()]

        for name, value in zip(names, values):
            header_dict[name] = value

        return header_dict
    
    @staticmethod
    def _determine_polarization(hu70ap, hu70cp):
        
        # Determine polarization based on motor positions
        if hu70cp > 30 and hu70ap > 30:
            polarization = 'LV'
        elif -2 < hu70cp < 2 and -2 < hu70ap < 2:
            polarization = 'LH'
        elif 2 <= hu70cp <= 30 and 2 <= hu70ap <= 30:
            polarization = 'C+'
        elif -30 <= hu70cp <= -2 and -30 <= hu70ap <= -2:
            polarization = 'C-'
        else:
            polarization = 'Unknown'
        
        return polarization

class TPS_Image(RIXS_Image):
    def __init__(self, file_path, file_path_background = None):
        super().__init__()
        self.file_path = file_path
        self.file_path_background = file_path_background if file_path_background is not None else None

        self._get_raw_data()
        self.n_images = self.raw_data.shape[0]
        self._get_attributes()
        self._get_normalization_factor()
        self._get_run_number()
        self._get_energy()

    def _get_raw_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        with h5py.File(self.file_path, "r") as file:
            self.raw_data = file["data"][:].astype(np.float32)
            if self.raw_data.ndim < 3:
                self.raw_data = np.expand_dims(self.raw_data, axis=0)

            self.raw_data = np.array([np.rot90(raw, k=-1) for raw in self.raw_data])
            if self.raw_data.ndim != 3:
                raise ValueError("The data must be 2D or 3D")
            print(f"Found {self.raw_data.shape[0]} images in the file {os.path.basename(self.file_path)}")


    def _get_run_number(self):
        """
        Get the run number from the image file.
        
        Returns
        -------
        int
            Run number of the image file
        """
        if "f" not in self.attributes:
            raise ValueError("No run number found in the image file")
        
        self.run_number = self.attributes["f"]
                

    def _get_attributes(self):
        """
        Get the attributes of the image file.
        
        Returns
        -------
        dict
            Dictionary containing the attributes of the image file
        """
        if self.file_path is not None:
            with h5py.File(self.file_path, "r") as file:
                header = file.attrs['header']

            self.attributes = {}
            for item in header.split(":")[1].split(","):
                name, value = item.strip().split(" ")
                self.attributes[name] = float(value)

        else:
            raise ValueError("No file loaded")

    def _get_energy(self):
        """
        Get the energy from the image file.
        
        Returns
        -------
        float
            Energy of the image file
        """
        if "agm" not in self.attributes:
            raise ValueError("No energy found in the image file")
        else:
            self.energy = self.attributes["agm"]


    def _get_normalization_factor(self):
        """
        Get the normalization factor from the image file.
        
        Returns
        -------
        float
            Normalization factor of the image file
        """
        with h5py.File(self.file_path, "r") as file:
            if "normalization_factor" in file.attrs:
                return file.attrs["normalization_factor"]
            
        if "Iph" not in self.attributes:
            raise ValueError("No normalization factor found in the image file")
        else:
            self.normalization_factor = self.attributes["Iph"]*1E9

    def single_photon_counting(
            self, 
            curve_a, curve_b=0,
            roi_x=(0,2048),roi_y=(0,2048),
            roi_x_for_dark=(1600,1800), roi_y_for_dark=(250,1800),
            subdivide_bins_factor_x=1, subdivide_bins_factor_y=2.7,
            factor_ADC=0.55,
            vertical_shift = 0,
            subtract_background_from_img=False,
            subtract_background_from_corner=False,
            dark_img=None,
            plot_raw_image=False):
        
        if factor_ADC != 0.55:
            print("Warning: factor_ADC should be 0.55 for TPS data.")

        if not subtract_background_from_img and not subtract_background_from_corner:
            print("Warning: No background subtraction is applied. This should be done for TPS data.")
        
        return  super().single_photon_counting( 
                        curve_a=curve_a,
                        curve_b=curve_b,
                        roi_x=roi_x,
                        roi_y=roi_y,
                        roi_x_for_dark=roi_x_for_dark,
                        roi_y_for_dark=roi_y_for_dark,
                        subdivide_bins_factor_x=subdivide_bins_factor_x, 
                        subdivide_bins_factor_y=subdivide_bins_factor_y,
                        factor_ADC=factor_ADC,
                        vertical_shift=vertical_shift,
                        subtract_background_from_img=subtract_background_from_img,
                        subtract_background_from_corner=subtract_background_from_corner,
                        dark_img=dark_img,
                        plot_raw_image=plot_raw_image)
    

class DLS_Image(RIXS_Image):
    def __init__(self, file_path):

        super().__init__()
        self.file_path = file_path


        start_time = time.time()
        print(f"Loading DLS image from {self.file_path}...")
        self.raw_data = self._get_raw_data()
        self.n_images = self.raw_data.shape[0]
        self._get_run_number()
        self._get_normalization_factor()
        self._get_attributes()
        elapsed_time = time.time() - start_time
        print(f"DLS image loaded successfully from {self.file_path}. Time taken: {elapsed_time:.2f} seconds.")

    def _get_run_number(self):
        """
        Extracts the run number from the image file.
        """
        match = re.search(r'i21-(\d+)', self.file_path)
        if match:
            self.run_number = int(match.group(1))
        else:
            raise ValueError(f"Could not extract run number from file path: {self.file_path}")
        return self.run_number

    def _get_raw_data(self):
        """
        Loads raw detector images and extracts relevant experimental parameters from the NeXus file.
        
        This method:
        1. Loads the raw detector images from the NeXus file
        2. Creates copies for processing (imgs_pure) and final results (imgs_processed)
        3. Extracts dark image correction parameters (alpha, beta)
        4. Gets image shift values from previous processing
        5. Retrieves detector counting time
        
        The extracted parameters are stored as instance attributes:
        - self.imgs: Raw detector images
        - self.imgs_pure: Copy of raw images for processing
        - self.imgs_processed: Array for storing processed images
        - self.dark_poly: Dark image correction parameters [alpha, beta]
        - self.shifts: Image shift values
        - self.data_count_time: Detector counting time
        
        Raises
        ------
        Exception
            If required parameters cannot be found in the NeXus file
        """
        print(f"Retrieving images for run {self.run_number}.")
        start_time = time.perf_counter()
        
        with nxload(self.file_path,mode='r') as f:
            try:
                self.raw_data = f.entry['andor']['data'].nxvalue 
            except:
                try:
                    self.raw_data = f.entry1['andor']['data'].nxvalue
                except:
                    raise ValueError("Could not retrieve raw images from NeXus file. Both 'entry' and 'entry1' paths failed.")

            if self.raw_data.ndim == 2:
                self.raw_data = np.expand_dims(self.raw_data, axis=0)
                

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _get_attributes(self):
        """
        Get the attributes from the .nxs file.
        """
        source_nexus = ["entry/", "entry1/"]
        metadata_paths = {
            "th": "instrument/manipulator/th",
            "chi": "instrument/manipulator/chi",
            "phi": "instrument/manipulator/phi",
            "tth": "instrument/spectrometer/armtth",
            'H': "/Q/H",
            'K': "/Q/K",
            'L': "/Q/L",
            "energy": "instrument/pgm/energy",
            "x": "instrument/manipulator/x",
            "y": "instrument/manipulator/y",
            "z": "instrument/manipulator/z",
            "T": "instrument/lakeshore336/sample",
            "polarization": "instrument/id/polarization",
            "count_time": "instrument/m4c1/count_time",
        }
        self.attributes = {}

        # Map the attributes using metadata_name_mapping
        with nxload(self.file_path,mode='r') as f:
            for key, mapped_key in metadata_paths.items():
                if key in self.header:
                    try:
                        self.attributes[key] = f[source_nexus[0]+mapped_key][:]
                    except (KeyError, AttributeError):
                        try: 
                            self.attributes[key] = f[source_nexus[1]+mapped_key][:]
                        except:
                            print(f"Warning: Could not retrieve {mapped_key} from NeXus file")

            #Retrieve the counting time
            try: 
                self.data_count_time = f.entry['instrument']['andor']['count_time']
            except:
                try:
                    self.data_count_time = f.entry1['instrument']['andor']['count_time']
                except:
                    print("WARNING: Could not retrieve count time from NeXus file")
                    self.data_count_time = 120
        
        return self.attributes

    def _get_normalization_factor(self):
        with nxload(self.file_path,mode='r') as f:  
            try:
                self.normalization_factor = f.entry['instrument']['m4c1']['m4c1'].nxvalue
            except:
                try:
                    self.normalization_factor = f.entry1['instrument']['m4c1']['m4c1'].nxvalue
                except:
                    print("WARNING: Could not retrieve normalization factor from NeXus file")
                    self.normalization_factor = 1
        if isinstance(self.normalization_factor, float):
            self.normalization_factor = [self.normalization_factor]*self.n_images

        return self.normalization_factor

    def _remove_bkg_and_filter(self, **kwargs):
        """
        Removes background and filters from spikes the raw image data using specified parameters.
        Background is fitted to filtered raw images with a linear model (translated and scaled).
        Parameters
        ----------
        **kwargs : dict
            Dictionary containing parameters for background removal and filtering.
            Expected keys include:
            - file_path_dark : str or list of str
                Path(s) to the dark image file(s)
            - dark_from_processed_file : bool
                If True, use dark image from processed file
            - hdf5_path_to_dark : str or None
                Path within HDF5 file to the dark image
            - dark_median_filter_kernel_size : list of int
                Kernel size for median filtering the dark image
            - dark_smoothing_parameters : list of int
                Parameters for smoothing the dark image
            - filtertype : str
                Type of filter to use ('gaussian', 'median', etc.)
            - mean_before_spike_removal_dark : bool
                If True, compute mean before spike removal in dark image
            - index_start_fit_bkg : int
                Index to start fitting the background
            - curve_a : float
                Linear coefficient for curvature correction
            - curve_b : float
                Quadratic coefficient for curvature correction
        Returns
        -------
        None
        """
        dark_img_raw, _ = self._get_dark_image(file_path_dark=kwargs.get('file_path_dark'),
                             dark_from_processed_file=kwargs.get('dark_from_processed_file', False),
                             hdf5_path_to_dark=kwargs.get('hdf5_path_to_dark', None),
                             dark_median_filter_kernel_size=kwargs.get('dark_median_filter_kernel_size', [5,15]),
                             dark_smoothing_parameters=kwargs.get('dark_smoothing_parameters', [3,15]),
                             filtertype=kwargs.get('filtertype', 'gaussian'),
                             mean_before_spike_removal_dark=kwargs.get('mean_before_spike_removal_dark', True),
                             index_start_fit_bkg=kwargs.get('index_start_fit_bkg', 1192))
        
        self._filter_and_smooth_dark_image(dark_img_raw,
                                            kernel_size = kwargs.get('data_median_filter_kernel_size', [5,15]),
                                            filter_parameter = kwargs.get('dark_smoothing_parameters', [3,15]),
                                            filtertype=kwargs.get('filtertype', 'gaussian'),
                                            mean_before_spike_removal_dark=kwargs.get('mean_before_spike_removal_dark', True))


        self._filter_img(kwargs.get('img_median_filter_kernel', [5,3,5]), 
                         kwargs.get('spikes_threshold', 1.4))

        self._fit_bkg_sklearn(index_start_fit_bkg=kwargs.get('index_start_fit_bkg', 1192))
        self._subtract_dark()
        self._correct_curvature(curve_a=kwargs.get('curve_a', 0), curve_b=kwargs.get('curve_b', 0))

        return self.imgs_processed, self.normalization_factor


    def _get_dark_image(self,
                        file_path_dark,
                        dark_from_processed_file=False,
                        hdf5_path_to_dark=None,
                        dark_median_filter_kernel_size=[5,15], 
                        dark_smoothing_parameters=[3,15], filtertype='gaussian', 
                        mean_before_spike_removal_dark=True,
                        index_start_fit_bkg=1192):
        
        file_path_dark = file_path_dark if isinstance(file_path_dark, list) else [file_path_dark]
        if dark_from_processed_file:
            self.dark_img = self._get_processed_dark_img(file_path_dark,
                                                        hdf5_path_to_dark)
        else:
            dark_img_raw, _ = self._get_dark_img_from_nxs(file_path_dark) #loading the dark images
            self._filter_and_smooth_dark_image(dark_img_raw,
                                                kernel_size = dark_median_filter_kernel_size, 
                                                filter_parameter = dark_smoothing_parameters, 
                                                filtertype=filtertype,
                                                mean_before_spike_removal_dark=mean_before_spike_removal_dark)

        #fit the dark image to the raw_data
        self.fit_bkg_sklearn(index_start_fit_bkg=index_start_fit_bkg)
        self.subtract_dark()

        print("Dark image processed successfully.")
        

    @staticmethod
    def _get_processed_dark_img(dark_hdf5_filename,
                               path_to_dark=None):
        """
        Gets an already processed dark image (with Leo code) from the hdf file.
        The hdf file should contain the dark image inside the specified path.
        Default path is ["dark_no_spikes_filtered"].
        """

        print(f"Using dark image from hdf file {dark_hdf5_filename}. \n No processing done. \n")
        #get dark from a hdf file
        with h5py.File(dark_hdf5_filename, "r") as f:
            if path_to_dark is None:
                dark_img = f["dark_no_spikes_filtered"][:, :]
            else:
                dark_img = f[path_to_dark][:, :]

        return np.array(dark_img)

    @staticmethod
    def _get_dark_img_from_nxs(file_path_dark): #backround
        """
        Loads and processes dark images from NeXus files.

        This method:
        1. Loads dark images from specified run numbers (self.dark_img_run)
        2. Converts images to float type
        3. Averages multiple dark images if present
        4. Accumulates total counting time

        The processed data is stored in:
        - self.dark_img: Averaged dark image array
        - self.count_time: Total counting time for dark images

        Raises
        ------
        Exception
            If dark image files cannot be found or loaded
        """
        print(f"Retrieving dark images.")
        start_time = time.perf_counter()

        dark_img = np.empty((0,2048,2048)) #initializing the array
        dark_count_time = 0
        for dark_path in file_path_dark:
            print(f"Retrieving dark image from file {dark_path}")
            with nxload(dark_path,mode='r') as f: #loading background
                try:
                    dark_img_raw = np.concatenate((dark_img, f.entry['andor']['data'].nxvalue))
                except:
                    dark_img_raw = np.concatenate((dark_img, f.entry1['andor']['data'].nxvalue))

                try:
                    dark_count_time += f.entry['instrument']['m4c1']['count_time'].nxvalue #count time
                except:
                    try:
                        dark_count_time += f.entry1['instrument']['m4c1']['count_time'].nxvalue
                    except:
                        dark_count_time += 180

        dark_img_raw = dark_img_raw.astype(float)
        #self.dark_img = self.dark_img.mean(axis=0)

        print(f"Total number of dark images retrieved: {dark_img.shape[0]}")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

        return dark_img_raw, dark_count_time

    def _filter_and_smooth_dark_image(self, 
                                      dark_img_raw, 
                                      kernel_size=[5,15], 
                                      filter_parameter=[3,15], filtertype='gaussian', 
                                      mean_before_spike_removal_dark=True):
        """
        Filters spikes from the dark image and applies smoothing filters.

        Parameters
        ----------
        kernel_size : list of int
            Dimensions for median filter kernel. If 2D [vertical_pixels, horizontal_pixels],
            will be converted to 3D [1, vertical_pixels, horizontal_pixels]
        filter_parameter : list of int
            Parameters for additional filtering:
            - For gaussian filter: sigma values [vertical, horizontal]
            - For FFT filter: cutoff frequencies [vertical, horizontal]
            - For butterworth filter: cutoff frequencies [vertical, horizontal]
        filtertype : str, default='gaussian'
            Type of filter to apply after median filtering
            - 'gaussian': Gaussian smoothing with sigma values
            - 'fft': Fourier transform based filter with cutoff frequencies
            - 'butterworth': Butterworth filter with cutoff frequencie
        mean_before_smoothing : bool, default=True
            If True, averages dark images before spike removal
            If False, removes spikes from each image independently, then averages
        """
        start_time = time.perf_counter()
        # Initialize arrays for normalization parameters
        aa = np.zeros(dark_img_raw.shape[0])
        bb = np.zeros(dark_img_raw.shape[0])
        dark_norm = np.zeros_like(dark_img_raw)

        # Ensure kernel_size is 3D
        if len(kernel_size) == 2:
            kernel_size = [1] + kernel_size

        if mean_before_spike_removal_dark:
            # Average dark images first, then remove spikes
            print("Averaging dark images before spike removal.")
            dark_img_raw = dark_img_raw.mean(axis=0)
            if kernel_size[1]+kernel_size[2] > 1.0:
                print(f"Removing spikes from dark image with median filter: kernel_parameter = {kernel_size[1]}x{kernel_size[2]}.")
                aa = dark_img_raw[1600:1700,:].mean()
                bb = dark_img_raw[100:200,:].mean()-dark_img_raw[1600:1700,:].mean()
                dark_norm = (dark_img_raw-aa)/bb
                dark_med = medfilt2d(dark_norm, kernel_size[1:]) #applying the median filter
                spikes_dark = dark_norm - dark_med #getting the spikes
                dark_filt = np.where(spikes_dark>0.4, dark_med, dark_norm) #filtering the spikes
                dark_filt = dark_filt * bb + aa
            else:
                print("No spike removal for dark image.")
                dark_filt = dark_img_raw.copy()
        else:
            # Remove spikes from each image independently, then average
            print("Removing spikes from each dark image independently.")
            if sum(kernel_size) > 1.0:
                print(f"Removing spikes from dark images with median filter: kernel_parameter = {kernel_size[0]}x{kernel_size[1]}x{kernel_size[2]}.")
                dark_filt = np.zeros_like(dark_img_raw)

                for i in range(dark_img_raw.shape[0]):
                    # Normalize each image independently
                    aa[i] = dark_img_raw[i,1600:1700,:].mean()
                    bb[i] = dark_img_raw[i,100:200,:].mean()-dark_img_raw[i,1600:1700,:].mean()
                    dark_norm[i] = (dark_img_raw[i]-aa[i])/bb[i]

                # Apply 3D median filter to the normalized image
                dark_med = median_filter(dark_norm, size=kernel_size) 
                spikes_dark = dark_norm - dark_med
                dark_filt_temp = np.where(spikes_dark>0.4, dark_med, dark_norm)
                    
                # Rescale back
                for i in range(dark_filt.shape[0]):
                    dark_filt[i] = dark_filt_temp[i] * bb[i] + aa[i]
            else:
                print("No spike removal for dark image.")
                dark_filt = dark_img_raw.copy()
            
            # Take mean after spike removal
            dark_filt = dark_filt.mean(axis=0)

        # Apply smoothing
        if filtertype == 'fft':
            dark_filt = self._apply_fft_filter_and_plot(dark_filt, cutoff_frequency = filter_parameter)
        elif filtertype=='gaussian':
            if filter_parameter[0]+filter_parameter[1]>1.5:
                print(f"Filtering dark image with {filtertype} filter.")
                dark_filt = gaussian_filter(dark_filt, filter_parameter)
            else:
                print("No filtering of image.")
                dark_filt = dark_filt.copy()          
        elif filtertype=='butterworth': 
            dark_filt = self._apply_butterworth_filter_and_plot(dark_filt, cutoff_frequency = filter_parameter, order=2)

        self.dark_img = np.squeeze(dark_filt.mean(axis=0)) if self.dark_img.ndim==3 else dark_filt

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")


    @staticmethod
    def _apply_fft_filter_and_plot(image, cutoff_frequency=[70,70], plot_flag=True):
        """
        Apply a 2D FFT-based low-pass filter to remove high-frequency components from an image.

        Parameters
        ----------
        image : ndarray
            2D input image to be filtered
        cutoff_frequency : list of float, default=[70,70]
            Cutoff frequencies [vertical, horizontal] that define the elliptical mask
            in frequency space. Higher values allow more high frequencies to pass.
        plot_flag : bool, default=True
            If True, displays plots of the original image, FFT magnitude spectrum
            with cutoff boundary, and filtered result.

        Returns
        -------
        ndarray
            The filtered image after applying the low-pass filter
        """
        # Compute the 2D FFT of the image and shift the zero frequency component to the center
        f_transform = fft2(image)
        f_transform = fftshift(f_transform)
        
        # Compute the magnitude spectrum of the FFT
        magnitude_spectrum = np.log(np.abs(f_transform) + 1)
        
        # Get the dimensions of the image
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Create the cutoff mask
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt(((i - crow)/cutoff_frequency[0])**2 + ((j - ccol)/cutoff_frequency[1])**2)
                if distance <= 1:
                    mask[i, j] = 1.0

        # Apply the mask to the FFT of the image
        f_transform_filtered = f_transform * mask
        
        # Transform back to the spatial domain
        f_transform_filtered = ifftshift(f_transform_filtered)
        image_filtered = np.abs(ifft2(f_transform_filtered))

        if plot_flag:
            # # Plot the original image, the FFT magnitude spectrum with cutoff, and the filtered image
            # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # # Original Image
            # im0 = axs[0].imshow(image, cmap='gray', vmin=300, vmax=350)
            # axs[0].set_title("Original Image")
            # axs[0].axis('off')
            
            # # FFT Magnitude Spectrum with Cutoff
            # im1 = axs[1].imshow(magnitude_spectrum, cmap='gray')
            # axs[1].set_title("FFT with Cutoff Frequency")
            # axs[1].set_xlabel("Frequency X")
            # axs[1].set_ylabel("Frequency Y")
            
            # # Draw the cutoff circle
            # theta = np.linspace(0, 2 * np.pi, 100)
            # x = cutoff_frequency[1] * np.cos(theta) + ccol
            # y = cutoff_frequency[0] * np.sin(theta) + crow
            # axs[1].plot(x, y, color='red', linestyle='--', linewidth=2)
            
            # # Filtered Image
            # im2 = axs[2].imshow(image_filtered, cmap='gray')
            # axs[2].set_title("Filtered Image")
            # axs[2].axis('off')

            # # Add colorbar to the plots
            # fig.colorbar(im0, ax=axs[0], orientation='vertical')
            # fig.colorbar(im1, ax=axs[1], orientation='vertical')
            # fig.colorbar(im2, ax=axs[2], orientation='vertical')

            plt.figure()
            plt.plot(magnitude_spectrum[:,ccol])

            plt.show()
            
        return image_filtered

    @staticmethod
    def _apply_butterworth_filter_and_plot(image, cutoff_frequency=50, order=2, plot_flag=True):
        """
        Apply an FFT-based low-pass filter to the image and plot the FFT with cutoff.

        params:
        image: 2D numpy array, the input image.
        cutoff_frequency: float, the cutoff frequency for the low-pass filter.
        """
        # Compute the 2D FFT of the image and shift the zero frequency component to the center
        f_transform = fft2(image)
        f_transform = fftshift(f_transform)
        
        # Compute the magnitude spectrum of the FFT
        magnitude_spectrum = np.log(np.abs(f_transform) + 1)
        
        # Get the dimensions of the image
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Create a grid of frequencies
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(u, v, sparse=False, indexing='ij')

        # Modify the distance calculation to account for different cutoff frequencies
        D = np.sqrt((U / cutoff_frequency[0])**2 + (V / cutoff_frequency[1])**2)

        # Create the Butterworth filter
        mask = 1 / (1 + D**(2 * order))

        # Apply the mask to the FFT of the image
        f_transform_filtered = f_transform * mask
        
        # Transform back to the spatial domain
        f_transform_filtered = ifftshift(f_transform_filtered)
        image_filtered = np.abs(ifft2(f_transform_filtered))

        if plot_flag:
            # Plot the original image, the FFT magnitude spectrum with cutoff, and the filtered image
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original Image
            im0 = axs[0].imshow(image, cmap='gray', vmin=300, vmax=350)
            axs[0].set_title("Original Image")
            axs[0].axis('off')
            
            # FFT Magnitude Spectrum with Cutoff
            im1 = axs[1].imshow(magnitude_spectrum, cmap='gray')
            axs[1].set_title("FFT with Cutoff Frequency")
            axs[1].set_xlabel("Frequency X")
            axs[1].set_ylabel("Frequency Y")
            
            # Draw the cutoff circle
            theta = np.linspace(0, 2 * np.pi, 100)
            x = cutoff_frequency[1] * np.cos(theta) + ccol
            y = cutoff_frequency[0] * np.sin(theta) + crow
            axs[1].plot(x, y, color='red', linestyle='--', linewidth=2)
            
            # Filtered Image
            im2 = axs[2].imshow(image_filtered, cmap='gray')
            axs[2].set_title("Filtered Image")
            axs[2].axis('off')

            # Add colorbar to the plots
            fig.colorbar(im0, ax=axs[0], orientation='vertical')
            fig.colorbar(im1, ax=axs[1], orientation='vertical')
            fig.colorbar(im2, ax=axs[2], orientation='vertical')
            
            plt.show()

        return image_filtered

    def _fit_bkg_sklearn(self,
                        index_start_fit_bkg=1192):
        """
        Fits a linear background model to the detector images using scikit-learn.
        
        This method:
        1. Takes a portion of the dark image and detector images above a threshold index
        2. Fits a linear model (y = ax + b) to each detector image using the dark image as predictor
        3. Stores the fitted parameters (a,b) in self.dark_poly for later background subtraction
        
        The background fitting is done using scikit-learn's LinearRegression to find the optimal
        scaling (a) and offset (b) parameters that relate the dark image to each detector image.
        
        The fit is performed on a subset of the image data starting from index_start_fit_bkg
        to avoid including the signal region in the background fit.
        """

        print(f"Fitting background to spectrum.")
        start_time = time.perf_counter()

        chunk_bkg = self.dark_img[index_start_fit_bkg:-100, :].reshape(-1,1)

        model = LinearRegression(copy_X=True)

        for num_image in range(0,self.imgs_pure.shape[0]):
            # Flatten the matrices
            img_flat = self.raw_data[num_image,index_start_fit_bkg:-100,:].reshape(-1,1)

            # Fit the model
            model.fit(chunk_bkg, img_flat)

            # Extract the parameters a and b
            self.dark_poly[0,num_image] = model.coef_[0]
            self.dark_poly[1,num_image] = model.intercept_

        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _fit_bkg(self):

        print(f"Fitting background to spectrum.")
        start_time = time.perf_counter()

        #fit the background
        index_start_fit = self.index_start_fit_bkg 

        # Extract the last 400 rows of K
        chunk_bkg = self.dark_img_filtered[index_start_fit:-100, :]

        for num_image in range(0,self.imgs_pure.shape[0]):
            # calculate chunk of the image
            chunk_img = self.imgs[num_image,index_start_fit:-100,:]

            # Define the objective function
            def objective(params):
                a, b = params
                difference = a * chunk_bkg + b - chunk_img

                return np.sum(difference**2)
            
            # Initial guess for the parameters
            initial_guess = [1, 0]

            # Perform the optimization
            result = minimize(objective, initial_guess)

            # Get the optimal values of a and b
            a_opt, b_opt = result.x
            self.dark_poly[0,num_image] = a_opt
            self.dark_poly[1,num_image] = b_opt

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _filter_img(self, img_median_filter_kernel=[5,3,5], spikes_threshold=1.4):
        """
        Removes spikes from detector images using a 3D median filter.

        The function applies a median filter to detect and remove anomalous high-intensity pixels (spikes)
        from the detector images. It compares the original images with the median-filtered version to
        identify spikes above a threshold, then replaces those pixels with the median-filtered values.

        Parameters
        ----------
        img_median_filter_kernel : list of int, default=[5,3,5]
            Dimensions of the 3D median filter kernel:
            [frames, vertical_pixels, horizontal_pixels]
            The frames dimension is capped at min(5, number of images)
        
        spikes_threshold : float, default=1.4 
            Threshold for spike detection in counts/second.
            Pixels with (original - median)/count_time > threshold are classified as spikes.
        """
        start_time = time.perf_counter()

        if self.data_count_time is None:
            raise Exception("Provide data counting time first.")
        
        if img_median_filter_kernel[0]+img_median_filter_kernel[1]+img_median_filter_kernel[2]>1.0:
            print(f"Removing spikes from images: kernel size {min(img_median_filter_kernel[0], self.raw_data.shape[0])}x{img_median_filter_kernel[1]}x{img_median_filter_kernel[2]}, spikes_threshold={spikes_threshold}.")

            print("Using median_filter of scipy.ndimage.")
            img_corr_med = median_filter(self.raw_data, size=(min(self.raw_data.shape[0], 5), 
                                                               img_median_filter_kernel[1], img_median_filter_kernel[2])) #applying the median filter
            spikes = (self.raw_data - img_corr_med) / self.data_count_time #getting the spikes
            self.imgs_processed = np.where(spikes>spikes_threshold, img_corr_med ,self.raw_data) #filtering the spikes

            print("Found these spikes per image: ", end="")
            for spike_2d in spikes:
                count = np.sum(spike_2d>spikes_threshold)
                print(f"{count}, ", end="")
            print("\n")
        else:
            print(f"No spike removal.")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _subtract_dark(self):
        """
        Subtracts the scaled and offset dark image from each raw image.

        The dark image subtraction follows the formula:
            imgs_processed = imgs - (alpha * dark_img + beta)

        where:
        - alpha (dark_poly[0]): Scaling factor for the dark image
        - beta (dark_poly[1]): Offset/background term
        - dark_img: Filtered dark image
        - imgs: Raw detector images
        - imgs_processed: Background-subtracted images

        The alpha and beta parameters are determined separately for each image
        in the sequence using the fit_bkg() method.

        The subtraction is performed in-place, modifying the imgs_processed attribute.
        """
        
        print(f"Subtracting background from images.")
        start_time = time.perf_counter()

        for num_image in range(0,self.imgs_processed.shape[0]):
            self.imgs_processed[num_image] = self.imgs_processed[num_image] - (self.dark_poly[0,num_image]*self.dark_img + self.dark_poly[1,num_image])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _correct_curvature(self,
                           curve_a,
                           curve_b=0):
        """
        Corrects the curvature of detector images using a linear slope correction.
        The curvature correction is performed by remapping each pixel position using:
            y_new = y - curve_a * x - curve_b * x**2
        where:
        - x, y are the original pixel coordinates
        - curve_a is the curvature parameter (self.slope)
        - y_new is the corrected y-coordinate

        The correction is applied to each image in imgs_pure using 2D histogram
        binning to remap the intensity values to the corrected coordinates.

        Parameters
        ----------
        curve_a : float
            Linear curvature correction parameter (slope)
        curve_b : float
            Quadratic curvature correction parameter

        Raises
        ------
        Exception
            If self.slope is not defined
        """

        print(f"Correcting curvature of image.")
        start_time = time.perf_counter()
        
        for num_image in range(0,self.imgs_processed.shape[0]):

            xdim,ydim=self.imgs_processed[num_image,:,:].shape
            x=np.arange(xdim+1)
            y=np.arange(ydim+1)
            xx,yy=np.meshgrid(x[:-1]+0.5,y[:-1]+0.5)
            #xxn=xx-curv[0]*yy-curv[1]*yy**2
            xxn=xx-curve_a*yy-curve_b*yy**2 #correcting the curvature
            #yyn=yy-curv*xx

            self.imgs_processed[num_image,:,:] = np.histogramdd((xxn.flatten(),yy.flatten()),bins=[y,x],weights=self.imgs_processed[num_image,:,:].T.flatten())[0]

            #im_corr = np.histogramdd((xx.flatten(),yyn.flatten()),bins=[y,x],weights=im.T.flatten())[0]#.T
            #print(ret.shape)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")



