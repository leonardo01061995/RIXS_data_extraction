import os
import time
import h5py
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from cmcrameri import cm
from abc import ABC, abstractmethod
import fabio

class RIXS_Image(ABC):

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
    def _get_energy(self):
        pass

    @abstractmethod
    def _get_attributes(self):
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
                mean_factors_dark = np.ones(self.raw_data.shape[0])

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
                mean_factors_dark = np.ones(self.raw_data.shape[0])

            self.raw_data = cropped_data

            for raw, norm_factor_dark in zip(self.raw_data, mean_factors_dark):
                res_now, _ = self._centroid(
                    img=raw - dark_img_cropped + norm_factor_dark,
                    energy=self.energy,
                    bkg_mean=bkg,
                    factor_ADC=factor_ADC,
                    avoid_double=False,
                    curve_a=curve_a, 
                    curve_b=curve_b,
                )
                self.res = res_now if self.res is None else self.res + res_now              

        #bin into a grid
        self.img_spc, _ = self._bin(self.res, 
                            image_size_h=self.raw_data.shape[-1], 
                            image_size_v=self.raw_data.shape[-2], 
                            subdivide_bins_factor_x=subdivide_bins_factor_x, 
                            subdivide_bins_factor_y=subdivide_bins_factor_y,
                            vertical_shift = vertical_shift)

        #normalize the image
        self.img_spc_normalized = self.img_spc / self.normalization_factor

        # Plot the raw image if requested
        if plot_raw_image:
            self.plot_raw_image()

        return self.img_spc, self.img_spc_normalized
    

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
        factor_ADC=0.56,
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

        return res, double
    
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

        self.file_path = file_path
        
        self.img_spc = None
        self.img_spc_normalized = None
        self.histogram = None
        self.res = None

        self.raw_data = self._get_raw_data()
        self._get_run_number()
        self._get_image_number()
        self._get_absolute_image_number()
        self._get_normalization_factor()
        self.energy = self._get_energy()

    def _get_raw_data(self):
        self.raw_data = np.flipud(fabio.open(self.file_path).data)
        self.header = fabio.open(self.file_path).header

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

        counter_names = self.header["counter_mne"]
        counter_values = self.header["counter_pos"]
        # Step 1: Split the strings into lists
        names_list = counter_names.split()
        values_list = [float(x) for x in counter_values.split()]  # Convert to float

        # Step 2: Find the position of "mir", the normalization value
        target_name = "mir"
        position = names_list.index(target_name)

        # Step 3: Retrieve the corresponding value, divided by 1E6 because it's around 1E7
        mir_value = values_list[position] / 1E6

        return mir_value
    
    def _get_energy(self):
        """
        Get the energy from the header
        """
        counter_names = self.header["motor_mne"]
        counter_values = self.header["motor_pos"]
        # Step 1: Split the strings into lists
        names_list = counter_names.split()
        values_list = [float(x) for x in counter_values.split()]  # Convert to float

        # Step 2: Find the position of "energy"
        target_name = "energy"
        position = names_list.index(target_name)

        # Step 3: Retrieve the corresponding value
        self.energy = values_list[position]

        return self.energy
    
    def single_photon_counting(self, 
            curve_a, curve_b=0,
            roi_x=(0,2048),roi_y=(0,2048),
            roi_x_for_dark=(1600,1800), roi_y_for_dark=(250,1800),
            subdivide_bins_factor_x=1, subdivide_bins_factor_y=2.7,
            factor_ADC=1.2,
            vertical_shift = 0,
            subtract_background_from_img=False,
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

class TPS_Image(RIXS_Image):
    def __init__(self, file_path, file_path_background = None):
        self.file_path = file_path
        self.file_path_background = file_path_background if file_path_background is not None else None

        self.img_spc = None
        self.img_spc_normalized = None
        self.histogram = None
        self.res = None

        self._get_raw_data()
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
        
    

class RIXS_Run:
    def __init__(
            self, 
            facility,
            folder, 
            run_number,
            identifier_string,
            exp_number):
        """
        Initialize esrf_run object
        
        Parameters
        ----------
        folder : str 
            Folder containing the data files
        facility : str
            Facility name: ESRF or TPS
        run_number : int
            Run number identifier
        unique_identifier : str
            Unique identifier string for the run:
        exp_number: str
            Code identfying the experiment (e.g. hc3000)
        """
        self.folder = folder
        self.facility = facility
        if facility not in ["ESRF", "TPS"]:
            raise ValueError("Facility must be either 'ESRF' or 'TPS'")
        self.identifier_string = identifier_string
        self.run_number = sorted(run_number) if isinstance(run_number, list) else run_number
        self.exp_number = exp_number

        #variables we need to store
        self.imgs_lc = None
        self.imgs_spc = None
        self.shifts = None
        self.real_shifts = None
        self.file_names = None
        self.pixel_row_start = None
        self.pixel_row_stop = None
        self.normalization_factors = []

        #first find the images
        start_time = time.perf_counter()
        if self.facility == "ESRF":
            self._search_images_ESRF()
        elif self.facility == "TPS":
            self._search_images_TPS()
        print(f"Elapsed time: {time.perf_counter() - start_time:.2f} seconds. \n")

    def _search_images_TPS(self):
        """
        Search for valid image numbers
        
        Returns
        -------
        list
            List of valid image numbers
        """

        # If run_number is a list, iterate over each entry
        run_numbers_to_check = sorted(self.run_number) if isinstance(self.run_number, list) else [self.run_number]

        # Get all the files in the folder that contain the run number
        files = [
            f for f in os.listdir(self.folder)
            if f.endswith(".hdf5") and 
            f.split("_")[0] == 'rixs'and int(f.split("_")[2]) in run_numbers_to_check and
            f.split("_")[1] == self.identifier_string and os.path.splitext(f)[0].split("_")[3].isdigit() and
            not f.endswith("000.hdf5")
        ]
        if not files:
            raise FileNotFoundError("No files found for the specified run number(s).")
        self.file_names = [os.path.join(self.folder, f) for f in files]

        print("Files to be used:")
        for f in files:
            print(f)
            

        return files

    def _search_images_ESRF(self):
        """
        Search for valid image numbers
        
        Returns
        -------
        list
            List of valid image numbers
        """
        # Initialize lists to store results
        file_names = []
        image_numbers = []
        absolute_image_numbers = []

        # If run_number is a list, iterate over each entry
        run_numbers_to_check = sorted(self.run_number) if isinstance(self.run_number, list) else [self.run_number]

        # Get all EDF files in directory
        # files = [f for f in os.listdir(self.folder) if f.lower().endswith('.edf')]
            # Use os.scandir for better performance
            #using a generator, not a list, more memory efficient
        files = sorted(
            (entry for entry in os.scandir(self.folder) if entry.name.lower().endswith('.edf')),
            key=lambda f: f.name  # Sort by file name (assuming natural order by numbers)
        )

        initial_index = 0
        for run_number_now in run_numbers_to_check:

            file_names_now = []
            image_numbers_now = []
            absolute_image_numbers_now = []

            print(f"Searching images for run number: {run_number_now}...")

            # Check each file
            file_found = False
            
            for file in files:
                file_path = os.path.join(self.folder, file)

                if int(file.name[-8:-4]) < self.first_absolute_image_number:
                    # Skip files with absolute image number less than the specified threshold
                    continue

                try:
                    # Open file once and reuse the object
                    img = EDF_Image(file_path)
                    
                    # Check if run number matches
                    if str(img.run_number) == str(run_number_now):  # Convert to string to ensure consistent comparison
                        file_names_now.append(file_path)
                        image_numbers_now.append(img.image_number)
                        absolute_image_numbers_now.append(img.absolute_image_number)
                        file_found = True
                    elif file_found:
                        # Early exit: stop if a non-matching file is found after valid ones
                        break        
                        
                except (fabio.fabioutils.NotGoodReader, IOError) as e:
                    print(f"Error reading {file}: {str(e)}")
                except Exception as e:
                    print(f"Unexpected error with {file}: {str(e)}")
                    continue

            print(f"Found {len(file_names_now)} files for run number {run_number_now} from image {absolute_image_numbers_now[0]} to {absolute_image_numbers_now[-1]}.")
            file_names += file_names_now
            image_numbers += image_numbers_now
            absolute_image_numbers += absolute_image_numbers_now

        if not file_names:
            print(f"Warning: No valid EDF files found for run number {self.run_number} \n\n")

        else:
            sorted_indices = sorted(range(len(absolute_image_numbers)), key=lambda i: absolute_image_numbers[i])
            self.file_names = [file_names[i] for i in sorted_indices]
            self.image_numbers = [image_numbers[i] for i in sorted_indices]
            self.absolute_image_numbers = [absolute_image_numbers[i] for i in sorted_indices]
            print(f"Found {len(file_names)} files for run number {self.run_number} from image {self.absolute_image_numbers[0]} to {self.absolute_image_numbers[-1]}.")

        return file_names, image_numbers
    
    @staticmethod
    def _find_curvature(im,frangex,frangey, plotting=False, deg=2):
        """
        Analyze the curvature of the given image data within specified x and y ranges.

        Parameters
        ----------
        im : numpy.ndarray
            A 2D array representing the image data to be analyzed.
        frangex : list or tuple
            A range of x-coordinates (start, end) to consider for the analysis.
        frangey : list or tuple
            A range of y-coordinates (start, end) to consider for the analysis.
        plotting : bool, optional
            If True, generates plots for visual inspection of the reference and cross-correlation results. Default is False.
        deg : int, optional
            The degree of the polynomial to fit to the shifts. Default is 2 (quadratic fit).

        Returns
        -------
        None
            The method prints the extracted curvature coefficients and stores them in the instance variable `self.curv`.
        """
        ref=np.ones([frangey[1]-frangey[0],im[:,:].shape[0]]) * im[frangey[0]:frangey[1],frangex[0]:frangex[1]].mean(axis=1)[:,np.newaxis]
        crosscorr=fftconvolve(im[frangey[0]-100:frangey[1]+100,:],ref[::-1,:],axes=0)
        if plotting==True:
            f,ax=plt.subplots(2)
            ax[0].plot(ref[:,5])
            ax[0].plot(ref[:,10])
            for i in np.arange(frangex[0],frangex[1],10):
                ax[1].plot(crosscorr[:,i])
        shifts=np.argmax(crosscorr,axis=0)
        curv=np.polyfit(np.arange(im[:,:].shape[0]),shifts,deg=deg)
        print(f"Extracted curvature: f{curv}")

        return curv[1], curv[0]

    def process_images(self, 
                       extract_curvature = False,
                       curve_a=0,
                       curve_b=0,
                       factor_ADC = 1.2,
                       calibration = 0,
                       roi_x=(0,2048),
                       roi_y=(0,2048),
                       subdivide_bins_factor_x=1, 
                       subdivide_bins_factor_y=2.7,
                       subtract_background_from_img=False,
                       subtract_background_from_corner=False,
                       background_pre_processed_path=None,
                       roi_x_for_dark=(1600,1800),
                       roi_y_for_dark=(250,1800),
                       pixel_row_start=None, pixel_row_stop=None,
                       find_aligning_range=False, 
                       fit_shifts=False, 
                       correlation_batch_size=1,
                       poly_order=1,
                       plot_raw_image=False):
        """
        Process multiple EDF image files using single photon counting
        
        Parameters
        ----------
        subdivide_bins_factor_x : float, optional
            Factor by which to subdivide each pixel along x-axis for sub-pixel resolution, default 1.0
        subdivide_bins_factor_y : float, optional
            Factor by which to subdivide each pixel along y-axis for sub-pixel resolution, default 1.0
        curve_a : float
            Coefficient that represents the linear (first-order) coefficient of the slope
        curve_b : float
            Coefficient that represents the quadratic (2nd-order) coefficient of the slope
        factor_ADC : float, optional
            Factor to convert ADU to electrons, default 0.6 (TPS uses electron-multiplied CCD)
        calibration : float, optional
            Calibration factor for the energy axis in meV/pixel, default 2.1
        roi_x : tuple of int, optional
            Region of interest along the x-axis, default (0, 2048)
        roi_y : tuple of int, optional
            Region of interest along the y-axis, default (0, 2048)
        extract_curvature : bool, optional
            If True, extract curvature from the image, default False
        subtract_background_from_img : bool, optional
            If True, subtract background from a dark image, default False
        background_pre_processed : str, optional
            Path to the pre-processed background image, default None
        roi_x_for_dark : tuple of int, optional
            Region of interest along the x-axis for dark image subtraction, default (1600, 1800)
        roi_y_for_dark : tuple of int, optional
            Region of interest along the y-axis for dark image subtraction, default (250, 1800)
        pixel_row_start : int
            Starting pixel row for the correlation region (typically near elastic line)
        pixel_row_stop : int 
            Ending pixel row for the correlation region (typically near elastic line)
            
        Returns
        -------
        numpy.ndarray
            3D array containing processed images stacked along first axis
        """

        self.extract_curvature = extract_curvature
        self.curve_a = -curve_a
        self.curve_b = -curve_b
        self.factor_ADC = factor_ADC
        self.calibration = calibration
        self.subdivide_bins_factor_x = subdivide_bins_factor_x
        self.subdivide_bins_factor_y = subdivide_bins_factor_y
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.subtract_background_from_img = subtract_background_from_img
        self.subtract_background_from_corner = subtract_background_from_corner
        self.background_pre_processed_path = background_pre_processed_path
        self.roi_x_for_dark = roi_x_for_dark
        self.roi_y_for_dark = roi_y_for_dark

        # Check if curvature extraction is disabled and no curve parameters are provided
        if not self.extract_curvature and (self.curve_a == 0 and self.curve_b == 0):
           print("Careful: Curvature extraction is disabled and the given slope is zero")

        if not find_aligning_range and (pixel_row_start is None or pixel_row_stop is None):
            raise ValueError("If find_aligning_range is False, both pixel_row_start and pixel_row_stop must be provided.")

        if not self.file_names:
            return
        
        
        # Check if curvature extraction is enabled
        if self.extract_curvature:
           # Call the find_curvature method with appropriate parameters
           self.curv_a, self.curve_b = self._find_curvature(self.imgs_spc.mean(axis=0), 
                                            frangex=(0, self.imgs_spc.shape[2]), 
                                            frangey=(0, self.imgs_spc.shape[1]))   

        processed_images = []
        imgs_processed = []

        #extract the background from pre-processed image
        if self.subtract_background_from_img:
            self.dark_img = self.load_dark_image(self.background_pre_processed_path, "data")
             
        print("Performing single photon counting and curvature correction...")
        start_time = time.perf_counter()
        for filename in self.file_names:
            # Create edf_image instance and process
            if self.facility == "ESRF":
                img = EDF_Image(filename)
            elif self.facility == "TPS":
                img = TPS_Image(filename)

            processed, _ = img.single_photon_counting(
                curve_a=self.curve_a, 
                curve_b=self.curve_b,
                roi_x=self.roi_x,
                roi_y=self.roi_y,
                subdivide_bins_factor_x=self.subdivide_bins_factor_x, 
                subdivide_bins_factor_y=self.subdivide_bins_factor_y,
                factor_ADC=self.factor_ADC,
                subtract_background_from_img=self.subtract_background_from_img,
                subtract_background_from_corner=self.subtract_background_from_corner,
                dark_img=self.dark_img if self.subtract_background_from_img else None,
                roi_x_for_dark=self.roi_x_for_dark,
                roi_y_for_dark=self.roi_y_for_dark,
                plot_raw_image=plot_raw_image)
            
            processed_images.append(processed)
            imgs_processed.append(img)
            self.normalization_factors.append(img.normalization_factor)
            
        self.imgs_spc = np.stack(processed_images, axis=0)
        self.normalization_factors = np.stack(self.normalization_factors)
        print(f"Elapsed time: {time.perf_counter() - start_time:.2f} seconds. \n")

        mean_normalization_factor = np.mean(self.normalization_factors)
        std_normalization_factor = np.std(self.normalization_factors)
        print(f"Normalization factor: {mean_normalization_factor} +- {std_normalization_factor}")


        #correct the shift
        if find_aligning_range:
            self.pixel_row_start, self.pixel_row_stop = self._find_aligning_range(threshold=0.1)
        else:
            self.pixel_row_start = pixel_row_start
            self.pixel_row_stop = pixel_row_stop
            
        if self.imgs_spc.shape[0] > 1: 
            ###### ONly calculate the shift: then re-bin each image
            self._calculate_shift(self.pixel_row_start, self.pixel_row_stop, fit_shifts, 
                            correlation_batch_size, poly_order)
               
            ###### now re-bin the edf images
            print("Re-binning edf images with known shifts")
            processed_images = []
            for i,img in enumerate(imgs_processed):
                processed_images.append(img.single_photon_counting(self.curve_a, 
                    curve_b = self.curve_b,
                    roi_x = self.roi_x,
                    roi_y = self.roi_y,
                    subdivide_bins_factor_x = self.subdivide_bins_factor_x, 
                    subdivide_bins_factor_y = self.subdivide_bins_factor_y,
                    vertical_shift = self.shifts[i])[1]*mean_normalization_factor)

        else:
            print("Only one file found. No energy correlation necessary.")

        self.imgs_lc = np.stack(processed_images, axis=0)
        
        return self.imgs_lc
    
    def _find_aligning_range(self, threshold=0.1):
        """
        Find the position of the first significant peak in the image spectrum.
        Uses a moving average to smooth the data and identifies where signal rises
        above background.
        
        Parameters
        ----------
        window_size : int, optional
            Size of the moving average window for smoothing, default 5
        threshold : float, optional
            Threshold value above which signal is considered significant,
            as fraction of maximum intensity, default 0.1
            
        Returns
        -------
        tuple
            A tuple containing the start and stop indices of the interval around the significant peak in the image spectrum.
        """

        print("Attempting to find the range around the elastic line...")

        # Get the spectrum (mean along horizontal axis)
        if self.imgs_spc is not None:
            spectrum = self.imgs_spc.mean(axis=0).mean(axis=1)
        else:
            raise ValueError("Image must be processed first using single_photon_counting")
        
        # Find where signal rises above threshold * max intensity
        threshold_value = threshold * np.max(spectrum)
        # Calculate the moving average with a window of 3
        moving_avg = np.convolve(spectrum, np.ones(3)/3, mode='same')
        peak_start = np.where(moving_avg > threshold_value)[0]
        
        if len(peak_start) == 0:
            raise ValueError("No peak found above threshold")
        
        # Use the last peak position (elastic line) instead of first peak
        elastic_line = peak_start[-1]
        range_start = elastic_line - 100
        range_stop = elastic_line + 15
        
        print(f"Using range: {range_start}, {range_stop}")
        
        # Return the range around the elastic line
        return range_start, range_stop
    

    def plot_image_and_spectra(self, plot_image=True, plot_spectra=True, plot_shifts=True):
        """
        Plot the RIXS spectra
        """

        if plot_image:
            # Create figure and axis objects with subplots()
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
            
            # First subplot (top)
            img1 = self.imgs_spc[:,:,:].mean(axis=0)
            vmin1 = img1.mean() - img1.std()
            vmax1 = img1.mean() + img1.std()
            im1 = ax1.imshow(img1, cmap=cm.lipari, vmin=vmin1, vmax=vmax1)
            fig.colorbar(im1, ax=ax1)

            img2 = self.imgs_lc[:,:,:].mean(axis=0)
            vmin2 = img2.mean() - img2.std() 
            vmax2 = img2.mean() + img2.std()
            im2 = ax2.imshow(img2, cmap=cm.lipari, vmin=vmin2, vmax=vmax2)
            fig.colorbar(im2, ax=ax2)

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.show()
        
        if plot_spectra:
            # Second figure, spectra
            imgs_lc_not_normalized = np.array([self.imgs_lc[i, :, :] * self.normalization_factors[i] / self.normalization_factors.mean() for i in range(self.imgs_lc.shape[0])])
            plt.figure(figsize=(10,5))
            for i in range(self.imgs_lc.shape[0]):
                plt.plot(imgs_lc_not_normalized[i,:,:].sum(axis=1) * self.imgs_lc.shape[0])

            plt.plot(imgs_lc_not_normalized[:,:,:].sum(axis=0).sum(axis=1), 'r--', linewidth=2,
                    label='Mean (like RIXStoolbox)')
            plt.plot(self.imgs_lc[:,:,:].mean(axis=0).mean(axis=1) * self.imgs_lc.shape[0] * self.imgs_lc.shape[2],
                    'k-', linewidth=2, label='Mean')
            
            if self.imgs_spc.shape[0]>1:
                plt.axvline(x=self.pixel_row_start, color='k', linestyle='--', label='Start Pixel Row')
                plt.axvline(x=self.pixel_row_stop, color='k', linestyle='--', label='Stop Pixel Row')
            plt.legend()
            plt.show()

        if plot_shifts and self.imgs_spc.shape[0] > 1:
            # Third figure, shifts
            plt.figure(figsize=(10,5))
            plt.plot(self.real_shifts, 'ko-', label='Real Shifts')  # 'ko-' for black circles connected by lines
            plt.plot(self.shifts, 'ro-', label='Used Shifts')
            plt.xlabel('Image Index')
            plt.ylabel('Shift Value')
            plt.title('Real Shifts of Images')
            plt.grid()
            plt.legend()
            plt.show()


    def _calculate_shift(self, pixel_row_start, pixel_row_stop, fit_shifts, correlation_batch_size, poly_order):

        """
        correct_shift2 method

        This method applies the calculated shifts to the images stored in the 
        imgs_spc attribute. It iterates through each image, checking the 
        magnitude of the shift. If the shift exceeds a threshold (0.5 in this 
        case), it performs interpolation to adjust the image based on the 
        calculated shift. If the shift is within the threshold, the original 
        image is directly assigned to the imgs_lc attribute without modification. 
        The method also tracks and prints the elapsed time for the operation.

        Returns
        -------
        None
        """

        print(f"Calculating energy shift of {self.imgs_spc.shape[0]} images.")
        start_time = time.perf_counter()

        # Initialize arrays
        self.shifts = np.zeros(self.imgs_spc.shape[0])
        self.real_shifts = self.shifts.copy()
        
        #pre-average the images to have a more reliable cross-correlation
        #spec_avg has shape: (imgs_spc.shape[1], imgs_spc.shape[0]//batch_size + reminder)
        spec_avg = self.average_images_in_batches(self.imgs_spc, min(correlation_batch_size, self.imgs_spc.shape[0]))
        
        # Get reference spectrum once (avoid recomputing for each iteration)
        ref_spectrum = spec_avg[:,0]

        real_shifts = np.zeros(spec_avg.shape[1])
        # Calculate the shift for the remaining images
        for num_image in range(1, spec_avg.shape[1]):
            # Get current spectrum
            curr_spectrum = spec_avg[:,num_image]
            
            # Calculate shift
            real_shifts[num_image] = self.correlate_spectra(
                curr_spectrum, ref_spectrum, pixel_row_start, pixel_row_stop)

        # index_aux = range(0, self.imgs_spc.shape[0], correlation_batch_size)
        index_aux = np.arange(0, self.imgs_spc.shape[0], correlation_batch_size)[:self.imgs_spc.shape[0] // correlation_batch_size]
        if fit_shifts:
            coeffs = np.polyfit(index_aux, real_shifts, deg=poly_order)
            self.shifts = np.polyval(coeffs, range(0, self.imgs_spc.shape[0]))
        else:
            # Extrapolate by specifying left and right values
            self.shifts = np.interp(range(0, self.imgs_spc.shape[0]), index_aux, real_shifts, left=real_shifts[0], right=real_shifts[-1])
            # Smooth the shifts using a moving average with window size of correlation_batch_size*2
            # window_size = correlation_batch_size * 3
            # kernel = np.ones(window_size) / window_size
            # self.shifts = np.convolve(self.shifts, kernel, mode='same')
            
            sigma = correlation_batch_size   # Adjust sigma based on correlation_batch_size
            self.shifts = gaussian_filter1d(self.shifts, sigma=sigma)

        print("Shifting images by: ", end="")
        for i in range(len(self.shifts)):
            print(f"{self.shifts[i]:.2f}, ", end="")
        print("\n")
            
        #save the real shifts somewhere for plotting
        for i in range(0, real_shifts.shape[0]):
            self.real_shifts[i*correlation_batch_size:(i+1)*correlation_batch_size] = real_shifts[i]
            
        print("")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    @staticmethod
    def load_dark_image(filename, dataset_name):
        """
        Load a dark image from a file.

        Parameters
        ----------
        filename : str
            Path to the file containing the dark image.

        Returns
        -------
        numpy.ndarray
            The loaded dark image.
        """
        # Load the dark image from the specified file
        with h5py.File(filename, 'r') as f:
            dark_image = f[dataset_name][:]
        
        return dark_image

    @staticmethod
    def correlate_spectra(spec, spec_ref, pixel_start, pixel_stop):
        """
        this function correlates two spectra and returns the lag value to be used for shifting the images

        params:
        spec: the spectrum to be shifted
        spec_ref: the reference spectrum
        pixel_start: the starting pixel
        pixel_stop: the stopping pixel

        return:
        lag: the lag value to be used for shifting the images
        """
        # Ensure spec and spec_ref are 1D arrays
        if spec.ndim != 1 or spec_ref.ndim != 1:
            raise ValueError("Both spec and spec_ref must be 1D arrays.")

        factor = 2  # Change this value as needed

        # Ensure pixel_start and pixel_stop are within bounds
        if pixel_start < 0 or pixel_stop >= len(spec):
            raise ValueError("pixel_start and pixel_stop must be within the bounds of the spectrum.")

        if factor == 1:
            # Skip interpolation if factor is 1
            xx = np.arange(pixel_start, pixel_stop + 1)
            spec = spec[xx]  # Use the original spectrum directly
            spec_ref = spec_ref[xx]  # Use the original reference spectrum directly
        else:
            xx = np.arange(pixel_start, pixel_stop + 1 / factor, 1 / factor)
            spec = np.interp(xx, np.arange(0, len(spec)), spec)  # Interpolating the spectrum
            spec_ref = np.interp(xx, np.arange(0, len(spec_ref)), spec_ref)  # Interpolating the reference spectrum

        crosscorr = correlate(spec, spec_ref)
        lag_values = np.arange(-spec.shape[0] + 1, spec.shape[0], 1)
        lag = lag_values[np.argmax(crosscorr)] / factor

        return lag
    
    @staticmethod
    def average_images_in_batches(imgs, batch_size):
        """
        Averages images in batches from a 3D array.

        Parameters
        ----------
        imgs : 3D array[float]
            Array containing the images stacked along the first axis
        max_lc_images : int
            Maximum number of low-count images to consider for averaging

        Returns
        -------
        averaged_imgs : 3D array[float]
            Array containing the 1D RIXS spectra stacked along the second axis
        """
        n_images = imgs.shape[0]
        
        if batch_size == 0:
            raise ValueError("Batch size must be greater than zero. Consider increasing max_lc_images.")
        
        n_batches = n_images // batch_size
        averaged_spectra = np.zeros((imgs.shape[1], n_batches))

        for i in range(n_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            averaged_img = np.mean(imgs[start_index:end_index, :, :], axis=0)
            averaged_spectra[:,i] = averaged_img.mean(axis=1)        

        return averaged_spectra
    
    def _calculate_energy_axis(self, auto_alignment, index_elastic_line=None):
        """
        Calculate the energy axis for the RIXS spectra using the calibration factor.
        The energy loss is calculated based on the pixel positions and the calibration factor.
        If auto_alignment is True, the index of the elastic line is determined automatically.
        Parameters:
        ----------
        auto_alignment : bool
            If True, automatically determine the index of the elastic line.
        index_elastic_line : int, optional
            If provided, use this index for the elastic line instead of calculating it automatically.
        Returns:
        -------
        numpy.ndarray
            The energy loss axis for the RIXS spectra.
        """
        # Get the number of pixels in the vertical direction
        n_pixels = self.imgs_lc.shape[1]
        
        if auto_alignment:
            # Calculate the mean along axis=0 and then axis=1
            mean_spectrum = self.imgs_lc.mean(axis=0).mean(axis=1)
            
            # Smooth the spectrum with a Gaussian filter
            smoothed_spectrum = gaussian_filter1d(mean_spectrum, sigma=3)
            
            # Retrieve the index of the maximum in the pixel range
            index_elastic_line = np.argmax(
                smoothed_spectrum[self.pixel_row_start:self.pixel_row_stop]) + self.pixel_row_start

        # Calculate the energy axis using the calibration factor
        energy_loss = -(np.linspace(0, n_pixels, n_pixels) - index_elastic_line) \
            * self.calibration/1000/self.subdivide_bins_factor_y
        
        energy_loss = energy_loss[::-1]      

        return energy_loss 
    
    def save_pre_processed_dataset(self, path_dir=None, filename_save_auto=True, filename_save = '', auto_alignment=True, index_elastic_line=None):
        if self.facility == "ESRF":
            self.save_pre_processed_dataset_ESRF(path_dir, filename_save_auto, filename_save)
        elif self.facility == "TPS":
            self.save_pre_processed_dataset_TPS(path_dir, filename_save_auto, filename_save, auto_alignment, index_elastic_line)

    def save_pre_processed_dataset_ESRF(self, path_dir, filename_save_auto, filename_save = ''):
        """
        Save the pre-processed dataset to a file
        """
        
        if not filename_save_auto and filename_save == '':
            raise ValueError("Either filename_save_auto should be True or filename_save should be provided.")
        
        if filename_save_auto:
            if isinstance(self.run_number, list):
                run_number_str = "_".join(f"{int(num):04d}" for num in self.run_number)  # Join list elements with "_" and format as 4 digits
            else:
                run_number_str = f"{int(self.run_number):04d}"  # Convert single run number to a 4-digit string

            filename_save = f"{self.identifier_string}_run_{run_number_str}_{np.shape(self.imgs_lc)[0]:03d}_imgs.hdf5"

        full_path = os.path.join(path_dir, filename_save)
        with h5py.File(full_path, 'w') as hf:
            hf.create_dataset('data', data=self.imgs_lc, dtype='float32')
            hf.attrs['exp_number'] = self.exp_number
            hf.attrs['filename_dir'] = self.identifier_string
            hf.attrs['normalization_factor'] = self.normalization_factors.sum()
            hf.attrs['factor_ADC'] = self.factor_ADC
            hf.attrs['calibration'] = self.calibration
            hf.attrs['curve_a'] = self.curve_a
            hf.attrs['curve_b'] = self.curve_b
            hf.attrs['roi_x'] = self.roi_x
            hf.attrs['roi_y'] = self.roi_y
            hf.attrs['subdivide_bins_factor_x'] = self.subdivide_bins_factor_x
            hf.attrs['subdivide_bins_factor_y'] = self.subdivide_bins_factor_y
            # Save only the first element if self.run_number is a list
            if isinstance(self.run_number, list):
                hf.attrs['run_number'] = f"{self.identifier_string}_run#_{int(self.run_number[0]):04d}"  # Save formatted run number
            else:
                hf.attrs['run_number'] = f"{self.identifier_string}_run#_{int(self.run_number):04d}"

        print(f"Pre-processed dataset saved as {filename_save} \n\n")


    def save_pre_processed_dataset_TPS(self, path_dir=None, filename_save_auto=True, filename_save = '',
                                   auto_alignment=True, index_elastic_line=None):
        """
        Save the pre-processed dataset to a file in HDF5 format.
        Parameters:
        -----------
        path_dir : str
            The directory path where the file will be saved.
        filename_save_auto : bool
            If True, automatically generate the filename based on the object's attributes.
        filename_save : str, optional
            The custom filename to save the dataset. If not provided and `filename_save_auto` is False, 
            a ValueError will be raised.
        auto_alignment : bool, optional
            If True, automatically align the spectrum based on the elastic line.
        index_elastic_line : int, optional
            Index of the elastic line in the spectrum. If not provided, it will be calculated.
        Raises:
        -------
        ValueError
            If both `filename_save_auto` is False and `filename_save` is an empty string.
        Notes:
        ------
        - The method saves the dataset (`self.imgs_lc`) and additional metadata as attributes in the HDF5 file.
        - If `self.run_number` is a list, only the first element is saved as part of the metadata.
        - The generated filename (if `filename_save_auto` is True) includes the directory path, run number, 
          and the number of images in the dataset.
        """
        
        if not filename_save_auto and filename_save == '':
            raise ValueError("Either filename_save_auto should be True or filename_save should be provided.")
        
        if path_dir is None:
            path_dir = self.folder
            print("Warning: Path directory is not provided. Using the default folder path.")
        
        if filename_save_auto:
            if isinstance(self.run_number, list):
                filename_save = os.path.join(path_dir, f"rixs_{os.path.basename(self.file_names[0]).split('_')[1]}_{'_'.join([str(rn) for rn in self.run_number])}_processed.hdf5")
            else:
                filename_save = os.path.join(path_dir, f"{os.path.splitext(os.path.basename(self.file_names[0]))[0][:-4]}_processed.hdf5")


        full_path = os.path.join(path_dir, filename_save)
        with h5py.File(full_path, 'w') as hf:
            # Open the first file to copy attributes
            with h5py.File(self.file_names[0], 'r') as original_file:
                for attr_name, attr_value in original_file.attrs.items():
                    hf.attrs[attr_name] = attr_value

            # Modify the 'header' attribute to replace the "Iph" value
            header = hf.attrs['header']
            header_parts = header.split(":")[1].split(",")
            for i, part in enumerate(header_parts):
                if "Iph" in part:
                    key, value = part.strip().split(" ")
                    decimal_places = len(value.split(".")[1]) if "." in value else 0
                    new_value = f"{self.normalization_factors.sum():.{decimal_places}f}"
                    header_parts[i] = f"{key} {new_value}"
            hf.attrs['header'] = header.split(":")[0]+": "+ ",".join(header_parts)

            # Save the sum of self.imgs_lc, each multiplied by the corresponding normalization factor
            summed_data = np.sum(
                [self.imgs_lc[i] * self.normalization_factors[i] for i in range(self.imgs_lc.shape[0])],
                axis=0
            ) / self.normalization_factors.mean()
            rixs_spectrum = summed_data.sum(axis=1)

            # Save the summed data to the new file
            hf.create_dataset('data', data=summed_data)

            # Save the 1D RIXS spectrum
            #calculate the energy axis
            energy_loss = self._calculate_energy_axis(auto_alignment=auto_alignment, 
                                                  index_elastic_line=index_elastic_line)
            hf.create_dataset('rixs_spectrum', data=np.column_stack((energy_loss, rixs_spectrum[::-1])))

            # Save the normalization factor as an attribute
            hf.attrs['normalization_factor'] = self.normalization_factors.sum()
            hf.attrs['factor_ADC'] = self.factor_ADC
            hf.attrs['calibration'] = self.calibration
            hf.attrs['curve_a'] = self.curve_a
            hf.attrs['curve_b'] = self.curve_b
            hf.attrs['roi_x'] = self.roi_x
            hf.attrs['roi_y'] = self.roi_y
            hf.attrs['subdivide_bins_factor_x'] = self.subdivide_bins_factor_x
            hf.attrs['subdivide_bins_factor_y'] = self.subdivide_bins_factor_y

        print(f"Pre-processed hdf5 dataset saved as {filename_save} \n")


    def save_txt_rixs_spectra(self, path_dir=None, filename_save_auto=True, filename_save="",
                              auto_alignment=True, index_elastic_line=None):
        """
        Save the RIXS spectra to a text file with two columns:
        pixel number and intensity.
        
        Parameters
        ----------
        path_dir : str, optional
            Directory path to save the file. If not provided, uses the default folder path.
        filename_save : str
            Path to save the output text file
        filename_save_auto : bool
            If True, automatically generate the filename based on the object's attributes.
        auto_alignment : bool, optional
            If True, automatically align the spectrum based on the elastic line.
        index_elastic_line : int, optional  
            Index of the elastic line in the spectrum. If not provided, it will be calculated.
        """

        if not filename_save_auto and filename_save=="":
            raise ValueError("Either filename_save_auto should be True or filename_save should be provided.")
        
        if path_dir is None:
            path_dir = self.folder
            print("Warning: Path directory is not provided. Using the default folder path.")


        if filename_save_auto:
            if isinstance(self.run_number, list):
                filename_save = os.path.join(path_dir, f"rixs_{os.path.basename(self.file_names[0]).split('_')[1]}_{'_'.join([str(rn) for rn in self.run_number])}_processed_normalized.txt")
            else:
                filename_save = os.path.join(path_dir, f"{os.path.splitext(os.path.basename(self.file_names[0]))[0][:-4]}_processed_normalized.txt")
        
        # Calculate mean spectrum across all horizontal positions
        summed_data = np.sum(
                [self.imgs_lc[i] * self.normalization_factors[i] for i in range(self.imgs_lc.shape[0])],
                axis=0
            ) / self.normalization_factors.mean()
        rixs_spectrum = summed_data.sum(axis=1) / self.normalization_factors.sum()
        
        #calculate the energy axis
        energy_loss = self._calculate_energy_axis(auto_alignment=auto_alignment, 
                                                  index_elastic_line=index_elastic_line)

        # Save to file with numpy's savetxt
        np.savetxt(filename_save, 
                   np.column_stack((energy_loss, rixs_spectrum[::-1])),
                   fmt='%.6f %.6f',  # integer for pixels, float with 6 decimal places for intensity
                   delimiter=' ')
        
        print(f"Pre-processed txt dataset saved as {filename_save} \n")

