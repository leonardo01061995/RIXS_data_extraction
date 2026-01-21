import os
import time
import h5py
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.interpolate import RegularGridInterpolator as rgi
from cmcrameri import cm
from abc import ABC, abstractmethod
import fabio
import fnmatch
from nexusformat.nexus import *
import xarray as xr

from rixs_images import EDF_Image, TPS_Image, DLS_Image
from one_d_rixs_spectra import Generated_1D_RIXS_Spectra
      

class RIXS_Raw_Images:
    def __init__(
            self, 
            facility,
            folder, 
            run_number,
            identifier_string,
            exp_number,
            first_absolute_image_number = -1):
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
        if facility not in ["ESRF", "TPS", "DLS"]:
            raise ValueError("Facility must be either 'ESRF' or 'TPS' or 'DLS'")
        self.identifier_string = identifier_string
        self.run_number = sorted(run_number) if isinstance(run_number, list) else run_number
        self.exp_number = exp_number

        #variables we need to store
        self.all_imgs_processed = None
        self.shifts = None
        self.real_shifts = None
        self.file_names = None
        self.pixel_row_start = None
        self.pixel_row_stop = None
        self.one_d_processed_spectra = []
        self.normalization_factors = []
        self.first_absolute_image_number = first_absolute_image_number

        #first find the images
        start_time = time.perf_counter()
        if self.facility == "ESRF":
            self._search_images_ESRF()
        elif self.facility == "TPS":
            self._search_images_TPS()
        elif self.facility == "DLS":
            self._search_images_DLS()

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
    
    def _search_images_DLS(self):
        """
        this function finds a file in the directory where the experimental data is defined to be

        params:
        self: class object,  used for getting the directory = path
        number: int, the run number of the file to be found.
        path: str, the directory where the file is to be found

        return:
        result[-1]: str, the last file found in the directory
        """
        print(f"Retrieving filenames for run {self.run_number}:")

        start_time = time.perf_counter()
        path = self.folder

        for number in self.run_number:

            processed = False  #whether to search for processed or raw files

            result = []
            number=str(number)
            #path = self.directory
            for root, dirname, files in os.walk(path): 
                #we walk through the files using os.walk, where the folder to be analized is defined with the path (directory = root)
                for name in files:
                    if fnmatch.fnmatch(name, '*'+number+'*.nxs'):

                        if processed:
                            if "processed" in name:
                                result.append(os.path.join(root, name))
                        else:
                            if "processed" not in name:
                                result.append(os.path.join(root, name))

            if not result:
                raise FileNotFoundError(f"No files found for the specified run number {number}.")
            
            for file in result:
                print(f" - {file}")
            self.file_names = result

        return result 
    

    def generate_rixs_spectra(self, 
                              use_spc,
                            extract_curvature = False,
                            spc_parameters = {},
                            no_spc_parameters = {},
                            additional_metadata = {},
                            align_images = True,
                            pixel_row_start=None, pixel_row_stop=None,
                            find_aligning_range=False, 
                            process_shifts='', 
                            correlation_batch_size=1,
                            poly_order=1,
                            align_images_by_shifting_photons=False,
                            plot_generation=False,
                            keep_2d_images=False):
        """
        Process multiple RIXS image files using single photon counting
        
        Parameters
        ----------
        use_spc : bool, optional
            If True, use single photon counting, default True
        spc_parameters : dict, optional
            Parameters for single photon counting, default {}. List:
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

        align_images : bool, optional
            If True, align images based on cross-correlation, default True
        pixel_row_start : int
            Starting pixel row for the correlation region (typically near elastic line)
        pixel_row_stop : int 
            Ending pixel row for the correlation region (typically near elastic line)

        keep_2d_images : bool, optional
            If True, keep the 2D images after processing. This uses much more memory.
            Can be avoided for large datasets if only 1D spectra are needed. Default True
            
        Returns
        -------
        numpy.ndarray
            3D array containing processed images stacked along first axis
        """

        if not self.file_names:
            return

        #single photon counting parameters
        self.use_spc = use_spc
        self.align_images = align_images
        if process_shifts not in ['', 'fit', 'smooth']:
            raise ValueError("process_shifts must be one of '', 'fit', or 'smooth'.")

        # Check if curvature extraction is disabled and no curve parameters are provided
        curve_a = spc_parameters.get("curve_a", 0) if use_spc else no_spc_parameters.get("curve_a", 0)
        curve_b = spc_parameters.get("curve_b", 0) if use_spc else no_spc_parameters.get("curve_b", 0)
        if not extract_curvature and (curve_a == 0 and curve_b == 0):
            print("Careful: Curvature extraction is disabled and the given slope is zero")

        if not find_aligning_range and (pixel_row_start is None or pixel_row_stop is None):
            raise ValueError("If find_aligning_range is False, both pixel_row_start and pixel_row_stop must be provided.")
        
        self.ds_1d = xr.Dataset()
        processed_images = []
        # Determine axis names
        y_name = "Photons" if use_spc else "Counts"
        if self.facility == "ESRF":
            norm_name = "mirror"
        elif self.facility == "DLS":
            norm_name = "m4c1"
        elif self.facility == "TPS":
            norm_name = "monitor"
        else:
            norm_name = "norm"
             
        print("Performing generation of RIXS spectra from images and curvature correction.")
        start_time = time.perf_counter()
        raw_imgs = []
        for run_index, filename in enumerate(self.file_names):
            # Create edf_image instance and process
            if self.facility == "ESRF":
                img = EDF_Image(filename)
            elif self.facility == "TPS":
                img = TPS_Image(filename)
            elif self.facility == "DLS":
                img = DLS_Image(filename)
            else:
                raise ValueError("Facility must be either 'ESRF', 'TPS' or 'DLS'")
            
            if extract_curvature:
                # Call the find_curvature method with appropriate parameters
                curve_a, curve_b = self._find_curvature(img.raw_data.mean(axis=0), 
                                                    frangex=(0, img.raw_data.shape[2]), 
                                                    frangey=(0, img.raw_data.shape[1]))   
                if use_spc:
                    spc_parameters["curve_a"] = curve_a
                    spc_parameters["curve_b"] = curve_b
                else:
                    no_spc_parameters["curve_a"] = curve_a

                    no_spc_parameters["curve_b"] = curve_b

            _, _ = img.process_imgs(use_spc=self.use_spc,
                                            spc_parameters=spc_parameters,
                                            no_spc_parameters=no_spc_parameters,
                                            plot_generation=plot_generation)

            raw_imgs.append(img)

            #save each spectrum inside a xarray dataarray. All the spectra from different runs will be saved inside ds xarray dataset
            for i, rixs_2d in enumerate(img.imgs_processed):

                # Create x, y, norm arrays as 1D vectors
                x = np.arange(rixs_2d.shape[1])/spc_parameters["subdivide_bins_factor_y"] if use_spc else np.arange(rixs_2d.shape[1])
                y = rixs_2d.mean(axis=1)
                norm = np.full_like(x, img.normalization_factor[i])

                # Create DataArray with 1D vectors stacked along axis=1
                da = xr.DataArray(
                    np.stack([x, y, norm], axis=1),
                    dims=['points', 'variable'],
                    coords={"points": np.arange(x.shape[0]),
                            'variable': ['x', 'y', 'norm']},
                    attrs={
                        **getattr(img, "attributes", {}),
                        **additional_metadata,
                        "x_name": 'Pixel',
                        "y_name": y_name,
                        "norm_name": norm_name,
                        'run_number': img.run_number,
                    }
                )
                self.ds_1d[f"{i}"] = da

                #append the 1D processed spectra to the list
                self.one_d_processed_spectra.append(y)
            
            # img.normalization_factor is an array (one value per sub-image); extend the list with its elements
            self.normalization_factors.extend(np.asarray(img.normalization_factor).tolist())
            if keep_2d_images or align_images_by_shifting_photons:
                processed_images.append(img.imgs_processed)

        processed_images = np.stack(processed_images, axis=0)  # Shape: (n_images, n_rows, n_cols)
        self.normalization_factors = np.stack(self.normalization_factors)
        mean_normalization_factor = np.mean(self.normalization_factors)
        std_normalization_factor = np.std(self.normalization_factors)
        print(f"Normalization factor: {mean_normalization_factor} +- {std_normalization_factor}")
        print(f"Total elapsed time for spectrum generation: {time.perf_counter() - start_time:.2f} seconds. \n")

        if self.align_images and len(self.one_d_processed_spectra) > 1:
            print("Calculating energy correlation and aligning images. 2D images will be stored in memory.", end="\n\n")
            if find_aligning_range:
                self.pixel_row_start, self.pixel_row_stop = self._find_aligning_range(processed_images, threshold=0.1)
            else:
                self.pixel_row_start = pixel_row_start
                self.pixel_row_stop = pixel_row_stop

            ###### ONly calculate the shift: then re-bin each image
            self._calculate_shift(self.pixel_row_start, self.pixel_row_stop, process_shifts=process_shifts,
                            correlation_batch_size=correlation_batch_size, poly_order=poly_order)

            if use_spc and align_images_by_shifting_photons:
                ###### now re-bin the images if you used single-photon counting
                print("Re-binning images with known shifts by vertically shifting photons.")
                for i,img in enumerate(raw_imgs):
                        processed_images[i,:,:] = img.single_photon_counting(
                            **spc_parameters,
                            vertical_shift = self.shifts[i])[0] 
                # self.all_imgs_processed = np.stack(processed_images, axis=0)
                        
            else:
                print("Shifting images with known shifts by interpolation.")
                # self.all_imgs_processed = np.zeros((len(processed_images), processed_images[0].shape[0], processed_images[0].shape[1]))
                for num_image, processed_img in enumerate(processed_images):
                    if abs(self.shifts[num_image])>0.25: #if the shift is too large, we shift the image
                        xdim,ydim=processed_img.shape
                        x=np.arange(xdim)
                        y=np.arange(ydim)
                        interp = rgi((x-self.shifts[num_image], y), processed_img, bounds_error=False, fill_value=0)

                        xx,yy=np.meshgrid(x,y)
                        processed_images[num_image,:,:] = interp((xx,yy)).T #shifting the images

            self.all_imgs_processed = processed_images


        elif self.align_images and len(self.one_d_processed_spectra) == 1:
            print("Only one file found. No energy correlation necessary.")
            self.all_imgs_processed = processed_images
        else:
            print("Aligning images is disabled. No energy correlation will be performed.")
            if keep_2d_images:
                # self.ds_2d = xr.Dataset()
                # #save the imgs from processed_imgs, and the attributes from the rixs_img objects
                # for ii, (img_2d, img_object) in enumerate(zip(processed_images, raw_imgs)):
                #     da = xr.DataArray(
                #         img_2d,
                #         dims=['points', 'variable'],
                #         coords={"points": np.arange(x.shape[0]),
                #                 'variable': ['x', 'y', 'norm']},
                #         attrs={
                #             **getattr(img_object, "attributes", {}),
                #             **additional_metadata,
                #             "x_name": 'Pixel',
                #             "y_name": y_name,
                #             "norm_name": norm_name,
                #             'run_number': img_object.run_number,
                #         }
                #     )
                #     self.ds_2d[f"{ii}"] = da.copy()
                self.imgs_imgs_processed = processed_images
            

        #old code for alignment
        # ########## Alignment, but only for single-photon counting images if wanted
        # if self.align_images and len(self.one_d_processed_spectra) > 1 and use_spc and align_images_by_shifting_photons: 
        #     #correct the shift
        #     if find_aligning_range:
        #         self.pixel_row_start, self.pixel_row_stop = self._find_aligning_range(threshold=0.1)
        #     else:
        #         self.pixel_row_start = pixel_row_start
        #         self.pixel_row_stop = pixel_row_stop

        #     ###### ONly calculate the shift: then re-bin each image
        #     self._calculate_shift(self.pixel_row_start, self.pixel_row_stop, fit_shifts, 
        #                     correlation_batch_size, poly_order)
               
        #     ###### now re-bin the images if you used single-photon counting
        #     print("Re-binning images with known shifts by vertically shifting photons.")
        #     processed_images = []
        #     for i,img in enumerate(processed_images):
        #             processed_images.append(img.single_photon_counting(
        #                 **spc_parameters,
        #                 vertical_shift = self.shifts[i])[0])         

        # elif self.align_images and img.n_images == 1: 
        #     print("Only one file found. No energy correlation necessary.")
        # # else:
        # #     print("Aligning images is disabled. No energy correlation will be performed.")

        return Generated_1D_RIXS_Spectra(self.ds_1d)


    def _find_aligning_range(processed_images, threshold=0.1):
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
        
        spectrum = processed_images.mean(axis=0).mean(axis=1)
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
            img1 = self.all_imgs_processed[:,:,:].mean(axis=0)
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
            
            if self.all_imgs_processed.shape[0]>1:
                plt.axvline(x=self.pixel_row_start, color='k', linestyle='--', label='Start Pixel Row')
                plt.axvline(x=self.pixel_row_stop, color='k', linestyle='--', label='Stop Pixel Row')
            plt.legend()
            plt.show()

        if self.align_images and plot_shifts and self.all_imgs_processed.shape[0] > 1:
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


    def save_2d_dataset_hdf5(self, path_dir, filename_save_auto=True, filename_save='',
                         save_normalized_images=True):
        """
        Save the pre-processed dataset to a hdf5 file, containing a single dataset 'data'
        and attributes with experiment number, filepath_dir and run_number.
        Parameters
        ----------
        path_dir : str
            Directory where to save the file
        filename_save_auto : bool
            If True, automatically generate the filename based on run number and number of images
        filename_save : str
            If filename_save_auto is False, use this filename to save the file
        """
        
        if not filename_save_auto and filename_save == '':
            raise ValueError("Either filename_save_auto should be True or filename_save should be provided.")
        
        if filename_save_auto:
            if isinstance(self.run_number, list):
                run_number_str = "_".join(f"{int(num):04d}" for num in self.run_number)  # Join list elements with "_" and format as 4 digits
            else:
                run_number_str = f"{int(self.run_number):04d}"  # Convert single run number to a 4-digit string

            filename_save = f"{self.folder}_run_{run_number_str}_{np.shape(self.imgs_lc)[0]:03d}_imgs.hdf5"

        full_path = os.path.join(path_dir, filename_save)
        #calculate the normalized images
        if save_normalized_images:
            #calculate the normalized images
            mean_norm_factor = self.normalization_factors.mean() 
            imgs_to_save = np.array([self.all_imgs_processed[i, :, :] / self.normalization_factors[i] * mean_norm_factor for i in range(self.all_imgs_processed.shape[0])])

        with h5py.File(full_path, 'w') as hf:
            hf.create_dataset('data', data=imgs_to_save, dtype='float32')
            hf.attrs['exp_number'] = self.exp_number
            hf.attrs['filename_dir'] = self.folder
            # Save only the first element if self.run_number is a list
            if isinstance(self.run_number, list):
                hf.attrs['run_number'] = f"{self.folder}_run#_{int(self.run_number[0]):04d}"  # Save formatted run number
            else:
                hf.attrs['run_number'] = f"{self.folder}_run#_{int(self.run_number):04d}"

        print(f"Pre-processed dataset saved as {filename_save} \n\n")


    def save_2d_dataset_xarray(self, path_dir=None, filename_save_auto=True, filename_save=''):
        # Validate ds_1d
        if not hasattr(self, "ds_1d") or len(self.ds_1d.keys()) == 0:
            raise ValueError("self.ds_1d is empty or missing. Run generate_rixs_spectra first.")

        # Determine output directory and filename
        if path_dir is None:
            path_dir = self.folder
        if not filename_save_auto and filename_save == '':
            raise ValueError("Either filename_save_auto should be True or filename_save should be provided.")

        n_entries = len(self.ds_1d.keys())
        if filename_save_auto:
            if isinstance(self.run_number, list):
                run_number_str = "_".join(f"{int(num):04d}" for num in self.run_number)
            else:
                run_number_str = f"{int(self.run_number):04d}"
            filename_save = f"{os.path.basename(self.folder)}_run_{run_number_str}_{n_entries:03d}_imgs.nc"
        full_path = os.path.join(path_dir, filename_save)

        # Find source 2D image container (try several known attributes)
        if self.all_imgs_processed is None:
            raise ValueError("Images not yet processed. Ensure 2D images are available before saving.")

        if self.all_imgs_processed.shape[0] < n_entries:
            raise ValueError("Number of available 2D images does not match number of entries in ds_1d.")

        # Build new xarray Dataset with 2D DataArrays
        ds_2d = xr.Dataset()
        sorted_keys = sorted(self.ds_1d.keys(), key=lambda k: int(k))
        for idx, key in enumerate(sorted_keys):
            da_1d = self.ds_1d[key]
            # extract scalar normalization (norm was stored as repeated vector)
            norm_val = None
            if 'variable' in da_1d.coords and 'norm' in da_1d.coords['variable'].values:
                norm_vec = da_1d.sel(variable='norm').values
                if norm_vec.size > 0:
                    norm_val = float(norm_vec.flat[0])
            if norm_val is None:
                norm_val = 1.0

            x_coords = np.arange(self.all_imgs_processed.shape[2])
            y_coords = np.arange(self.all_imgs_processed.shape[1])

            attrs = dict(da_1d.attrs) if hasattr(da_1d, "attrs") else {}
            attrs.update({"normalization_factor": float(norm_val)})

            da2 = xr.DataArray(
                self.all_imgs_processed[idx,:,:],
                dims=("row", "col"),
                coords={"row": y_coords, "col": x_coords},
                attrs=attrs
            )
            ds_2d[key] = da2

        # assign to instance and save to netcdf
        ds_2d.to_netcdf(full_path)
        print(f"2D xarray dataset saved to {full_path}")

        return ds_2d


    def _calculate_shift(self, pixel_row_start, pixel_row_stop, 
                         process_shifts='', correlation_batch_size=1, poly_order=1):

        """
        correct_shift2 method

        This method applies the calculated shifts to the images stored in the 
        all_imgs_processed attribute. It iterates through each image, checking the 
        magnitude of the shift. If the shift exceeds a threshold (0.5 in this 
        case), it performs interpolation to adjust the image based on the 
        calculated shift. If the shift is within the threshold, the original 
        image is directly assigned to the imgs_lc attribute without modification. 
        The method also tracks and prints the elapsed time for the operation.

        Returns
        -------
        None
        """

        print(f"Calculating energy shift of {self.all_imgs_processed.shape[0]} images.")
        start_time = time.perf_counter()

        # Initialize arrays
        self.shifts = np.zeros(self.all_imgs_processed.shape[0])
        self.real_shifts = self.shifts.copy()
        
        #pre-average the images to have a more reliable cross-correlation
        #spec_avg has shape: (all_imgs_processed.shape[1], all_imgs_processed.shape[0]//batch_size + reminder)
        spec_avg = self.average_images_in_batches(self.all_imgs_processed, min(correlation_batch_size, self.all_imgs_processed.shape[0]))
        
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

        # index_aux = range(0, self.all_imgs_processed.shape[0], correlation_batch_size)
        index_aux = np.arange(0, self.all_imgs_processed.shape[0], correlation_batch_size)[:self.all_imgs_processed.shape[0] // correlation_batch_size]

        if process_shifts=='fit':
            coeffs = np.polyfit(index_aux, real_shifts, deg=poly_order)
            self.shifts = np.polyval(coeffs, range(0, self.all_imgs_processed.shape[0]))
        elif process_shifts=='smooth':
            # Extrapolate by specifying left and right values
            self.shifts = np.interp(range(0, self.all_imgs_processed.shape[0]), index_aux, real_shifts, left=real_shifts[0], right=real_shifts[-1])          
            sigma = correlation_batch_size   # Adjust sigma based on correlation_batch_size
            self.shifts = gaussian_filter1d(self.shifts, sigma=sigma)
        else:
            self.shifts = np.interp(range(0, self.all_imgs_processed.shape[0]), index_aux, real_shifts, left=real_shifts[0], right=real_shifts[-1]) 


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
    
    @staticmethod
    def load_dark_image(filename, dataset_name):
        """
        Load a dark image from a hdf5 file.

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
    
    def _calculate_energy_axis(self,calibration, auto_alignment, index_elastic_line=None):
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
            * calibration/1000/self.subdivide_bins_factor_y
        
        energy_loss = energy_loss[::-1]      

        return energy_loss 
    
    def save_pre_processed_dataset(self, path_dir=None, filename_save_auto=True, filename_save = '', auto_alignment=True, index_elastic_line=None):
        if self.facility == "ESRF":
            self.save_pre_processed_dataset_ESRF(path_dir, filename_save_auto, filename_save)
        elif self.facility == "TPS":
            self.save_pre_processed_dataset_TPS(path_dir, filename_save_auto, filename_save, auto_alignment, index_elastic_line)

    # def save_pre_processed_dataset_ESRF(self, path_dir, filename_save_auto, filename_save = ''):
    #     """
    #     Save the pre-processed dataset to a file
    #     """
        
    #     if not filename_save_auto and filename_save == '':
    #         raise ValueError("Either filename_save_auto should be True or filename_save should be provided.")
        
    #     if filename_save_auto:
    #         if isinstance(self.run_number, list):
    #             run_number_str = "_".join(f"{int(num):04d}" for num in self.run_number)  # Join list elements with "_" and format as 4 digits
    #         else:
    #             run_number_str = f"{int(self.run_number):04d}"  # Convert single run number to a 4-digit string

    #         filename_save = f"{self.identifier_string}_run_{run_number_str}_{np.shape(self.imgs_lc)[0]:03d}_imgs.hdf5"

    #     full_path = os.path.join(path_dir, filename_save)
    #     with h5py.File(full_path, 'w') as hf:
    #         hf.create_dataset('data', data=self.imgs_lc, dtype='float32')
    #         hf.attrs['exp_number'] = self.exp_number
    #         hf.attrs['filename_dir'] = self.identifier_string
    #         hf.attrs['normalization_factor'] = self.normalization_factors.sum()
    #         hf.attrs['factor_ADC'] = self.factor_ADC
    #         hf.attrs['calibration'] = self.calibration
    #         hf.attrs['curve_a'] = self.curve_a
    #         hf.attrs['curve_b'] = self.curve_b
    #         hf.attrs['roi_x'] = self.roi_x
    #         hf.attrs['roi_y'] = self.roi_y
    #         hf.attrs['subdivide_bins_factor_x'] = self.subdivide_bins_factor_x
    #         hf.attrs['subdivide_bins_factor_y'] = self.subdivide_bins_factor_y
    #         # Save only the first element if self.run_number is a list
    #         if isinstance(self.run_number, list):
    #             hf.attrs['run_number'] = f"{self.identifier_string}_run#_{int(self.run_number[0]):04d}"  # Save formatted run number
    #         else:
    #             hf.attrs['run_number'] = f"{self.identifier_string}_run#_{int(self.run_number):04d}"

    #     print(f"Pre-processed dataset saved as {filename_save} \n\n")


    # def save_pre_processed_dataset_TPS(self, path_dir=None, filename_save_auto=True, filename_save = '',
    #                                auto_alignment=True, index_elastic_line=None):
    #     """
    #     Save the pre-processed dataset to a file in HDF5 format.
    #     Parameters:
    #     -----------
    #     path_dir : str
    #         The directory path where the file will be saved.
    #     filename_save_auto : bool
    #         If True, automatically generate the filename based on the object's attributes.
    #     filename_save : str, optional
    #         The custom filename to save the dataset. If not provided and `filename_save_auto` is False, 
    #         a ValueError will be raised.
    #     auto_alignment : bool, optional
    #         If True, automatically align the spectrum based on the elastic line.
    #     index_elastic_line : int, optional
    #         Index of the elastic line in the spectrum. If not provided, it will be calculated.
    #     Raises:
    #     -------
    #     ValueError
    #         If both `filename_save_auto` is False and `filename_save` is an empty string.
    #     Notes:
    #     ------
    #     - The method saves the dataset (`self.imgs_lc`) and additional metadata as attributes in the HDF5 file.
    #     - If `self.run_number` is a list, only the first element is saved as part of the metadata.
    #     - The generated filename (if `filename_save_auto` is True) includes the directory path, run number, 
    #       and the number of images in the dataset.
    #     """
        
    #     if not filename_save_auto and filename_save == '':
    #         raise ValueError("Either filename_save_auto should be True or filename_save should be provided.")
        
    #     if path_dir is None:
    #         path_dir = self.folder
    #         print("Warning: Path directory is not provided. Using the default folder path.")
        
    #     if filename_save_auto:
    #         if isinstance(self.run_number, list):
    #             filename_save = os.path.join(path_dir, f"rixs_{os.path.basename(self.file_names[0]).split('_')[1]}_{'_'.join([str(rn) for rn in self.run_number])}_processed.hdf5")
    #         else:
    #             filename_save = os.path.join(path_dir, f"{os.path.splitext(os.path.basename(self.file_names[0]))[0][:-4]}_processed.hdf5")


    #     full_path = os.path.join(path_dir, filename_save)
    #     with h5py.File(full_path, 'w') as hf:
    #         # Open the first file to copy attributes
    #         with h5py.File(self.file_names[0], 'r') as original_file:
    #             for attr_name, attr_value in original_file.attrs.items():
    #                 hf.attrs[attr_name] = attr_value

    #         # Modify the 'header' attribute to replace the "Iph" value
    #         header = hf.attrs['header']
    #         header_parts = header.split(":")[1].split(",")
    #         for i, part in enumerate(header_parts):
    #             if "Iph" in part:
    #                 key, value = part.strip().split(" ")
    #                 decimal_places = len(value.split(".")[1]) if "." in value else 0
    #                 new_value = f"{self.normalization_factors.sum():.{decimal_places}f}"
    #                 header_parts[i] = f"{key} {new_value}"
    #         hf.attrs['header'] = header.split(":")[0]+": "+ ",".join(header_parts)

    #         # Save the sum of self.imgs_lc, each multiplied by the corresponding normalization factor
    #         summed_data = np.sum(
    #             [self.imgs_lc[i] * self.normalization_factors[i] for i in range(self.imgs_lc.shape[0])],
    #             axis=0
    #         ) / self.normalization_factors.mean()
    #         rixs_spectrum = summed_data.sum(axis=1)

    #         # Save the summed data to the new file
    #         hf.create_dataset('data', data=summed_data)

    #         # Save the 1D RIXS spectrum
    #         #calculate the energy axis
    #         energy_loss = self._calculate_energy_axis(auto_alignment=auto_alignment, 
    #                                               index_elastic_line=index_elastic_line)
    #         hf.create_dataset('rixs_spectrum', data=np.column_stack((energy_loss, rixs_spectrum[::-1])))

    #         # Save the normalization factor as an attribute
    #         hf.attrs['normalization_factor'] = self.normalization_factors.sum()
    #         hf.attrs['factor_ADC'] = self.factor_ADC
    #         hf.attrs['calibration'] = self.calibration
    #         hf.attrs['curve_a'] = self.curve_a
    #         hf.attrs['curve_b'] = self.curve_b
    #         hf.attrs['roi_x'] = self.roi_x
    #         hf.attrs['roi_y'] = self.roi_y
    #         hf.attrs['subdivide_bins_factor_x'] = self.subdivide_bins_factor_x
    #         hf.attrs['subdivide_bins_factor_y'] = self.subdivide_bins_factor_y

    #     print(f"Pre-processed hdf5 dataset saved as {filename_save} \n")


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

