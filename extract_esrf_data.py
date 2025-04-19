import numpy as np
import re
import h5netcdf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cmcrameri import cm
import h5py
from scipy.stats import norm, poisson
from scipy.signal import fftconvolve
import os
import time
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import CubicSpline
from scipy.signal import correlate
import xarray as xr
from scipy.ndimage import gaussian_filter1d


class esrf_xas:
    def __init__(self, file_path):

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        else:
            self.file_path = file_path
            print(f"File found: {file_path}")

    def extract_motor_positions(self, x):
        """
        Extract motor positions from the HDF5 file and save them into a dictionary.
        
        Parameters:
        x (int or str): The index used to construct the group path within the HDF5 file. 
                        It will be converted to an integer if provided as a string.
        
        Returns:
        dict: A dictionary containing motor names as keys and their positions as values.
        """
        x = int(x)
        self.motor_positions = {}
        try:
            with h5py.File(self.file_path, 'r') as h5_file:
                group_path = f'{x}.1/instrument/positioners'
                for motor_name in h5_file[group_path].keys():
                    self.motor_positions[motor_name] = h5_file[f'{group_path}/{motor_name}'][()]
        except KeyError as e:
            print(f"Dataset not found: {e}")
            return None
        except OSError as e:
            print(f"File error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        return self.motor_positions

    def determine_polarization(self):

        # Check if motor positions are present
        if not hasattr(self, 'motor_positions') or not self.motor_positions:
            raise ValueError("Motor positions have not been extracted. Please run extract_motor_positions first.")
        
        # Determine polarization based on motor positions
        if self.motor_positions['hu70cp'] > 30 and self.motor_positions['hu70ap'] > 30:
            polarization = 'LV'
        elif -2 < self.motor_positions['hu70cp'] < 2 and -2 < self.motor_positions['hu70ap'] < 2:
            polarization = 'LH'
        elif 2 <= self.motor_positions['hu70cp'] <= 30 and 2 <= self.motor_positions['hu70ap'] <= 30:
            polarization = 'C+'
        elif -30 <= self.motor_positions['hu70cp'] <= -2 and -30 <= self.motor_positions['hu70ap'] <= -2:
            polarization = 'C-'
        else:
            polarization = 'Unknown'
        
        return polarization
        


    def extract_xas(self, x):
        """
        Extracts datasets from an HDF5 file based on a given index.
        This function reads an HDF5 file and extracts the 'energy_enc' and 'ifluo_raw' datasets
        from a specific group path determined by the provided index.
        Parameters:
        x (int or str): The index used to construct the group path within the HDF5 file. 
                        It will be converted to an integer if provided as a string.
        Returns:
        tuple: A tuple containing two numpy arrays:
            - energy_enc (numpy.ndarray): The extracted 'energy_enc' dataset.
            - ifluo_raw (numpy.ndarray): The extracted 'ifluo_raw' dataset.
               Returns (None, None) if the datasets are not found or if an error occurs.
        Raises:
        KeyError: If the specified datasets are not found in the HDF5 file.
        OSError: If there is an error opening or reading the HDF5 file.
        Exception: For any other unexpected errors.
        """
        
        x = int(x)
        try:
            with h5py.File(self.file_path, 'r') as h5_file:
                group_path = f'{x}.1/measurement'
                try:
                    self.energy = h5_file[f'{group_path}/energy_enc'][:]
                except:
                    self.energy = h5_file[f'{group_path}/energy'][:]
                self.xas = h5_file[f'{group_path}/ifluo_xmcd'][:]
                self.i0 = h5_file[f'{group_path}/i0_xmcd'][:]
                self.mir = h5_file[f'{group_path}/mir_xmcd'][:]
        except KeyError as e:
            print(f"Dataset not found: {e}")
            return None, None
        except OSError as e:
            print(f"File error: {e}")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None, None
        
        _ = self.extract_motor_positions(x)

        return self.energy, self.xas, self.i0, self.mir
    
    
    @staticmethod
    def normalize_xas(energy, xas, i0=None, mir=None,
                      normalize_by_i0=True, 
                      normalize_by_smoothed_mir=False,
                      linear_bkg=False,
                      poly_order=1,
                      range_baseline=None,
                      range_step=None):
        """
        Normalize the XAS data by dividing by the I0 data and optionally subtracting a linear background.
        
        Parameters:
        xas (numpy.ndarray): The XAS data.
        i0 (numpy.ndarray): The I0 data.
        mir (numpy.ndarray): The MIR data.
        energy (numpy.ndarray): The energy values.
        normalize_by_i0 (bool): A flag indicating whether to normalize the XAS data by the I0 data.
                                If True, the XAS data will be divided by the I0 data.
        normalize_by_smoothed_mir (bool): A flag indicating whether to normalize the XAS data by the smoothed MIR data.
        linear_bkg (bool): A flag indicating whether to fit and subtract a linear background from the XAS data.
                            If True, a linear background will be fitted to the specified ranges and subtracted.
        poly_order (int): The order of the polynomial to fit for the linear background.
        range_baseline (tuple): A tuple containing the energy range (in eV) to use for fitting the linear baseline.
                                The baseline will be calculated as the mean value within this range.
        range_step (tuple): A tuple containing the energy range (in eV) to use for normalizing the XAS data.
                            The step value will be calculated as the mean value within this range.
        
        Returns:
        tuple: A tuple containing two numpy arrays:
            - energy (numpy.ndarray): The energy values.
            - xas_norm (numpy.ndarray): The normalized XAS data.
        """
        
        if normalize_by_i0:

            if i0 is None:
                raise ValueError("I0 data must be provided to normalize by I0.")
            xas_norm_i0 = xas / i0
        else:
            xas_norm_i0 = xas / 1E6

        if normalize_by_smoothed_mir:
            # Apply Gaussian filter to smooth the 'mir' data

            if mir is None:
                raise ValueError("MIR data must be provided to normalize by smoothed MIR.")

            mir_smoothed = gaussian_filter1d(mir, sigma=20)  # Adjust sigma as needed

            # Normalize by the smoothed 'mir' data
            xas_norm_i0 = xas / mir_smoothed

            # # Plot the original and smoothed 'mir' data
            # plt.figure(figsize=(10, 5))
            # plt.plot(self.energy, self.mir, label='Original MIR')
            # plt.plot(self.energy, mir_smoothed, label='Smoothed MIR', linestyle='--')
            # plt.xlabel('Energy (eV)')
            # plt.ylabel('MIR')
            # plt.legend()
            # plt.title('Original and Smoothed MIR')
            # plt.grid()
            # plt.show()

        if not linear_bkg:
            # Subtract the baseline, and normalize by the step
            if range_baseline is not None:
                n1_baseline = np.searchsorted(energy, range_baseline[0])
                n2_baseline = np.searchsorted(energy, range_baseline[1])
                baseline = np.mean(xas_norm_i0[n1_baseline:n2_baseline])
            else:
                baseline = 0

            if range_step is not None:
                n1_step = np.searchsorted(energy, range_step[0])
                n2_step = np.searchsorted(energy, range_step[1])
                step = np.mean(xas_norm_i0[n1_step:n2_step] / baseline -1)
            else:
                step = 1
            xas_norm = (xas_norm_i0 / baseline - 1) / step

        if linear_bkg:
            # Fit a linear background to the specified range
            if range_baseline is not None:
                n1_baseline = np.searchsorted(energy, range_baseline[0])
                n2_baseline = np.searchsorted(energy, range_baseline[1])
                baseline = np.mean(xas_norm_i0[n1_baseline:n2_baseline])
            else:
                raise Exception("Range for baseline fitting must be provided when linear_bkg is True.")
            
            x_fit = energy[n1_baseline:n2_baseline]
            y_fit = xas_norm_i0[n1_baseline:n2_baseline]
            coeffs = np.polyfit(x_fit, y_fit, poly_order)
            linear_bkg = np.polyval(coeffs, energy)
            xas_pure = (xas_norm_i0 - linear_bkg)
            # Normalize by the step
            if range_step is not None:
                n1_step = np.searchsorted(energy, range_step[0])
                n2_step = np.searchsorted(energy, range_step[1])
                step = np.mean(xas_pure[n1_step:n2_step])
            else:
                step = 1

            xas_norm = xas_pure / step

        return energy, xas_norm
    
    def extract_xmcd(self, 
                     runs=None,
                     automatically_sort_pm = True,
                     runs_p=None,
                     runs_m=None,
                     normalize_after=False,
                     normalize_by_i0=True, 
                     normalize_by_smoothed_mir = False,
                     linear_bkg=False,
                     poly_order = 1,
                     range_baseline=None,
                     range_step=None,
                     plot=False):
        
        runs_p = runs_p if runs_p is not None else []
        runs_m = runs_m if runs_m is not None else []

        if automatically_sort_pm:

            print("Automatically sorting C+ and C- runs...")
            runs_p = []
            runs_m = []

            if runs is None:
                raise ValueError("If automatically_sort_pm is True, runs must be provided.")
            for run in runs:
                _ = self.extract_motor_positions(run)
                if self.determine_polarization() == 'C+':
                    runs_p.append(run)
                elif self.determine_polarization() == 'C-':
                    runs_m.append(run)
                else:
                    print(f"Warning: Run {run} is nor C+ neither C- \n\n")
            print(f" Runs C+: {runs_p} \n Runs C-: {runs_m}")

        all_runs_p = []
        all_runs_m = []
        for run in runs_p:
            energy, xas_norm,_,_ = self.extract_xas(run)

            if not normalize_after:
                energy, xas_norm = self.normalize_xas(self.energy, self.xas, i0=self.i0, mir=self.mir,
                                                    normalize_by_i0=normalize_by_i0, 
                                                    normalize_by_smoothed_mir=normalize_by_smoothed_mir,
                                                    linear_bkg=linear_bkg, 
                                                    poly_order=poly_order,
                                                    range_baseline=range_baseline, 
                                                    range_step=range_step)
            if not all_runs_p:
                energy0 = energy
            run_data = np.interp(energy0, energy, xas_norm)
            all_runs_p.append(run_data)
        
        for run in runs_m:
            energy, xas_norm,_,_ = self.extract_xas(run)

            if not normalize_after:
                energy, xas_norm = self.normalize_xas( self.energy, self.xas, i0=self.i0, mir=self.mir,
                                                    normalize_by_i0=normalize_by_i0, 
                                                    normalize_by_smoothed_mir=normalize_by_smoothed_mir,
                                                    linear_bkg=linear_bkg, 
                                                    poly_order=poly_order,
                                                    range_baseline=range_baseline, 
                                                    range_step=range_step)
            
            run_data = np.interp(energy0, energy, xas_norm)
            all_runs_m.append(run_data)

        avg_runs_p = np.mean(all_runs_p, axis=0)
        avg_runs_m = np.mean(all_runs_m, axis=0)

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            
            # Plot all runs_p and their average
            for i, run_data in enumerate(all_runs_p):
                axs[0].plot(energy0, run_data, label=f"Run C+ {runs_p[i]}")
            axs[0].plot(energy0, avg_runs_p, label="Average C+", linewidth=2, color='black')
            axs[0].set_xlabel('Energy (eV)')
            axs[0].set_ylabel('XAS')
            axs[0].legend()
            axs[0].grid()
            axs[0].set_title(f'C+ Runs and Average: B = {self.motor_positions["magnet"]:.1f} T')
            
            # Plot all runs_m and their average
            for i, run_data in enumerate(all_runs_m):
                axs[1].plot(energy0, run_data, label=f"Run C- {runs_m[i]}")
            axs[1].plot(energy0, avg_runs_m, label="Average C-", linewidth=2, color='black')
            axs[1].set_xlabel('Energy (eV)')
            axs[1].set_ylabel('XAS')
            axs[1].legend()
            axs[1].grid()
            axs[1].set_title(f'C- Runs and Average B = {self.motor_positions["magnet"]:.1f} T')
            
            plt.tight_layout()
            plt.show()

        if normalize_after:
            _, avg_runs_p = self.normalize_xas(energy, avg_runs_p, i0=self.i0, mir=self.mir,
                                                normalize_by_i0=normalize_by_i0, 
                                                normalize_by_smoothed_mir=normalize_by_smoothed_mir,
                                                linear_bkg=linear_bkg, 
                                                poly_order=poly_order,
                                                range_baseline=range_baseline, 
                                                range_step=range_step)
            _, avg_runs_m = self.normalize_xas(energy, avg_runs_m, i0=self.i0, mir=self.mir,
                                                normalize_by_i0=normalize_by_i0, 
                                                normalize_by_smoothed_mir=normalize_by_smoothed_mir,
                                                linear_bkg=linear_bkg, 
                                                poly_order=poly_order,
                                                range_baseline=range_baseline, 
                                                range_step=range_step)

        avg = (avg_runs_p + avg_runs_m)/2
        xmcd = (avg_runs_p - avg_runs_m)

        xmcd_data = np.column_stack((energy0, avg, xmcd))

        if plot:
            # Plot XMCD data
            plt.figure(figsize=(10, 4))
            plt.plot(xmcd_data[:, 0], avg_runs_p, label=f"C+", linewidth=2)
            plt.plot(xmcd_data[:, 0], avg_runs_m, label=f"C-", linewidth=2)
            plt.plot(xmcd_data[:, 0], xmcd_data[:, 2] * 3, label=f"3*(C+ - C-)", linewidth=2)
            plt.xlabel('Energy (eV)')
            plt.ylabel('XMCD')
            plt.title(f'XMCD, B = {self.motor_positions["magnet"]:.1f} T')
            plt.legend()
            plt.grid()
            plt.show()


        return all_runs_p, all_runs_m, xmcd_data     
    
    
class esrf_run_spectrum:

    def __init__(self, folder, runs, scans):
        """
        Initialize an esrf_run_spectrum object.

        Parameters
        ----------
        folder : str
            Path to the folder containing .spec files
        runs : list
            List of run numbers to be extracted.
        scans : list of lists
            List of scans to be extracted for each run.
        """
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"The folder {folder} does not exist.")
        
        self.folder = folder
        print(f"Folder found: {folder}")
        self.runs = runs if isinstance(runs, list) else [runs]
        # Check for duplicates in runs
        if len(self.runs) != len(set(self.runs)):
            raise ValueError("Duplicate run numbers found in the provided runs.")
        self.scans = scans
    

    def extract_and_process_spectra(self,
                        x_name,
                        y_name,
                        norm_name, 
                        motor_names,
                        align_spectra=False,
                        pixel_row_start=None,
                        pixel_row_stop=None,
                        fit_shifts=False,
                        smooth_shifts=False,
                        correlation_batch_size=10,
                        poly_order=1,
                        elastic_line_point=None,
                        plot=False,
                        scans_from_same_run=False):
        """
        Process the extracted spectra to align energy shifts and calculate average spectrum.
        Parameters
        ----------
        x_name : str
            Name of the x-axis variable.
        y_name : str
            Name of the y-axis variable.
        norm_name : str
            Name of the normalization variable.
        motor_names : list
            List of motor names to be included in the extraction.
        align_spectra : bool, optional
            Whether to align the spectra. Default is False.
        pixel_row_start : int, optional
            Starting pixel row for the correlation region (typically near elastic line). Default is None.
        pixel_row_stop : int, optional
            Ending pixel row for the correlation region (typically near elastic line). Default is None.
        fit_shifts : bool, optional
            Whether to fit the shifts. Default is False.
        smooth_shifts : bool, optional
            Whether to smooth the shifts. Default is False.
        correlation_batch_size : int, optional
            Size of the batch for correlation. Default is 10.
        poly_order : int, optional
            Polynomial order for fitting shifts. Default is 1.
        elastic_line_point : float, optional
            Point to set the elastic line. Default is None.
        plot : bool, optional
            Whether to plot the spectra. Default is False.
        scans_from_same_run : bool, optional
            Whether the name first line of each scan/run contains the "scan" number, 
            like in the new ESRF spec files. Default is False, like in the old ones.
        """

        if not hasattr(self, 'spectra_xarray'):
            _ = self._extract_1d_runs(self.runs, self.scans, x_name, y_name, norm_name, motor_names,
                                        scans_from_same_run=scans_from_same_run)

        ### find the aligning range
        # Stack all y_data arrays along a new dimension
        x_data = self.spectra_xarray[list(self.spectra_xarray.data_vars)[0]].sel(variable='x').values
        all_y_data = np.stack([self.spectra_xarray[spec].sel(variable='y').values for spec in self.spectra_xarray.data_vars], axis=0)
        # Calculate the average spectrum along the new dimension
        avg_spectrum = np.mean(all_y_data, axis=0)
        # Find aligning range using the _find_aligning_range method
        if pixel_row_start is None or pixel_row_stop is None:
            pixel_row_start, pixel_row_stop = self._find_aligning_range(x_data,avg_spectrum, threshold=0.1)

        if np.shape(all_y_data)[0] == 1:
            align_spectra = False

        if align_spectra:
            ### correct the energy shifts
            _ = self._correct_shift(all_y_data, 
                                    pixel_row_start, pixel_row_stop, 
                                    fit_shifts=fit_shifts, 
                                    smooth_shifts=smooth_shifts,
                                    correlation_batch_size=correlation_batch_size, 
                                    poly_order=poly_order)
            
        _ = self.set_elastic_energy(elastic_line_point=elastic_line_point)
        
        if plot:
            self.plot_spectra(align_spectra, pixel_row_start=pixel_row_start, pixel_row_stop=pixel_row_stop)


    def _search_runs(self, runs):
            """
            Search for filenames corresponding to the specifid runs.
            Returns
            -------
            list
                List of valid image numbers
            """
            # Initialize lists to store results
            file_names = []

            # Ensure runs is a list
            if not isinstance(runs, list):
                runs = [runs]

            # Sort the runs
            runs = sorted(runs)

            # Format run numbers to 4 digits
            formatted_runs = [f"{int(run):02d}" for run in runs]

            # Get all .spec files in directory
            files = sorted(
                (entry for entry in os.scandir(self.folder) if entry.name.lower().endswith('.spec')),
                key=lambda f: f.name  # Sort by file name (assuming natural order by numbers)
            )

            for file in files:
                file_path = os.path.join(self.folder, file)
                for run in formatted_runs:
                    if run in file.name:
                        file_names.append(file_path)

            if not file_names:
                print(f"Warning: No valid .spec files found for runs {runs} \n\n")
            else:
                print(f"Found {len(file_names)} files: runs {runs}.")

            return file_names
    

    def _extract_1d_runs(self, runs, scans,
                        x_name, y_name, norm_name, motor_names,
                        plot=False,
                        scans_from_same_run=False):
        """
        Extract 1D runs from .spec files in the specified folder.

        Parameters
        ----------
        runs : list
            List of run numbers to be extracted.
        x_name : str
            Name of the x-axis variable.
        y_name : str
            Name of the y-axis variable.
        norm_name : str
            Name of the normalization variable.
        motor_names : list
            List of motor names to be included in the extraction.
        xr.Dataset
            An xarray Dataset containing the extracted data for each run and scan.

        Returns
        -------
        list
            List of SpecFile objects for each found .spec file
        """
        # Search for .spec files corresponding to the provided runs
        spec_files = self._search_runs(runs)
        # Ensure scans is a list
        if not isinstance(scans, list):
            scans = [scans]

        # Initialize SpecFile objects for each found .spec file
        self.spectra_xarray = xr.Dataset()
        for i, (file, run) in enumerate(zip(spec_files, runs)):
            specfile = SpecFile(file, run)
            extracted_data = specfile.extract_data(scans[i], x_name, y_name, norm_name, motor_names,
                                                   scans_from_same_run=scans_from_same_run)
            
            for scan_name, data_array in extracted_data.items():
                new_name = f"run_{run}_scan_{scan_name.split('_')[1]}"
                self.spectra_xarray[new_name] = data_array

        if plot:
            self.plot_spectra()

        return self.spectra_xarray


    def save_to_hdf5(self, file_path_save):
        """
        Save the spectra to an HDF5 file.
        """
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_hdf5 is True.")
        SpecFile(ds=self.spectra_xarray).save_to_hdf5(file_path_save)

    def save_to_csv_for_originlab(self, 
                    file_path_save,
                    motor_names, motor_name_mapping):
        """
        Save the spectra to a CSV file.
        """
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_csv is True.")
        SpecFile(ds=self.spectra_xarray).save_to_csv_for_originlab(file_path_save, motor_names, motor_name_mapping)

    def save_to_txt(self, file_path_save, save_avg_spectrum=False):
        """
        Save the spectra to a text file.
        """ 
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_tx is True.")
        SpecFile(ds=self.spectra_xarray).save_to_txt(file_path_save, save_avg_spectrum=save_avg_spectrum)


    def plot_spectra(self, align_spectra=False, pixel_row_start=None, pixel_row_stop=None):
        """
        Plot the extracted spectra.
        """
        if not hasattr(self, 'spectra_xarray'):
            raise ValueError("No spectra have been extracted. Please run extract_1d_runs or process_spectra first.")

        plt.figure()
        max_spectra = 35
        spectra_to_plot = list(self.spectra_xarray.items())[::max(1, len(self.spectra_xarray) // max_spectra)]
        color_list = [cm.lipari(i) for i in np.linspace(0, 1, len(spectra_to_plot))]
        for i, (spec_name, spec_data) in enumerate(spectra_to_plot):
            plt.plot(spec_data.sel(variable='x'), 
                    spec_data.sel(variable='y')/spec_data.sel(variable='norm'), label=spec_name, color=color_list[i])
        
        # Plot the average spectrum
        sum_spectrum = np.sum([self.spectra_xarray[spec].sel(variable='y').values for spec in self.spectra_xarray.data_vars], axis=0)
        sum_norm = np.sum([self.spectra_xarray[spec].sel(variable='norm').values for spec in self.spectra_xarray.data_vars], axis=0)
        avg_spectrum = sum_spectrum / sum_norm
        plt.plot(self.spectra_xarray[spec_name].sel(variable='x'), avg_spectrum, 
                 label='Average Spectrum', linewidth=2, color='black')
        
        # Plot vertical dashed lines for pixel_row_start and pixel_row_stop
        if pixel_row_start is not None and pixel_row_stop is not None:
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_start], color='k', linestyle='--', label='Start Pixel Row')
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_stop], color='k', linestyle='--', label='Stop Pixel Row')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.title('Extracted Spectra')
        plt.grid()
        plt.show()

        #shifts
        if align_spectra:
            plt.figure(figsize=(10,5))
            plt.plot(self.real_shifts, 'ko-', label='Real Shifts')  # 'ko-' for black circles connected by lines
            plt.plot(self.shifts, 'ro-', label='Used Shifts')
            plt.xlabel('Image Index')
            plt.ylabel('Shift Value')
            plt.title('Real Shifts of Images')
            plt.grid()
            plt.legend()
            plt.show()


    @staticmethod
    def _find_aligning_range(x_data, spectrum, threshold=0.05):
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

        # Find where signal rises above threshold * max intensity
        threshold_value = threshold * np.max(spectrum)
        # Calculate the moving average with a window of 3
        moving_avg = np.convolve(spectrum, np.ones(3)/3, mode='same')
        peak_start = np.where(moving_avg > threshold_value)[0]
        
        if len(peak_start) == 0:
            raise ValueError("No peak found above threshold")
        
        # Use the last peak position (elastic line) instead of first peak
        elastic_line = peak_start[-1]
        range_start = elastic_line - 50
        range_stop = elastic_line + 15
        
        print(f"Using range: {range_start}, {range_stop} (x_data: {x_data[range_start]:.2f}, {x_data[range_stop]:.2f})")
        
        # Return the range around the elastic line
        return range_start, range_stop
    
    def set_elastic_energy(self, elastic_line_point=None):
        if elastic_line_point is None:
            raise ValueError("elastic_line_point is not defined. Please set it before calling this method.")

        for spec_name in self.spectra_xarray.data_vars:
            self.spectra_xarray[spec_name].loc[dict(variable='x')] -= elastic_line_point


    def _correct_shift(self, spectra, pixel_row_start, pixel_row_stop, 
                       fit_shifts, smooth_shifts, correlation_batch_size, poly_order):
        """
        Corrects energy shifts between spectra by aligning them to the first spectrum.

        The alignment is performed by:
        1. Computing cross-correlation between each spectrum and the first spectrum
        2. Finding the optimal shift that maximizes correlation
        3. Applying the shift correction using interpolation
        4. Storing the shift values for each spectrum

        Parameters
        ----------
        pixel_row_start : int
            Starting pixel row for the correlation region (typically near elastic line)
        pixel_row_stop : int 
            Ending pixel row for the correlation region (typically near elastic line)
        """
        print(f"Calculating energy shifts of {spectra.shape[0]} spectra. Shifting spectra by: ")
        start_time = time.perf_counter()

        # Initialize arrays
        self.shifts = np.zeros(spectra.shape[0])
        self.real_shifts = self.shifts.copy()
        self.corrected_spectra = np.zeros_like(spectra)
        
        # Pre-average the spectra to have a more reliable cross-correlation
        if correlation_batch_size > 1:
            spec_avg = self._average_images_in_batches(spectra, np.min((correlation_batch_size, spectra.shape[0])))
        else:
            spec_avg = spectra.copy()

        # Get reference spectrum once (avoid recomputing for each iteration)
        ref_spectrum = spec_avg[0, :]

        real_shifts = np.zeros(spec_avg.shape[0])
        # Calculate the shift for the remaining spectra
        for num_spectrum in range(1, spec_avg.shape[0]):

            # Get current spectrum
            curr_spectrum = spec_avg[num_spectrum, :]
            
            # Calculate shift
            real_shifts[num_spectrum] = self._correlate_spectra(
                curr_spectrum, ref_spectrum, pixel_row_start, pixel_row_stop)

        index_aux = np.arange(0, spectra.shape[0], correlation_batch_size)[:spectra.shape[0] // correlation_batch_size]
        if fit_shifts:
            coeffs = np.polyfit(index_aux, real_shifts, deg=poly_order)
            self.shifts = np.polyval(coeffs, range(0, spectra.shape[0]))
        elif smooth_shifts:
            # Extrapolate by specifying left and right values
            self.shifts = np.interp(range(0, spectra.shape[0]), index_aux, real_shifts, left=real_shifts[0], right=real_shifts[-1])          
            sigma = correlation_batch_size   # Adjust sigma based on correlation_batch_size
            self.shifts = gaussian_filter1d(self.shifts, sigma=sigma)
        else:
            self.shifts = np.interp(range(0, spectra.shape[0]), index_aux, real_shifts, left=real_shifts[0], right=real_shifts[-1]) 

        
         
        # Save the real shifts somewhere for plotting
        for i in range(0, real_shifts.shape[0]):
            self.real_shifts[i * correlation_batch_size:(i + 1) * correlation_batch_size] = real_shifts[i]

        for num_spectrum, (spec_name, _) in enumerate(self.spectra_xarray.items()):
            if abs(self.shifts[num_spectrum]) > 0.5:
                print(f"{self.shifts[num_spectrum]:.2f}, ", end="")
                
                # Interpolate
                interp = np.interp(
                    np.arange(spectra.shape[1]) - self.shifts[num_spectrum], 
                    np.arange(spectra.shape[1]), 
                    spectra[num_spectrum, :],
                    left=0, right=0
                )
                
                self.spectra_xarray[spec_name].loc[dict(variable='y')] = interp.copy()
            else:
                self.shifts[num_spectrum] = 0
                print("0.00, ", end="")


        print("")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _correlate_spectra(self, spec, spec_ref, pixel_start, pixel_stop):
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

        # if factor == 1:
        #     # Skip interpolation if factor is 1
        #     xx = np.arange(pixel_start, pixel_stop + 1)
        #     xx_extended = np.arange(pixel_start-20, pixel_stop + 1)
        #     spec = spec[xx]  # Use the original spectrum directly
        #     spec_ref = spec_ref[xx]  # Use the original reference spectrum directly
        # else:
        #     xx = np.arange(pixel_start, pixel_stop + 1 / factor, 1 / factor)
        #     spec = np.interp(xx, np.arange(0, len(spec)), spec)  # Interpolating the spectrum
        #     spec_ref = np.interp(xx, np.arange(0, len(spec_ref)), spec_ref)  # Interpolating the reference spectrum

        # crosscorr = correlate(spec/np.mean(spec), spec_ref/np.mean(spec_ref))
        # lag_values = np.arange(-spec.shape[0] + 1, spec.shape[0], 1)
        # lag = lag_values[np.argmax(crosscorr)] / factor

        if factor > 1:
            # xx = np.arange(0, len(spec_ref) + 1 / factor, 1 / factor)
            xx = np.linspace(0, len(spec_ref)-1, (len(spec_ref) - 1) * factor + 1)
            spec = np.interp(xx, np.arange(0, len(spec)), spec)  # Interpolating the spectrum
            spec_ref = np.interp(xx, np.arange(0, len(spec_ref)), spec_ref)  # Interpolating the reference spectrum

        lag_values, crosscorr = self._custom_cross_correlation(spec, spec_ref, int(pixel_start*factor), int(pixel_stop*factor))
        crosscorr_restricted = crosscorr[(lag_values >= -10) & (lag_values <= 10)]
        first_index = np.argmax(lag_values >= -10)
        lag = (lag_values[np.argmax(crosscorr_restricted)] + first_index) / factor

        return lag

    @staticmethod
    def _custom_cross_correlation(g, f, window_start, window_end):
        """
        Cross-correlate f and g, evaluating only within [window_start:window_end] of f.
        The shifted g always uses its full values (not zero-padded).
        
        Returns:
            lags: array of lag values
            correlations: correlation value at each lag
        """
        
        f = np.asarray(f)
        g = np.asarray(g)

        N = len(f)
        M = len(g)

        window_indices = np.arange(window_start, window_end)
        norm_f = np.linalg.norm(f[window_indices])
        lags = np.arange(-M + 1, N)  # All possible lags
        
        correlations = []

        for lag in lags:
            # Shift g by lag (positive lag: shift right)
            g_shifted = np.zeros_like(f)
            # if lag==0:
            #     print("ciao")
            for i in range(N):
                g_idx = i - lag  # reverse of convolution
                if 0 <= g_idx < M:
                    # if g_idx == 160:
                    #     print('ciao')
                    g_shifted[i] = g[g_idx]
            
            # Compute dot product only inside the window
            norm_g = np.linalg.norm(g_shifted[window_indices])
            if norm_g == 0:
                correlations.append(0)
                continue
            dot = np.dot(f[window_indices]/norm_f, g_shifted[window_indices]/norm_g)
            correlations.append(dot)

        return lags, np.array(correlations)


    @staticmethod
    def _average_images_in_batches(spectra, batch_size):
        """
        Averages images in batches from a stack of 1D spectra.

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
        n_images = spectra.shape[0]
        
        if batch_size == 0:
            raise ValueError("Batch size must be greater than zero. Consider increasing max_lc_images.")
        
        n_batches = n_images // batch_size
        averaged_spectra = np.zeros((n_batches, spectra.shape[1]))

        for i in range(n_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            averaged_spectra[i,:] = np.mean(spectra[start_index:end_index, :], axis=0)       

        return averaged_spectra


class SpecFile:
    def __init__(self, filename=None, run=None, ds=None):
        
        if ds is None and filename is None:
            raise ValueError("Either 'ds' or 'filename' must be provided.")
        if ds is None:
            #no xarray dataset provided, reading from file
            self.filename = filename
            self.file_content = self._read_file()
            self.run = run 
        else:
            #xarray dataset provided
            self.ds = ds
    
    def _read_file(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"File {self.filename} not found.")
    
    def _read_spec_file(self, scans_selected, scans_from_same_run=False):
        """
        This method extracts scan data, column names, motor names and values, and the date from a .spec file.
        It then organizes this information into an xarray.Dataset.
        Parameters
        ----------
        scans : str or list of str
            List of scan numbers to extract. If a string is provided, it will be split into a list of scan numbers.
        Returns
        -------
        xarray.Dataset
            An xarray.Dataset containing the scan data and metadata, including column names, motor names, motor values, and the date.
        """
        data_all = []
        colname_all = []
        motval_all = []
        motval_all_scans = []
        motname_all_scans = []
        motname_all = []
        date = None
        
        if scans_selected == ['all']:

            if scans_from_same_run:
                scan_pattern = r"#S (\d+)  Scan \d+:(\d+)"
            else:
                scan_pattern = r"#S (\d+)"

            scan_positions = [m.start() for m in re.finditer(scan_pattern, self.file_content)]
            scans = [re.search(scan_pattern, self.file_content[pos:]).group(2) for pos in scan_positions]
        else:
            scans = scans_selected
        scans = scans.split() if isinstance(scans, str) else scans
        
        for scan in scans:

            motname_all = []
            motval_all = []
            if scans_from_same_run:
                scan_pattern = rf'#S (\d+)  Scan \d+:{scan}(?:\s|$)'
            else:
                scan_pattern = rf'#S {scan}(?:\s|$)'

            scan_positions = [m.start() for m in re.finditer(scan_pattern, self.file_content)]
            if len(scan_positions) == 0:
                print(f"Scan {scan} not found.")
                continue
            elif len(scan_positions) > 1:
                scan_start = scan_positions[1]
            else:
                scan_start = scan_positions[0]
            scan_end = self.file_content.find("#S", scan_start + 1)
            scan_content = self.file_content[scan_start:scan_end] if scan_end != -1 else self.file_content[scan_start:]
            
            # Extract column names
            col_match = re.search(r"#L\s{1,2}(.+)", scan_content)
            # col_match = re.search(r"#L  (.+)", scan_content)
            colnames = col_match.group(1).split('  ') if col_match else []
            colname_all.append(colnames)
            
            # Extract motor names and values
            motname_matches = re.findall(r"#O\d+\s{1,2}(.+)", scan_content)
            for motname_match in motname_matches:
                motnames = motname_match.split('  ') if motname_match else []
                motname_all.extend(motnames)
            motname_all_scans.append(motname_all)
            
            motval_matches = re.findall(r"#P\d+\s{1,2}(.+)", scan_content)
            for motval_match in motval_matches:
                motvals = [float(val) if val != 'b\'ERR\'' else np.nan for val in re.split(r'\s{1,2}', motval_match.strip())] if motval_match else []
                motval_all.extend(motvals)
            motval_all_scans.append(motval_all)
            
            # Extract numerical data
            data_lines = [line for line in scan_content.split("\n") if not line.startswith("#") and not line.startswith(" ") and line.strip()]
            data = np.loadtxt(data_lines) if data_lines else np.array([])
            data_all.append(data)
            
            # Extract date
            if not date:
                date_match = re.search(r"#D (.+)", self.file_content)
                date = date_match.group(1) if date_match else None
        
        # Create xarray Dataset
        ds = xr.Dataset()
        for i, scan in enumerate(scans):
            if i < len(data_all):
                ds[f'scan_{scan}'] = xr.DataArray(
                    data_all[i],
                    dims=['points', 'datasets'],
                    coords={
                    'points': np.arange(data_all[i].shape[0]),  # Progressive values from 0
                    'datasets': colname_all[i]
                    }
                )
            ds[f'scan_{scan}'].attrs.update({name: value for name, value in zip(motname_all_scans[i], motval_all_scans[i])})
        
        ds.attrs['date'] = date
        return ds

    def extract_data(self,scans, x_name, y_name, norm_name, motor_names=None,
                     scans_from_same_run=False):
        """
        Extract and normalize data based on specified x, y, and normalization datasets,
        and include specified motor names and values.

        Parameters
        ----------
        x_name : str
            Name of the dataset to be used as x-axis.
        y_name : str
            Name of the dataset to be used as y-axis.
        norm_name : str
            Name of the dataset to be used for normalization.
        motor_names : list of str
            List of motor names to include in the output.

        Returns
        -------
        xarray.Dataset
            Dataset containing the normalized data and specified motor values.
        """
        ds = self._read_spec_file(scans_selected=scans, scans_from_same_run=scans_from_same_run)
        normalized_data = xr.Dataset()
        
        for scan in ds.data_vars:
            if x_name in ds[scan].datasets and y_name in ds[scan].datasets and norm_name in ds[scan].datasets:
                x_data = np.copy(ds[scan].sel(datasets=x_name).values)
                y_data = np.copy(ds[scan].sel(datasets=y_name).values)
                norm_data = np.copy(ds[scan].sel(datasets=norm_name).values)

                if motor_names:  
                    motor_values = [ds[scan].attrs[motor] for motor in motor_names if motor in ds[scan].attrs]
                
                # Concatenate x_data, y_data, and norm_data along a new dimension
                data = np.stack([x_data, y_data, norm_data], axis=1)
                
                # Create the DataArray with multiple coordinates for the 'points' dimension
                normalized_data[scan] = xr.DataArray(
                    data=data,
                    dims=['points', 'variable'],
                    coords={
                    'points': np.arange(data.shape[0]),
                    'variable': ['x', 'y', 'norm']
                    }
                )
                normalized_data[scan].attrs['x_name'] = x_name
                normalized_data[scan].attrs['y_name'] = y_name
                normalized_data[scan].attrs['norm_name'] = norm_name
                
                if motor_names:
                    for motor, value in zip(motor_names, motor_values):
                        normalized_data[scan].attrs[motor] = value

                normalized_data[scan].attrs['date'] = ds.attrs['date']
                normalized_data[scan].attrs['filename'] = self.run
        
        return normalized_data

    def save_to_hdf5(self, filename):
        """
        Save the xarray.Dataset to an HDF5 file with motor values as attributes.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to save.
        filename : str
            The name of the HDF5 file to save the dataset to.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        # for scan in self.ds.data_vars:
        #     motor_values = self.ds[scan].motor_values.values
        #     self.ds[scan].attrs['motor_values'] = motor_values.tolist()
        
        self.ds.to_netcdf(filename, engine='h5netcdf')
        print(f"Dataset saved to {filename}")

    def save_to_csv_for_originlab(self, filename, motor_names, motor_name_mapping,
                                  metadata_in_origin=['Q', 'theta','2theta','phi','energy','polarization',
                                                      'mirror','sample','B [T]', 'T [K]', 'comments'],
                                  save_avg_spectrum=False):
        """
        Save the xarray.Dataset to a CSV file with columns 'Energy Loss (eV)', 'Intensity (arb. units)', and 'Error (arb. units)'.
        Include a header with the values of the specified motor parameters.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to save.
        filename : str
            The name of the CSV file to save the dataset to.
        motor_names : list of str
            List of motor names to include in the header.
        motor_name_mapping : dict
            Dictionary mapping motor names in the dataset to motor names expected by OriginLab.
        """

        # Ensure the filename ends with ".csv"
        if not filename.lower().endswith('.csv'):
            base, ext = os.path.splitext(os.path.basename(filename))
            if ext.lower() != '.csv':
                print(f"Warning: Changing file extension to .csv for {filename}")
                filename = os.path.join(os.path.dirname(filename), base + '.csv')
            filename += '.csv'

        with open(filename, 'w', encoding='utf-8') as file:

            # Write the header with motor values
            # Collect motor values for all scans
            motor_values_all_scans = {motor: [] for motor in motor_names}
            for scan in self.ds.data_vars:
                for motor in motor_names:
                    if motor in self.ds[scan].attrs:
                        motor_values_all_scans[motor].append(self.ds[scan].attrs[motor])
                    else:
                        motor_values_all_scans[motor].append("")

            header = ''
            # Write motor values in the header
            for metadata in metadata_in_origin:
                header_parts = [metadata]
                if metadata in motor_name_mapping:
                    if motor_name_mapping[metadata] is not None:
                        header_parts += [
                            f"{value:.2f}" for value in motor_values_all_scans[motor_name_mapping[metadata]]
                        ]
                else:
                    header_parts += [" "] * len(motor_values_all_scans[motor_names[0]])
                header += ','.join(header_parts * 2)  # Repeat each motor value twice
                header += '\n'

            # Write a line of "Energy Loss" and {scan} alternating
            header += ','.join([f"Energy Loss,{scan}" for scan in self.ds.data_vars])
            header += '\n'
            units = ','.join(['(eV),(arb. units)'] * len(self.ds.data_vars))
            units += '\n'
            header += units
            file.write(f"{header}\n")

            # Write the column names
            if save_avg_spectrum:
                # Sum all the RIXS spectra and save a single 'x' and 'y'
                sum_spectrum = np.sum([self.ds[scan].sel(variable='y').values for scan in self.ds.data_vars], axis=0)
                sum_norm = np.sum([self.ds[scan].sel(variable='norm').values for scan in self.ds.data_vars], axis=0)
                avg_spectrum = sum_spectrum / sum_norm
                x_values = self.ds[list(self.ds.data_vars)[0]].sel(variable='x').values
                # Write the data directly to the file
                for x, y in zip(x_values, avg_spectrum):
                    file.write(f"{x},{y}\n")
            else:
                # Create a DataFrame to store all scans
                data_frames = []
                # Stack all x and y values as adjacent columns
                all_data = []
                for scan in self.ds.data_vars:
                    x_values = self.ds[scan].sel(variable='x').values
                    y_values = self.ds[scan].sel(variable='y').values
                    all_data.append(np.column_stack((x_values, y_values)))

                # Concatenate all data along the second axis
                concatenated_data = np.concatenate(all_data, axis=1)

                # Write the concatenated data to the file
                for row in concatenated_data:
                    file.write(','.join(map(str, row)) + '\n')

        print(f"Dataset saved to {filename}")


    def save_to_txt(self, filename, save_avg_spectrum=False):
        """
        Save the xarray.Dataset to a text file with just the datasets

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to save.
        filename : str
            The name of the text file to save the dataset to.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        
        if save_avg_spectrum:
            # Sum all the RIXS spectra and save a single 'x' and 'y'
            sum_spectrum = np.sum([self.ds[scan].sel(variable='y').values for scan in self.ds.data_vars], axis=0)
            sum_norm = np.sum([self.ds[scan].sel(variable='norm').values for scan in self.ds.data_vars], axis=0)
            avg_spectrum = sum_spectrum / sum_norm
            x_values = self.ds[list(self.ds.data_vars)[0]].sel(variable='x').values
            # Write the data directly to the file
            np.savetxt(filename, np.concatenate([x_values, avg_spectrum]), delimiter=',', fmt='%.6f, %.6f')
            print(f"Dataset with only average spectrum saved to {filename}")
        else:
            # Create a DataFrame to store all scans
            data_frames = []
            # Stack all x and y values as adjacent columns
            all_data = []
            for scan in self.ds.data_vars:
                x_values = self.ds[scan].sel(variable='x').values
                y_values = self.ds[scan].sel(variable='y').values
                all_data.append(np.column_stack((x_values, y_values)))

            # Concatenate all data along the second axis
            concatenated_data = np.concatenate(all_data, axis=1)

            # Write the concatenated data to the file using numpy's savetxt
            np.savetxt(filename, concatenated_data, delimiter=' ', fmt='%.6f')
            print(f"Dataset saved to {filename}")