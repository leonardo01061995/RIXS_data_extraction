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
from scipy.special import wofz
from scipy.optimize import curve_fit
import pandas as pd
from IPython.display import display
from spec_files import SpecFile
from one_d_rixs_spectra import Generated_1D_RIXS_Spectra


class ESRF_XAS_Spectrum:
    def __init__(self, filepaths, runs):
        """
        Initialize an esrf_multiple_xas object.

        Parameters
        ----------
        filepaths : list
            List of paths to the HDF5 files.
        runs : list
            List of run numbers to be extracted.
        """
        self.runs = runs if isinstance(runs, list) else [runs]
        # Check for duplicates in runs
        if len(self.runs) != len(set(self.runs)):
            raise ValueError("Duplicate run numbers found in the provided runs.")
        
        # Check if filepaths is a list or a single string
        if isinstance(filepaths, list):
            if len(filepaths) != len(runs):
                raise ValueError("If filepaths is a list, it must have the same length as runs.")
            if not all(isinstance(f, str) and os.path.isfile(f) for f in filepaths):
                raise ValueError("All elements in filepaths must be valid file paths.")
        elif isinstance(filepaths, str):
            filepaths = [filepaths] * len(runs)
            if not all(os.path.isfile(f) for f in filepaths):
                raise ValueError("filepaths must be a valid file path when provided as a string.")
        else:
            raise ValueError("filepaths must be either a list of file paths or a single file path string.")
        
        self.filepaths = filepaths


    def _extract_motor_positions(self, filepath, x):
        """
        Extract motor positions from the HDF5 file and save them into a dictionary.
        
        Parameters:
        x (int or str): The index used to construct the group path within the HDF5 file. 
                        It will be converted to an integer if provided as a string.
        
        Returns:
        dict: A dictionary containing motor names as keys and their positions as values.
        """
        x = int(x)
        motor_positions = {}
        try:
            with h5py.File(filepath, 'r') as h5_file:
                group_path = f'{x}.1/instrument/positioners'
                for motor_name in h5_file[group_path].keys():
                    motor_positions[motor_name] = h5_file[f'{group_path}/{motor_name}'][()]
        except KeyError as e:
            print(f"Dataset not found: {e}")
            return None
        except OSError as e:
            print(f"File error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        
        motor_positions['polarization'] = self._determine_polarization(motor_positions)

        return motor_positions

    @staticmethod
    def _determine_polarization(motor_positions):
        
        # Determine polarization based on motor positions
        if motor_positions['hu70cp'] > 30 and motor_positions['hu70ap'] > 30:
            polarization = 'LV'
        elif -2 < motor_positions['hu70cp'] < 2 and -2 < motor_positions['hu70ap'] < 2:
            polarization = 'LH'
        elif 2 <= motor_positions['hu70cp'] <= 30 and 2 <= motor_positions['hu70ap'] <= 30:
            polarization = 'C+'
        elif -30 <= motor_positions['hu70cp'] <= -2 and -30 <= motor_positions['hu70ap'] <= -2:
            polarization = 'C-'
        else:
            polarization = 'Unknown'
        
        return polarization
        
    def _extract_xas(self, filepath, x):
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
            with h5py.File(filepath, 'r') as h5_file:
                group_path = f'{x}.1/measurement'
                try:
                    energy = h5_file[f'{group_path}/energy_enc'][:]
                except:
                    energy = h5_file[f'{group_path}/energy'][:]
                xas = h5_file[f'{group_path}/ifluo_xmcd'][:]
                i0 = h5_file[f'{group_path}/i0_xmcd'][:]
                mir = h5_file[f'{group_path}/mir_xmcd'][:]
        except KeyError as e:
            print(f"Dataset not found: {e}")
            return None, None
        except OSError as e:
            print(f"File error: {e}")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None, None
        
        motor_positions = self._extract_motor_positions(filepath, x)
        data = np.stack((energy, xas, i0, mir), axis=1)
        return data, motor_positions
    
    def _normalize_xas(self,
                        energy,
                        xas,
                        i0=None,
                        mir=None,
                        normalization_method='i0',
                        poly_coeffs_i0=None,
                        poly_order_baseline=1,
                        poly_order_step=1,
                        energy_edge=None,
                        range_baseline=None,
                        range_step=None,
                        normalize_step=False,
                        remove_slope_step=False):
        """
        Normalize the XAS data by dividing by the I0 data and optionally subtracting a linear background.
        
        Parameters:
        xas (numpy.ndarray): The XAS data.
        i0 (numpy.ndarray): The I0 data.
        mir (numpy.ndarray): The MIR data.
        energy (numpy.ndarray): The energy values.
        normalization_method (str): The normalization method to use. Options are 'i0', 'smoothed_mir', 'baseline'
        poly_order_baseline (int): The order of the polynomial to fit for the linear background.
        poly_order_step (int): The order of the polynomial to fit for the step.
        range_baseline (tuple): A tuple containing the energy range (in eV) to use for fitting the linear baseline.
                                The baseline will be calculated as the mean value within this range.
        range_step (tuple): A tuple containing the energy range (in eV) to use for normalizing the XAS data.
                            The step value will be calculated as the mean value within this range.
        
        Returns:
        tuple: A tuple containing two numpy arrays:
            - energy (numpy.ndarray): The energy values.
            - xas_norm (numpy.ndarray): The normalized XAS data.
        """
        if range_baseline is None: 
            raise Exception("Range for baseline fitting must be provided.")
        
        if range_step is None:
            raise Exception("Range for step fitting must be provided.")
        
        if normalization_method not in ['i0', 'smoothed_mir', 'baseline', 'i0_fit']:
            raise ValueError("Invalid normalization method. Choose 'i0', 'smoothed_mir', or 'baseline'.")
        
        n1_baseline = np.searchsorted(energy, range_baseline[0])
        n2_baseline = np.searchsorted(energy, range_baseline[1])

        if normalization_method == 'i0':
            if i0 is None:
                raise ValueError("I0 data must be provided to normalize by I0.")
            xas_norm = xas / i0
            norm_signal = i0

        elif normalization_method == 'smoothed_mir':
            # Apply Gaussian filter to smooth the 'mir' data
            
            if mir is None:
                raise ValueError("MIR data must be provided to normalize by smoothed MIR.")

            mir_smoothed = gaussian_filter1d(mir, sigma=40)  # Adjust sigma as needed

            # Normalize by the smoothed 'mir' data
            xas_norm = xas / mir_smoothed 
            norm_signal = mir_smoothed

            # # Plot the original and smoothed 'mir' data
            # plt.figure(figsize=(10, 5))
            # plt.plot(energy, mir, label='Original MIR')
            # plt.plot(energy, mir_smoothed, label='Smoothed MIR', linestyle='--')
            # plt.xlabel('Energy (eV)')
            # plt.ylabel('MIR')
            # plt.legend()
            # plt.title('Original and Smoothed MIR')
            # plt.grid()
            # plt.show()

        elif normalization_method == 'i0_fit':
            if i0 is None:
                raise ValueError("I0 data must be provided to normalize by I0.")
            # Fit a polynomial to the I0 data
            i0_fit = np.polyval(poly_coeffs_i0, energy)
            baseline = np.mean(xas[n1_baseline:n2_baseline])
            xas_norm = xas / baseline
            xas_norm = xas_norm / i0_fit
            norm_signal = i0_fit

        elif normalization_method == 'baseline':
            baseline = np.mean(xas[n1_baseline:n2_baseline])
            xas_norm = xas / baseline
            norm_signal = np.full_like(energy, baseline)

        #remove the background in the baseline        
        x_fit = energy[n1_baseline:n2_baseline]
        y_fit = xas_norm[n1_baseline:n2_baseline]
        coeffs_baseline = np.polyfit(x_fit, y_fit, poly_order_baseline)
        linear_bkg_baseline = np.polyval(coeffs_baseline, energy)
        xas_pure = (xas_norm - linear_bkg_baseline)

        if remove_slope_step or normalize_step:
            #Calculate the step
            # Normalize by the step
            n1_step = np.searchsorted(energy, range_step[0])
            n2_step = np.searchsorted(energy, range_step[1])
            x_fit = energy[n1_step:n2_step]
            y_fit = xas_norm[n1_step:n2_step]
            coeffs_step = np.polyfit(x_fit, y_fit, poly_order_step)
            linear_bkg_step = np.polyval(coeffs_step, energy)

        if remove_slope_step:
            max_order_poly = max(poly_order_baseline+1, poly_order_step+1)
            # Expand coeffs_baseline and coeffs_step to have dimension max_order_poly, filling new values with zeros at the beginning
            coeffs_baseline = np.pad(coeffs_baseline, (max_order_poly - len(coeffs_baseline), 0), mode='constant')
            coeffs_step = np.pad(coeffs_step, (max_order_poly - len(coeffs_step), 0), mode='constant')
            if energy_edge is None:
                raise ValueError("Energy edge must be provided to remove slope after the edge.")
            # Compute the difference between the two polynomials after energy_edge, excluding the constant term
            # Get the indices after energy_edge
            indices_after_edge = np.where(energy > energy_edge)[0]
            if len(indices_after_edge) > 0:
                # Remove the constant term (last coefficient) from both polynomials
                coeffs_baseline_no_const = coeffs_baseline.copy()
                coeffs_baseline_no_const[-1] = 0
                coeffs_step_no_const = coeffs_step.copy()
                coeffs_step_no_const[-1] = 0
                # Compute the difference polynomial (without constant term)
                diff_poly_coeffs = coeffs_step_no_const - coeffs_baseline_no_const
                # Evaluate the difference polynomial at the energy points after energy_edge
                diff_curve = np.polyval(diff_poly_coeffs, energy[indices_after_edge])
                # Subtract this curve from xas_pure after energy_edge
                xas_pure[indices_after_edge] -= diff_curve - np.polyval(diff_poly_coeffs, energy_edge)

        xas_processed = xas_pure.copy()
        if normalize_step:
            if energy_edge is None:
                raise ValueError("Energy edge must be provided to normalize by step.")

            step = np.polyval(coeffs_step, energy_edge)-np.polyval(coeffs_baseline, energy_edge)
            xas_processed /= step


        return energy, xas_processed, norm_signal
    
    def extract_xmcd(self, 
                     automatically_sort_pm = True,
                     runs_p=None,
                     runs_m=None,
                     plot=False,
                     save_hdf5=False,
                     filepath_save='',
                     **norm_params):
        """
        Extract XMCD data from the specified runs and filepaths.
        Parameters
        ----------
        automatically_sort_pm : bool
            Whether to automatically sort runs into C+ and C- based on motor positions.
        runs_p : list, optional
            List of runs to be considered as C+ (positive polarization) runs.
        runs_m : list, optional
            List of runs to be considered as C- (negative polarization) runs.
        plot : bool
            Whether to plot the extracted XMCD data.
        save_hdf5 : bool
            Whether to save the extracted data to an HDF5 file.
        filepath_save : str
            The file path to save the HDF5 file.
        norm_params : dict  
            Parameters for normalization. Contains:
                normalization_method : str
                    The normalization method to use. Options are 'i0', 'smoothed_mir', 'baseline'
                poly_order_baseline : int
                    The order of the polynomial to fit for the linear background.
                poly_order_step : int
                    The order of the polynomial to fit for the step.
                energy_edge : float
                    The energy edge for removing the slope.
                range_baseline : tuple
                    A tuple containing the energy range (in eV) to use for fitting the linear baseline.
                range_step : tuple
                    A tuple containing the energy range (in eV) to use for normalizing the XAS data.
                normalize_step : bool
                    Whether to normalize the XAS data by the step.
                remove_slope_step : bool    
                    Whether to remove the slope after the energy edge.
        """
        
        runs_p = runs_p if runs_p is not None else []
        runs_m = runs_m if runs_m is not None else []

        if automatically_sort_pm:

            print("Automatically sorting C+ and C- runs...")
            runs_p = []
            filepaths_p = []
            runs_m = []
            filepaths_m = []

            for filepath, run in zip(self.filepaths, self.runs):
                motor_positions = self._extract_motor_positions(filepath, run)
                if self._determine_polarization(motor_positions) == 'C+':
                    runs_p.append(run)
                    filepaths_p.append(filepath)
                elif self._determine_polarization(motor_positions) == 'C-':
                    runs_m.append(run)
                    filepaths_m.append(filepath)
                else:
                    print(f"Warning: Run {run} is nor C+ neither C- \n\n")
            print(f" Runs C+: {runs_p} \n Runs C-: {runs_m}")

        #extract C+ scans
        self.filepaths = filepaths_p
        self.runs = runs_p
        self.all_xas_ds_p = self.extract_and_normalize_xas(**norm_params)
        
        self.avg_ds_p = self._average_xas(self.all_xas_ds_p)

        #extract C- scans
        self.filepaths = filepaths_m
        self.runs = runs_m
        self.all_xas_ds_m = self.extract_and_normalize_xas(**norm_params)
        
        
        #interpolate all the runs on the same energy0, from C+ xas spectra
        energy0 = self.avg_ds_p['avg_xas'].sel(variable='energy').values
        for run in self.all_xas_ds_m.data_vars:
            energy = self.all_xas_ds_m[run].sel(variable='energy').values
            xas_norm = self.all_xas_ds_m[run].sel(variable='xas_norm').values
            # i0 = self.all_xas_ds_m[run].sel(variable='i0').values
            # mir = self.all_xas_ds_m[run].sel(variable='mir').values
            norm_signal = self.all_xas_ds_m[run].sel(variable='norm').values

            #interpolate
            xas_norm_interp = np.interp(energy0, energy, xas_norm)
            # i0_interp = np.interp(energy0, energy, i0)
            # mir_interp = np.interp(energy0, energy, mir)
            norm_signal_interp = np.interp(energy0, energy, norm_signal)

            #replace the values
            self.all_xas_ds_m[run].loc[dict(variable='energy')] = energy0
            self.all_xas_ds_m[run].loc[dict(variable='xas_norm')] = xas_norm_interp
            # self.all_xas_ds_m[run].loc[dict(variable='i0')] = i0_interp
            # self.all_xas_ds_m[run].loc[dict(variable='mir')] = mir_interp  
            self.all_xas_ds_m[run].loc[dict(variable='norm')] = norm_signal_interp

        self.avg_ds_m = self._average_xas(self.all_xas_ds_m) 

        #calculate the data structure for xmcd
        xmcd = self.avg_ds_p['avg_xas'].sel(variable='xas_norm').values - self.avg_ds_m['avg_xas'].sel(variable='xas_norm').values

        self.xmcd_ds = xr.Dataset()
        self.xmcd_ds['C+'] = self.avg_ds_p['avg_xas'].copy(deep=True)
        self.xmcd_ds['C-'] = self.avg_ds_m['avg_xas'].copy(deep=True)
        self.xmcd_ds['xmcd'] = xr.DataArray(
            data=np.stack((energy0, xmcd), axis=1),
            dims=['points', 'variable'],
            coords={
                'points': np.arange(energy0.shape[0]),
                'variable': ['energy', 'xas_norm']
            }
        )

        if plot:
            self.plot_xmcd(string_title='XMCD')

        if save_hdf5:
            self.save_xas_to_hdf5(filepath_save, self.xmcd_ds)


    def extract_and_normalize_xas(self,
                     plot=False,
                     string_title='XAS',
                     save_hdf5=False,
                     save_avg_xas=False,
                     filepath_save='',
                     **norm_params):
        """
        Extract and normalize XAS data from the specified files.
        Parameters
        ----------
        plot : bool
            Whether to plot the extracted and normalized XAS data.
        string_title : str
            The title for the plot.
        save_hdf5 : bool
            Whether to save the extracted data to an HDF5 file.
        save_avg_xas: bool
            Whether to save the average XAS data to an HDF5 file.
        filepath_save : str
            The file path to save the HDF5 file.
        norm_params : dict
            Parameters for normalization. Contains:
                normalization_method : str
                    The normalization method to use. Options are 'i0', 'smoothed_mir', 'baseline'
                poly_order_baseline : int
                    The order of the polynomial to fit for the linear background.
                poly_order_step : int
                    The order of the polynomial to fit for the step.
                energy_edge : float
                    The energy edge for removing the slope.
                range_baseline : tuple
                    A tuple containing the energy range (in eV) to use for fitting the linear baseline.
                range_step : tuple
                    A tuple containing the energy range (in eV) to use for normalizing the XAS data.
                normalize_step : bool
                    Whether to normalize the XAS data by the step.
                remove_slope_step : bool    
                    Whether to remove the slope after the energy edge.
        """

        self.all_xas_ds = xr.Dataset()
        for ii, (filepath,run) in enumerate(zip(self.filepaths, self.runs)):
            data, motor_positions = self._extract_xas(filepath, run)

            _, xas_norm, norm_signal = self._normalize_xas(data[:,0], data[:,1], i0=data[:,2], mir=data[:,3],
                                              **norm_params)

            # data = np.stack((data[:,0], data[:,1], xas_norm, data[:,2], data[:,3]), axis=1)
            data = np.stack((data[:,0], data[:,1], xas_norm, norm_signal), axis=1)

            # Save data and motor positions as an xarray inside the xr.Dataset
            data_xr = xr.DataArray(
                data,
                dims=['points', 'variable'],
                coords={
                    'points': np.arange(data.shape[0]),
                    'variable': ['energy', 'xas', 'xas_norm', 'norm']
                }
            )
            # Add motor positions as attributes
            if motor_positions is not None:
                for k, v in motor_positions.items():
                    data_xr.attrs[k] = v
            # Add to the dataset with a unique name
            data_xr.attrs['filepath'] = filepath
            data_xr.attrs['run'] = run
            data_xr.attrs['normalization_method'] = norm_params.get('normalization_method')
            data_xr.attrs['poly_order_baseline'] = norm_params.get('poly_order_baseline')
            data_xr.attrs['poly_order_step'] = norm_params.get('poly_order_step')
            data_xr.attrs['energy_edge'] = norm_params.get('energy_edge')
            data_xr.attrs['range_baseline'] = norm_params.get('range_baseline')
            data_xr.attrs['range_step'] = norm_params.get('range_step')
            data_xr.attrs['normalize_step'] = str(norm_params.get('normalize_step'))
            data_xr.attrs['remove_slope_step'] = str(norm_params.get('remove_slope_step'))

            #save in the main xr.Dataset
            self.all_xas_ds[f'{ii}'] = data_xr.copy(deep=True)

        # Calculate and store the average XAS as an attribute
        self.avg_xas_ds = self._average_xas(self.all_xas_ds)
        # self.all_xas_ds[f'avg_xas'] = self.avg_xas['avg_xas'].copy(deep=True)

        if plot:
            self.plot_xas(string_title=string_title)

        if save_hdf5:
            if save_avg_xas:
                self.save_xas_to_hdf5(filepath_save, self.avg_xas_ds)
            else:
                self.save_xas_to_hdf5(filepath_save, self.all_xas_ds)

        return self.all_xas_ds

    @staticmethod
    def _average_xas(all_xas_ds):
        """
        Calculate the average XAS data from all runs.
        """
        
        # Calculate the average XAS data
        energy0 = all_xas_ds[list(all_xas_ds.data_vars)[0]].sel(variable='energy').values
        xas_list = []
        xas_norm_list = []
        # i0_list = []
        # mir_list = []
        norm_list = []
        runs=[]
        for ii, run_data in enumerate(all_xas_ds.data_vars):
            energy = all_xas_ds[run_data].sel(variable='energy').values
            xas = all_xas_ds[run_data].sel(variable='xas').values
            xas_norm = all_xas_ds[run_data].sel(variable='xas_norm').values
            # i0 = all_xas_ds[run_data].sel(variable='i0').values
            # mir = all_xas_ds[run_data].sel(variable='mir').values
            norm_values = all_xas_ds[run_data].sel(variable='norm').values
            xas_list.append(np.interp(energy0, energy, xas))
            xas_norm_list.append(np.interp(energy0, energy, xas_norm))
            # i0_list.append(np.interp(energy0, energy, i0))
            # mir_list.append(np.interp(energy0, energy, mir))
            norm_list.append(np.interp(energy0, energy, norm_values))
            runs.append(all_xas_ds[run_data].attrs.get('run', '?'))
            if ii==0:
                #copy all the attribures (e.g. motor positions) from the first run
                other_attrs = {key: value for key, value in all_xas_ds[run_data].attrs.items()
                                if key not in ["filepath", "run"]}
                filepath = os.path.dirname(all_xas_ds[run_data].attrs.get('filename', '?'))

            
            # Check for differences in attributes across runs
            for key, value in other_attrs.items():
                if key in all_xas_ds[run_data].attrs:
                    current_value = all_xas_ds[run_data].attrs[key]
                    if isinstance(current_value, (int, float)) and isinstance(value, (int, float)):
                        # Perform numeric comparison
                        if abs(current_value - value) > 0.1:
                            print(f"******Warning******: Attribute '{key}' differs by more than 0.1 between spectra. "
                                f"Value in current spectrum: {current_value}, "
                                f"Value in other_attrs: {value}")
                    else:
                        # Perform string comparison
                        if str(current_value) != str(value):
                            print(f"******Warning******: Attribute '{key}' differs between spectra. "
                                f"Value in current spectrum: {current_value}, "
                                f"Value in other_attrs: {value}")

        avg_xas_xarray = xr.DataArray(
            data=np.stack([
                energy0,
                np.mean(xas_list, axis=0),
                np.mean(xas_norm_list, axis=0),
                # np.mean(i0_list, axis=0),
                # np.mean(mir_list, axis=0)
                np.mean(norm_list, axis=0)
            ], axis=1),
            dims=['points', 'variable'],
            coords={
                'points': np.arange(len(energy0)),
                'variable': ['energy', 'xas', 'xas_norm', 'norm']
            }
        )
        avg_xas_xarray.attrs['filepath'] = filepath
        avg_xas_xarray.attrs['run'] = runs
        for key, value in other_attrs.items():
            avg_xas_xarray.attrs[key] = value
        avg_xas = xr.Dataset({'avg_xas': avg_xas_xarray})

        return avg_xas

    def plot_xas(self, string_title=''):
        plt.figure(figsize=(10, 4))
        
        # Plot all runs_p and their average
        for i, run_data in enumerate(self.all_xas_ds.data_vars):
            energy = self.all_xas_ds[run_data].sel(variable='energy').values
            data_norm = self.all_xas_ds[run_data].sel(variable='xas_norm').values
            plt.plot(energy, data_norm, label=f"Run {self.all_xas_ds[run_data].attrs.get('run', run_data)}")
        # Plot the average XAS if it exists
        if hasattr(self, 'avg_xas') and self.avg_xas is not None:
            avg_xas = self.avg_xas['avg_xas'].sel(variable='xas_norm').values
            plt.plot(self.avg_xas['avg_xas'].sel(variable='energy').values, avg_xas, 
                     label="Average XAS", linewidth=2, color='black')
        plt.xlabel('Energy (eV)')
        plt.ylabel('XAS (arb. units)')
        plt.legend()
        plt.grid()
        plt.title(f'{string_title}')
        plt.show()             

    def plot_xmcd(self, string_title=''):
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = np.empty((2, 2), dtype=object)
        axs[0, 0] = fig.add_subplot(gs[0, 0])
        axs[0, 1] = fig.add_subplot(gs[0, 1])
        # Merge the two columns in the second row for XMCD
        axs[1, 0] = fig.add_subplot(gs[1, :])
        
        # Plot all runs_p and their average
        for i, run_data in enumerate(self.all_xas_ds_p.data_vars):
            energy = self.all_xas_ds_p[run_data].sel(variable='energy').values
            data_norm = self.all_xas_ds_p[run_data].sel(variable='xas_norm').values
            axs[0,0].plot(energy, data_norm, label=f"Run C+ {self.all_xas_ds_p[run_data].attrs.get('run')}")
        # Plot the average XAS if it exists
        energy0 = self.avg_ds_p['avg_xas'].sel(variable='energy').values
        avg_runs_p = self.avg_ds_p['avg_xas'].sel(variable='xas_norm').values
        axs[0,0].plot(energy0, avg_runs_p, label="Average C+", linewidth=2, color='black')
        axs[0,0].set_xlabel('Energy (eV)')
        axs[0,0].set_ylabel('XAS')
        axs[0,0].legend()
        axs[0,0].grid()
        axs[0,0].set_title('C- Runs'+string_title)
        
        # Plot all runs_m and their average
        for i, run_data in enumerate(self.all_xas_ds_m.data_vars):
            energy = self.all_xas_ds_m[run_data].sel(variable='energy').values
            data_norm = self.all_xas_ds_m[run_data].sel(variable='xas_norm').values
            axs[0,1].plot(energy, data_norm, label=f"Run C+ {self.all_xas_ds_m[run_data].attrs.get('run')}")
        # Plot the average XAS if it exists
        energy0 = self.avg_ds_m['avg_xas'].sel(variable='energy').values
        avg_runs_m = self.avg_ds_m['avg_xas'].sel(variable='xas_norm').values
        axs[0,1].plot(energy0, avg_runs_m, label="Average C+", linewidth=2, color='black')
        axs[0,1].set_xlabel('Energy (eV)')
        axs[0,1].set_ylabel('XAS')
        axs[0,1].legend()
        axs[0,1].grid()
        axs[0,1].set_title('C- Runs'+string_title)

        # Plot XMCD data
        energy0 = self.xmcd_ds['xmcd'].sel(variable='energy').values
        xmcd_data = self.xmcd_ds['xmcd'].sel(variable='xas_norm').values
        axs[1,0].plot(energy0, avg_runs_p, label="Avg C+", color='red')
        axs[1,0].plot(energy0, avg_runs_m, label="Avg C-", color='blue')
        axs[1,0].plot(energy0, xmcd_data*3, label="XMCD*3", color='black')
        axs[1,0].set_xlabel('Energy (eV)')
        axs[1,0].set_ylabel('Intensity (arb. units)')
        axs[1,0].set_title('XMCD'+string_title)
        axs[1,0].grid()
        axs[1,0].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_xas_to_hdf5(file_path_save, ds):
        """
        Save the XAS data to an HDF5 file.
        """
        if file_path_save is None or file_path_save=='':
            raise ValueError("file_path_save must be defined when save_xas_to_hdf5 is True.")
        if not isinstance(ds, xr.Dataset):
                raise ValueError("ds must be an xarray.Dataset.")
            
        ds.to_netcdf(file_path_save, engine='h5netcdf')
        print(f"Dataset saved to {file_path_save} successfully.")
               
class XAS_Spectra:
    def __init__(self, ds=None, filepath=None, file_list=None, order_by_parameter=None):
        """
        Initialize an XAS_Spectra object.

        Parameters
        ----------
        ds : xarray.Dataset, optional
            An xarray Dataset containing the XAS data.
        filepath : str, optional
            The file path to the XAS data.
        file_list : list of str, optional
            A list of file paths to the XAS data files.
        order_by_parameter : str, optional
            The parameter by which to order the dataset's DataArrays. 
            This should be an attribute present in the DataArrays.
        """

        if sum(x is not None for x in [ds, filepath, file_list]) > 1:
            raise ValueError("Only one of ds, filepath, or file_list should be provided.")
        
        if ds is not None:
            if not isinstance(ds, xr.Dataset):
                raise ValueError("data must be an xarray.Dataset")
            self.ds = ds
        elif filepath is not None:
            self.filepath = filepath
            self._load()
        elif file_list is not None:
            if not isinstance(file_list, list):
                raise ValueError("file_list must be a list of file paths.")
            if not all(isinstance(f, str) for f in file_list):
                raise ValueError("All elements in file_list must be strings representing file paths.")
            if not all(os.path.isfile(f) for f in file_list):
                for i, f in enumerate(file_list):
                    if not os.path.isfile(f):
                        print(f"Invalid file path for file #{i+1}: {f}")
                raise ValueError("All elements in file_list must be valid file paths.")
            if order_by_parameter is None:
                raise ValueError("order_by_parameter must be provided when file_list is used.")
                
            self._package_spectra(file_list, order_by_parameter)
            print("Packaging all the spectra into a single file.")
        else:
            raise ValueError("Either data or filepath or file_list must be provided.")

    def _load(self):
        """
        Load the HDF5 file into an xarray.Dataset.
        """
        if not hasattr(self, "filepath"):
            raise ValueError("No filepath specified for loading.")
        self.ds = xr.open_dataset(self.filepath, engine="h5netcdf")
        return self.ds

    def _package_spectra(self, filelist, order_by_parameter=None):
        """
        Package all the spectra into a single xarray.Dataset.
        Parameters
        ----------
        filelist : list of str
            List of file paths to the HDF5 files to be packaged.
        """
        if not isinstance(filelist, list):
            raise ValueError("filelist must be a list of file paths.")
        if not all(isinstance(f, str) for f in filelist):
            raise ValueError("All elements in filelist must be strings representing file paths.")
        if not all(os.path.isfile(f) for f in filelist):
            for i, f in enumerate(filelist):
                if not os.path.isfile(f):
                    print(f"Invalid file path for file #{i+1}: {f}")
            raise ValueError("All elements in filelist must be valid file paths.")
        # Load each file and concatenate them into a single xarray.Dataset
        self.ds = xr.Dataset()
        for ii, filepath in enumerate(filelist):
            ds_now = xr.open_dataset(filepath, engine="h5netcdf")
            if len(ds_now.data_vars) > 1:
                print(f"Warning: More than one xrArray found in {filepath}.")
            for var in ds_now.data_vars:
                self.ds[str(ii)] = ds_now[var].copy(deep=True)
        
        if order_by_parameter is not None:
            self._order_by_parameter(order_by_parameter)
            print(f"Ordered dataset by parameter: {order_by_parameter}")

    def _order_by_parameter(self, parameter):
        """
        Order the dataset's DataArrays by a specified attribute.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")

        # Collect (scan_name, attribute_value) pairs
        scan_attr_pairs = []
        for scan in self.ds.data_vars:
            attr_value = self.ds[scan].attrs.get(parameter)
            if attr_value is None:
                raise ValueError(f"Parameter '{parameter}' to order dataset not found in attributes of scan '{scan}'.")
            scan_attr_pairs.append((scan, attr_value))

        # Sort by attribute value
        scan_attr_pairs.sort(key=lambda x: x[1])

        # Rebuild the dataset in the new order
        new_ds = xr.Dataset()
        for scan, _ in scan_attr_pairs:
            new_ds[scan] = self.ds[scan].copy(deep=True)
        self.ds = new_ds

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
        
        self.ds.to_netcdf(filename, engine='h5netcdf')
        print(f"Dataset saved to {filename}")

    def save_to_csv(self, filename, motor_names, motor_name_mapping,
                                  metadata_in_csv=['theta','2theta','phi','energy','polarization',
                                                      'sample','B [T]', 'T [K]',],
                                    save_normalized=True):
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
        metadata_in_csv : list of str
            List of metadata attributes to include in the csv file header.
        normalize_spectra : bool
            If True, normalize the spectra by the normalization dataset.
        divide_normalization_by_value : float
            Value to divide the normalization dataset by when normalizing the spectra (e.g. 1E6 for mirror at ESRF)
        positive_energy_loss : bool
            If True, set the direction of energy loss to positive. If False, set it to negative.
        """

        # Ensure the filename ends with ".csv"
        if not filename.lower().endswith('.csv'):
            base, ext = os.path.splitext(os.path.basename(filename))
            if ext.lower() != '.csv':
                print(f"Warning: Changing file extension to .csv for {filename}")
                filename = os.path.join(os.path.dirname(filename), base + '.csv')
            filename += '.csv'

        print(f"Saving XAS spectra to csv.\n")
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
            for metadata in metadata_in_csv:
                header_parts = [metadata]
                header_parts_1 = [' ' for _ in range(len(self.ds.data_vars))]
                header_parts_2 = []   
                           
                if metadata in motor_name_mapping:
                    if motor_name_mapping[metadata] is not None:
                        header_parts_2 += [(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                            for i, value in enumerate(motor_values_all_scans[motor_name_mapping[metadata]])
                        ]
                else:
                    header_parts_2 += [" "] * (len(self.ds.data_vars))

                for idx in range(1, len(self.ds.data_vars) * 2 + 1):
                    if idx % 2 == 1:
                        header_parts.append(header_parts_1[(idx - 1) // 2])
                    else:
                        header_parts.append(header_parts_2[(idx - 1) // 2])

                header += ','.join(header_parts)  # Repeat each motor value twice
                header += '\n'

            # Write a line of "Energy Loss" and {scan} alternating
            header += ' ,'+','.join([f"Energy,{scan}" for scan in self.ds.data_vars])
            header += '\n'
            units = ' ,'+','.join(['(eV),(arb. units)'] * len(self.ds.data_vars))
            units += '\n'
            header += units
            file.write(f"{header}\n")

            # Create a DataFrame to store all scans
            data_frames = []
            # Stack all x and y values as adjacent columns
            all_data = []
            for scan in self.ds.data_vars:
                x_values = self.ds[scan].sel(variable='energy').values
                if save_normalized:
                    # Use the normalized xas values
                    y_values = self.ds[scan].sel(variable='xas_norm').values
                else:
                    y_values = self.ds[scan].sel(variable='xas').values

                all_data.append(np.column_stack((x_values, y_values)))

            # Concatenate all data along the second axis
            concatenated_data = np.concatenate(all_data, axis=1)

            # Write the concatenated data to the file
            for row in concatenated_data:
                file.write(' ,'+','.join(map(str, row)) + '\n')

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
        save_avg_spectrum : bool
            If True, save only the average spectrum as a single 'x' and 'y'
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        
        if save_avg_spectrum:
            # Sum all the RIXS spectra and save a single 'x' and 'y'
            sum_spectrum = np.sum([self.ds[scan].sel(variable='xas_norm').values for scan in self.ds.data_vars], axis=0)
            sum_norm = np.sum([self.ds[scan].sel(variable='norm').values for scan in self.ds.data_vars], axis=0)
            avg_spectrum = sum_spectrum / sum_norm
            x_values = self.ds[list(self.ds.data_vars)[0]].sel(variable='energy').values
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

    def print_attributes(self, avoid=None):
        """
        Print the attributes of the xarray.Dataset.
        Parameters
        ----------
        avoid : list of str, optional
            List of attribute names to avoid printing (e.g., 'filename', 'scan', 'run').
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        
        for scan in self.ds.data_vars:
            print(f"Scan: {scan}")
            for attr, value in self.ds[scan].attrs.items():
                if avoid is not None and attr in avoid:
                    continue
                print(f"  {attr}: {value}", end=", ")
            print("\n")

    def print_attributes_table(self,
                               attributes_to_include=None,
                               attributes_to_exclude=None):
        """
        Print a table of attributes for each DataArray in the dataset.
        Rows: attribute names (union of all attributes across DataArrays)
        Columns: DataArray names
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
    
        if attributes_to_include is not None and not isinstance(attributes_to_include, list):
            raise ValueError("attributes_to_include must be a list of attribute names.")
        if attributes_to_exclude is not None and not isinstance(attributes_to_exclude, list):
            raise ValueError("attributes_to_exclude must be a list of attribute names.")
        if attributes_to_include is not None and attributes_to_exclude is not None:
            raise ValueError("Cannot specify both attributes_to_include and attributes_to_exclude.")

        # Collect all attribute names
        attr_names = set()
        for var in self.ds.data_vars:
            attr_names.update(self.ds[var].attrs.keys())
        attr_names = sorted(attr_names)
        if attributes_to_include is not None:
            # Filter attributes to include only those specified
            attr_names = [attr for attr in attr_names if attr in attributes_to_include]
        elif attributes_to_exclude is not None:
            # Filter attributes to exclude those specified
            attr_names = [attr for attr in attr_names if attr not in attributes_to_exclude]


        # Build a dictionary for DataFrame construction
        data = {}
        for var in self.ds.data_vars:
            data[var] = {attr: self.ds[var].attrs.get(attr, "") for attr in attr_names}

        # Create DataFrame and display
        df = pd.DataFrame(data, index=attr_names)
        df = df.drop(index=["filename", "scan"], errors="ignore")
        display(df)



class ESRF_run_spectrum:

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
        print(f"\n\nFolder found: {folder}")
        self.runs = runs if isinstance(runs, list) else [runs]
        # Check for duplicates in runs
        if len(self.runs) != len(set(self.runs)):
            raise ValueError("Duplicate run numbers found in the provided runs.")
        if not isinstance(scans, list):
            self.scans = [scans]
        else:
            self.scans = scans
        self.energy_axis_calculated = False
    


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
            formatted_runs = [f"{int(run):04d}" for run in runs]

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
    

    def extract_1d_runs(self,
                        x_name, y_name, norm_name, motor_names,
                        plot=False,
                        scans_from_same_run=False):
        """
        Extract 1D runs from .spec files in the specified folder.

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
        xr.Dataset
            An xarray Dataset containing the extracted data for each run and scan.

        Returns
        -------
        list
            List of SpecFile objects for each found .spec file
        """
        # Search for .spec files corresponding to the provided runs
        spec_files = self._search_runs(self.runs)

        # Initialize SpecFile objects for each found .spec file
        spectra_xarray = xr.Dataset()
        for i, (file, run) in enumerate(zip(spec_files, self.runs)):
            specfile = SpecFile(file, run)
            extracted_data = specfile.extract_data(self.scans[i], x_name, y_name, norm_name, motor_names,
                                                   scans_from_same_run=scans_from_same_run)
            
            for scan_name, data_array in extracted_data.items():
                new_name = f"run_{run}_scan_{scan_name.split('_')[1]}"
                spectra_xarray[new_name] = data_array

        self.spectra_xarray = Generated_1D_RIXS_Spectra(ds=spectra_xarray)

        if plot:
            self.plot_spectra()

        return self.spectra_xarray



def main():
    print('ciao')   

  

if __name__ == "__main__":
    main()