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
        self.scans = scans
        self.energy_axis_calculated = False
    

    def extract_and_process_spectra(self,
                        x_name,
                        y_name,
                        norm_name, 
                        motor_names,
                        sample_name = '',
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
            self.pixel_row_start, self.pixel_row_stop = self._find_aligning_range(x_data,avg_spectrum, threshold=0.1)
        else:
            self.pixel_row_start = pixel_row_start
            self.pixel_row_stop = pixel_row_stop

        if np.shape(all_y_data)[0] == 1:
            align_spectra = False

        #save the pixel_row_start, pixel_row_stop and sample name inside the spectra_xarrays
        for spec_name in self.spectra_xarray.data_vars:
            self.spectra_xarray[spec_name].attrs['pixel_row_start'] = self.pixel_row_start
            self.spectra_xarray[spec_name].attrs['pixel_row_stop'] = self.pixel_row_stop
            self.spectra_xarray[spec_name].attrs['sample_name'] = sample_name if sample_name else 'unknown'

        if align_spectra:
            ### correct the energy shifts
            _ = self._correct_shift(all_y_data, 
                                    self.pixel_row_start, self.pixel_row_stop, 
                                    fit_shifts=fit_shifts, 
                                    smooth_shifts=smooth_shifts,
                                    correlation_batch_size=correlation_batch_size, 
                                    poly_order=poly_order)
        
            
        # _ = self.set_elastic_energy(elastic_line_point=elastic_line_point)
        
        if plot:
            self.plot_spectra(align_spectra, pixel_row_start=self.pixel_row_start, pixel_row_stop=self.pixel_row_stop)


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


    def save_to_hdf5(self, file_path_save, 
                     save_avg_spectrum=False):
        """
        Save the spectra to an HDF5 file.
        """
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_hdf5 is True.")
        if save_avg_spectrum:
            if not hasattr(self, 'avg_spectrum_xr_dataset') or self.avg_spectrum_xr_dataset is None:
                ds_avg = self.calculate_average_spectrum()
                RIXS_spectra(ds=ds_avg).save_to_hdf5(file_path_save)
            else:
                RIXS_spectra(ds=self.avg_spectrum_xr_dataset).save_to_hdf5(file_path_save)
            print("Only the average spectrum was saved to the HDF5 file.")
        else:
            RIXS_spectra(ds=self.spectra_xarray).save_to_hdf5(file_path_save)
            

    def save_to_csv_for_originlab(self, 
                    file_path_save,
                    motor_names, motor_name_mapping,
                    save_avg_spectrum=False):
        """
        Save the spectra to a CSV file.
        """
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_csv is True.")
        if save_avg_spectrum:
            if not hasattr(self, 'avg_spectrum_xr_dataset') or self.avg_spectrum_xr_dataset is None:
                ds_avg = self.calculate_average_spectrum()
                RIXS_spectra(ds=ds_avg).save_to_csv_for_originlab(file_path_save, motor_names, motor_name_mapping)
            else:
                RIXS_spectra(ds=self.avg_spectrum_xr_dataset).save_to_csv_for_originlab(file_path_save, motor_names, motor_name_mapping)

        else:
            RIXS_spectra(ds=self.spectra_xarray).save_to_csv_for_originlab(file_path_save, motor_names, motor_name_mapping)

    def save_to_txt(self, file_path_save, save_avg_spectrum=False):
        """
        Save the spectra to a text file.
        """ 
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_tx is True.")
        if save_avg_spectrum:
            ds_avg = self.calculate_average_spectrum()
            RIXS_spectra(ds=ds_avg).save_to_txt(file_path_save, save_avg_spectrum=save_avg_spectrum)
        else:
            RIXS_spectra(ds=self.spectra_xarray).save_to_txt(file_path_save, save_avg_spectrum=save_avg_spectrum)

    def calculate_average_spectrum(self):
        """
        Calculate the average spectrum from the extracted spectra.
        """
        if not hasattr(self, 'spectra_xarray'):
            raise ValueError("No spectra have been extracted. Please run extract_1d_runs first.")

        # Calculate the average spectrum
        runs = []
        scans = []
        for num_spectrum, (spec_name, _) in enumerate(self.spectra_xarray.items()):
            if num_spectrum == 0:
                avg_spectrum = self.spectra_xarray[spec_name].sel(variable='y').values.copy()
                x_axis_0 = self.spectra_xarray[spec_name].sel(variable='x').values.copy()
                norm = self.spectra_xarray[spec_name].sel(variable='norm').values.copy()
                x_name = self.spectra_xarray[spec_name].attrs.get('x_name', 'x')
                y_name = self.spectra_xarray[spec_name].attrs.get('y_name', 'y')
                norm_name = self.spectra_xarray[spec_name].attrs.get('norm_name', 'norm')
                # Extract only the folder path from the filename
                filename = os.path.dirname(self.spectra_xarray[spec_name].attrs.get('filename', '?'))
                date = self.spectra_xarray[spec_name].attrs.get('date', 'date unknown')
                runs.append(self.spectra_xarray[spec_name].attrs.get('run', '?'))
                scans.append(self.spectra_xarray[spec_name].attrs.get('scan', '?'))
                other_attrs = {key: value for key, value in self.spectra_xarray[spec_name].attrs.items() 
                               if key not in ["x_name", "y_name", "norm_name", "filename", "date", "run", "scan",
                                              "pixel_row_start", "pixel_row_stop"]}

            else:
                x_axis = self.spectra_xarray[spec_name].sel(variable='x').values
                spec_now = self.spectra_xarray[spec_name].sel(variable='y').values
                interp = np.interp(
                        x_axis_0, 
                        x_axis, 
                        spec_now,
                        left=0, right=0
                    )
            
                avg_spectrum += interp
                norm += self.spectra_xarray[spec_name].sel(variable='norm').values
                runs.append(self.spectra_xarray[spec_name].attrs.get('run', '?'))
                scans.append(self.spectra_xarray[spec_name].attrs.get('scan', '?'))
                for key, value in other_attrs.items():
                    if key in self.spectra_xarray[spec_name].attrs:
                        current_value = self.spectra_xarray[spec_name].attrs[key]
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

        # Create the DataArray with multiple coordinates for the 'points' dimension
        data = np.stack([x_axis_0, avg_spectrum, norm], axis=1)
        
        avg_spectrum_xr = xr.DataArray(
            data=data,
            dims=['points', 'variable'],
            coords={
            'points': np.arange(data.shape[0]),
            'variable': ['x', 'y', 'norm']
            }
        )
        avg_spectrum_xr.attrs['x_name'] = x_name
        avg_spectrum_xr.attrs['y_name'] = y_name
        avg_spectrum_xr.attrs['norm_name'] = norm_name
        avg_spectrum_xr.attrs['filename'] = filename
        avg_spectrum_xr.attrs['run'] = runs
        avg_spectrum_xr.attrs['scan'] = scans
        avg_spectrum_xr.attrs['date'] = date
        avg_spectrum_xr.attrs['pixel_row_start'] = self.pixel_row_start
        avg_spectrum_xr.attrs['pixel_row_stop'] = self.pixel_row_stop
        
        # Add other attributes
        for key, value in other_attrs.items():
            avg_spectrum_xr.attrs[key] = value

        self.avg_spectrum_xr_dataset = xr.Dataset()
        self.avg_spectrum_xr_dataset['avg_spectrum'] = avg_spectrum_xr.copy()
        del avg_spectrum_xr

        return self.avg_spectrum_xr_dataset.copy(deep=True)

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
                    spec_data.sel(variable='y'), label=spec_name, color=color_list[i])
        
        # Plot the average spectrum
        # sum_spectrum = np.sum([self.spectra_xarray[spec].sel(variable='y').values for spec in self.spectra_xarray.data_vars], axis=0)
        # sum_norm = np.sum([self.spectra_xarray[spec].sel(variable='norm').values for spec in self.spectra_xarray.data_vars], axis=0)
        # avg_spectrum = sum_spectrum / sum_norm
        # plt.plot(self.spectra_xarray[list(self.spectra_xarray.data_vars)[0]].sel(variable='x'), avg_spectrum, 
        #          label='Average Spectrum', linewidth=2, color='black')
        # Plot the average spectrum if it does not exist
        if not hasattr(self, 'avg_spectrum_xr_dataset') or self.avg_spectrum_xr_dataset is None:
            ds_avg = self.calculate_average_spectrum()
        else:
            ds_avg = self.avg_spectrum_xr_dataset.copy(deep=True)

        avg_spec = ds_avg['avg_spectrum']
        plt.plot(avg_spec.sel(variable='x'), avg_spec.sel(variable='y')/len(self.spectra_xarray.data_vars), 
                 label='Average Spectrum', linewidth=2, color='black')
        
        # Plot vertical dashed lines for pixel_row_start and pixel_row_stop
        if pixel_row_start is not None and pixel_row_stop is not None:
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_start], color='k', linestyle='--', label='Start Pixel Row')
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_stop], color='k', linestyle='--', label='Stop Pixel Row')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity')
        # plt.legend()
        plt.title('Extracted Spectra')
        plt.grid()
        # plt.show()

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
            # plt.show()


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
    
    def set_elastic_energy(self, auto_elastic_determination=False, elastic_line_point=None, calibration=1):
        """
        Set the elastic line energy point and apply calibration to the spectra.
        Parameters
        ----------
        elastic_line_point : float
            The energy point of the elastic line.
        calibration : float
            The calibration factor to be applied to the spectra (in eV/pixel)
        """
        if self.energy_axis_calculated:
            raise ValueError("Energy axis has already been calculated. Skipping...")
        
        else:
            print("-> Setting elastic line energy point and applying calibration...")
            if elastic_line_point is None and not auto_elastic_determination:
                raise ValueError("elastic_line_point is not defined and autodetermination is off. Please set it before calling this method.")
            
            if auto_elastic_determination:
                # Automatically determine the elastic line point
                if self.pixel_row_start is None or self.pixel_row_stop is None:
                    raise ValueError("pixel_row_start and pixel_row_stop must be defined for automatic determination.")
                # Find the maximum point in the specified interval
                self.calculate_average_spectrum()
                avg_spectrum = self.avg_spectrum_xr_dataset['avg_spectrum'].sel(variable='y').values.copy()
                x_data = self.avg_spectrum_xr_dataset['avg_spectrum'].sel(variable='x').values.copy()
                max_index = np.argmax(avg_spectrum[self.pixel_row_start:self.pixel_row_stop]) + self.pixel_row_start
                
                # Define a range around the maximum point for center of mass calculation
                neighbor_range = 1  # Adjust this value as needed
                start_index = max(max_index - neighbor_range, self.pixel_row_start)
                end_index = min(max_index + neighbor_range + 1, self.pixel_row_stop)
                
                # Calculate the center of mass
                weights = avg_spectrum[start_index:end_index]
                indices = np.arange(start_index, end_index)
                elastic_line_point = np.sum(x_data[indices] * weights) / np.sum(weights)
                del avg_spectrum


            for spec_name in self.spectra_xarray.data_vars:
                self.spectra_xarray[spec_name].loc[dict(variable='x')] -= elastic_line_point
                self.spectra_xarray[spec_name].loc[dict(variable='x')] *= calibration
            
            self.energy_axis_calculated = True
            self.calculate_average_spectrum()

            print(f"Elastic line energy point set to {elastic_line_point}, calibration factor {calibration} eV/pixel.")


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

        # Round all the shifts to the nearest integer
        # self.shifts = np.round(self.shifts)        
         
        # Save the real shifts somewhere for plotting
        for i in range(0, real_shifts.shape[0]):
            self.real_shifts[i * correlation_batch_size:(i + 1) * correlation_batch_size] = real_shifts[i]

        # for num_spectrum, (spec_name, _) in enumerate(self.spectra_xarray.items()):
        #     if abs(self.shifts[num_spectrum]) > 0.4:
        #         print(f"{self.shifts[num_spectrum]:.2f}, ", end="")
                
        #         # Interpolate
        #         interp = np.interp(
        #             np.arange(spectra.shape[1]) - self.shifts[num_spectrum], 
        #             np.arange(spectra.shape[1]), 
        #             spectra[num_spectrum, :],
        #             left=0, right=0
        #         )
                
        #         self.spectra_xarray[spec_name].loc[dict(variable='y')] = interp.copy()
        #     else:
        #         self.shifts[num_spectrum] = 0
        #         print("0.00, ", end="")

        # for num_spectrum, (spec_name, _) in enumerate(self.spectra_xarray.items()):
        #     # Approximate self.shifts[num_spectrum] to the closest half-integer
        #     self.shifts[num_spectrum] = round(self.shifts[num_spectrum] * 2) / 2

        #     if abs(self.shifts[num_spectrum]) > 0.1:
        #         # Interpolate
        #         interp = np.interp(
        #             np.arange(spectra.shape[1]) - self.shifts[num_spectrum], 
        #             np.arange(spectra.shape[1]), 
        #             spectra[num_spectrum, :],
        #             left=0, right=0
        #         )
            
        #         self.spectra_xarray[spec_name].loc[dict(variable='y')] = interp.copy()

        for num_spectrum, (spec_name, _) in enumerate(self.spectra_xarray.items()):
            # Approximate self.shifts[num_spectrum] to the closest half-integer            
                self.spectra_xarray[spec_name].loc[dict(variable='x')] -= self.shifts[num_spectrum]
                print(f"{self.shifts[num_spectrum]:.2f}, ", end="")


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

        if factor == 1:
            # Skip interpolation if factor is 1
            xx = np.arange(pixel_start, pixel_stop + 1)
            spec = spec[xx]  # Use the original spectrum directly
            spec_ref = spec_ref[xx]  # Use the original reference spectrum directly
        else:
            xx = np.arange(pixel_start, pixel_stop + 1 / factor, 1 / factor)
            spec = np.interp(xx, np.arange(0, len(spec)), spec)  # Interpolating the spectrum
            spec_ref = np.interp(xx, np.arange(0, len(spec_ref)), spec_ref)  # Interpolating the reference spectrum

        crosscorr = correlate(spec/np.mean(spec), spec_ref/np.mean(spec_ref))
        lag_values = np.arange(-spec.shape[0] + 1, spec.shape[0], 1)
        lag = lag_values[np.argmax(crosscorr)] / factor

        # if factor > 1:
        #     # xx = np.arange(0, len(spec_ref) + 1 / factor, 1 / factor)
        #     xx = np.linspace(0, len(spec_ref)-1, (len(spec_ref) - 1) * factor + 1)
        #     spec = np.interp(xx, np.arange(0, len(spec)), spec)  # Interpolating the spectrum
        #     spec_ref = np.interp(xx, np.arange(0, len(spec_ref)), spec_ref)  # Interpolating the reference spectrum

        # lag_values, crosscorr = self._custom_cross_correlation(spec, spec_ref, int(pixel_start*factor), int(pixel_stop*factor))
        # crosscorr_restricted = crosscorr[(lag_values >= -10) & (lag_values <= 10)]
        # first_index = np.argmax(lag_values >= -10)
        # lag = (lag_values[np.argmax(crosscorr_restricted)] + first_index) / factor

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
            ds[f'scan_{scan}'].attrs['scan'] = scan
        
        
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
                    if 'hu70cp' in motor_names and 'hu70ap' in motor_names:
                        normalized_data[scan].attrs['polarization'] = self._determine_polarization(
                            normalized_data[scan].attrs['hu70ap'], 
                            normalized_data[scan].attrs['hu70cp']
                        )

                normalized_data[scan].attrs['date'] = ds.attrs['date']
                normalized_data[scan].attrs['run'] = self.run
                normalized_data[scan].attrs['scan'] = ds[scan].attrs['scan']
                normalized_data[scan].attrs['filename'] = self.filename
        
        return normalized_data
    
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


class RIXS_spectra:
    def __init__(self, ds=None, filepath=None, file_list=None,
                 order_by_parameter=None):
        """
        Initialize RIXS_spectra with either an xarray.Dataset or a file path to an HDF5 file.

        Parameters
        ----------
        data : xarray.Dataset, optional
            The dataset to use directly.
        filepath : str, optional
            Path to the HDF5 file to load.
        """
        self.ds = None
        if ds is not None:
            if not isinstance(ds, xr.Dataset):
                raise ValueError("data must be an xarray.Dataset")
            self.ds = ds
        elif filepath is not None:
            self.filepath = filepath
            self._load()
        elif file_list is not None:
            self._package_spectra(file_list, order_by_parameter)
            print("Packaging all the spectra into a single file.")
        else:
            raise ValueError("Either data or filepath must be provided.")

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

    def align_spectra(self, method, **kwargs):
        """
        Align the spectra in the dataset using the specified method.
        Parameters
        ----------
        method : str
            The alignment method to use. Options are 'cross-correlation' or 'fitting'.
        **kwargs : dict
            Additional parameters for the alignment method, such as 'resolution', 'fit_function', and 'plot'.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        if method == 'cross-correlation':
            # Implement correlation-based alignment
            self._align_spectra_cross_correlation()
        elif method == 'fitting':
            # fitting alignment
            resolution = kwargs.get('resolution', 0.1)
            fit_function = kwargs.get('fit_function', 'gaussian')
            plot = kwargs.get('plot', False)
            self._align_spectra_fitting(resolution=resolution, fit_function=fit_function, plot=plot) 
        else:
            raise ValueError(f"Unknown alignment method: {method}")
    
    def _align_spectra_cross_correlation(self):
        """
        Align the spectra in the dataset using cross-correlation.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        # Implement cross-correlation alignment logic here
        pass

    def _align_spectra_fitting(self, resolution=0.1, fit_function="gaussian",
                               plot=False):
        """
        Align the spectra in the dataset using fitting of elastic line.
        Parameters
        ----------
        resolution : float
            The resolution of the fitting function in eV.
        fit_function : str
            The type of fitting function to use. Options are 'gaussian' or 'pseudovoigt'.
        plot : bool
            If True, plot the fitting results.
        Raises
        ------
        ValueError
            If the dataset is not provided.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        
        # Define normalization limits and fit limits
        fit_min = -resolution * 3
        fit_max = resolution/2.5

        # Define gaussian and pseudovoigt functions
        def gaussian(x, amp, cen, fwhm):
            sigma = fwhm/(2*np.sqrt(2*np.log(2)))
            return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)

        def pseudovoigt(x, amp, cen, fwhm, eta):
            # eta: mixing parameter (0=Gaussian, 1=Lorentzian)
            gaussian_part = (1 - eta) * amp * np.exp(-4 * np.log(2) * ((x - cen) / fwhm) ** 2)
            lorentzian_part = eta * amp / (1 + 4 * ((x - cen) / fwhm) ** 2)
            return gaussian_part + lorentzian_part
        
        if fit_function=="gaussian":
            p0 = [1.0, 0.0]
            bounds = ([0, -np.inf], [np.inf, np.inf])
            def fitting_func(x, amp, cen):
                return gaussian(x, amp, cen, resolution)
        elif fit_function=="pseudovoigt":
            p0 = [1.0, 0.0, 0.2]
            bounds = ([0, -np.inf, 0], [np.inf, np.inf, 1])
            def fitting_func(x, amp, cen, eta):
                return pseudovoigt(x, amp, cen, resolution, eta)
        else:
            raise ValueError(f"Unknown fit function: {fit_function}")

        # Implement fitting alignment logic here
        # For each xArray in self.ds, extract x_values and y_values
        for scan in self.ds.data_vars:
            x_values = self.ds[scan].sel(variable='x').values
            y_values = self.ds[scan].sel(variable='y').values
            norm_values = self.ds[scan].sel(variable='norm').values
            x_values, y_values, norm_values, direction_changed = self._set_direction_energy_loss(x_values, y_values,
                                                                              norm_values=norm_values,
                                                                               positive_energy_loss=True)

            # Find indices within the normalization range
            norm_indices = np.where((x_values >= fit_min) & (x_values <= fit_max))[0]
            if len(norm_indices) == 0:
                raise ValueError(f"No data points found in normalization range for scan {scan}.")

            # Normalize y_values to the max value within the range
            max_val = np.max(y_values[norm_indices])
            if max_val == 0:
                normed_y = y_values
            else:
                normed_y = y_values / max_val

            # Fit the data using the specified fitting function
            # Fit only in the normalization range
            x_fit = x_values[norm_indices]
            y_fit = normed_y[norm_indices]

            # Perform the fit using curve_fit
            try:
                popt, pcov = curve_fit(fitting_func, x_fit, y_fit, p0=p0, bounds=bounds)
                initialfit = fitting_func(x_values, *popt)
            except Exception as e:
                print(f"Fit failed for scan {scan}: {e}")
                popt = None
                initialfit = np.full_like(x_values, np.nan)

            shift = popt[1]  # center position from fit
            x_values_shifted = x_values - shift
            # Update the x_values in the dataset
            self.ds[scan].loc[dict(variable='x')] = x_values_shifted if not direction_changed else -x_values_shifted[::-1]
            self.ds[scan].loc[dict(variable='y')] = y_values if not direction_changed else y_values[::-1]
            self.ds[scan].loc[dict(variable='norm')] = norm_values if not direction_changed else norm_values[::-1]

            if plot:
                plt.figure(figsize=(4, 4))
                plt.plot(x_values, normed_y, label='Original Data')
                plt.plot(x_fit, y_fit, 'o', label='Data for Fit')
                plt.plot(x_values, initialfit, label='Fitted Curve', color='red')
                plt.title(f"Scan {scan} - Shift: {shift:.2f}")
                plt.xlim(-resolution*3, resolution*3)
                plt.xlabel('Energy Loss (eV)')
                plt.ylabel('Intensity (arb. units)')
                plt.legend()
                plt.show()

    def set_direction_energy_loss_dataset(self, positive_energy_loss=True):
        """
        Set the direction of energy loss for the dataset based on the specified condition.

        Parameters
        ----------
        positive_energy_loss : bool
            If True, set the direction to positive energy loss.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")
        
        for scan in self.ds.data_vars:
            x_values = self.ds[scan].sel(variable='x').values
            y_values = self.ds[scan].sel(variable='y').values
            norm_values = self.ds[scan].sel(variable='norm').values
            x_values, y_values, norm_values, direction_changed = self._set_direction_energy_loss(
                x_values, y_values, norm_values=norm_values, positive_energy_loss=positive_energy_loss)
            
            # Update the xarray with the new values
            self.ds[scan].loc[dict(variable='x')] = x_values
            self.ds[scan].loc[dict(variable='y')] = y_values
            self.ds[scan].loc[dict(variable='norm')] = norm_values
        

    @staticmethod
    def _set_direction_energy_loss(x_values, y_values, norm_values = None, positive_energy_loss=True):
        # Calculate the sum of y_values for x_values < 0 and x_values > 0
        sum_left = np.sum(y_values[x_values < 0])
        sum_right = np.sum(y_values[x_values > 0])
        if sum_left > sum_right and positive_energy_loss:
            #more intensity at negative energy losses
            # Multiply x_values by -1 and flip all arrays
            x_values = -x_values
            idx = np.argsort(x_values)
            x_values = x_values[idx]
            y_values = y_values[idx]
            norm_values = norm_values[idx] if norm_values is not None else None
            direction_changed = True
        elif sum_left < sum_right and not positive_energy_loss:
            #more intensity at positive energy losses
            # Multiply x_values by -1 and flip all arrays
            x_values = -x_values
            idx = np.argsort(x_values)
            x_values = x_values[idx]
            y_values = y_values[idx]
            norm_values = norm_values[idx] if norm_values is not None else None
            direction_changed = True
        else:
            #no need to change direction
            direction_changed = False

        return x_values, y_values, norm_values, direction_changed
    
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
                                                      'mirror','sample','B [T]', 'T [K]',],
                                    normalize_spectra=False,
                                    divide_normalization_by_value=1,
                                    positive_energy_loss=True,):
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
        metadata_in_origin : list of str
            List of metadata attributes to include in the OriginLab file.
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

        print(f"Saving dataset- Spectra will be normalized by {self.ds['0'].attrs['norm_name']}, divided by {divide_normalization_by_value}.\n")
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

            #### old version
            # header = ''
            # # Write motor values in the header
            # for metadata in metadata_in_origin:
            #     header_parts = [metadata]
            #     if metadata == 'mirror':
            #         header_parts += [f"{np.mean(self.ds[scan].sel(variable='norm').values):.2f}" for scan in self.ds.data_vars]
            #     else:                
            #         if metadata in motor_name_mapping:
            #             if motor_name_mapping[metadata] is not None:
            #                 header_parts += [
            #                     f"{value:.2f}" if isinstance(value, (int, float)) else str(value) 
            #                     for value in motor_values_all_scans[motor_name_mapping[metadata]]
            #                 ]
            #         else:
            #             header_parts += [" "] * len(motor_values_all_scans[motor_names[0]])
            #     header += ','.join(header_parts * 2)  # Repeat each motor value twice
            #     header += '\n'

            header = ''
            # Write motor values in the header
            for metadata in metadata_in_origin:
                header_parts = [metadata]
                header_parts_1 = [' ' for _ in range(len(self.ds.data_vars))]
                header_parts_2 = []
                if metadata == 'mirror':
                        if normalize_spectra:
                            header_parts_2 += [f"{np.mean(self.ds[scan].sel(variable='norm').values)/divide_normalization_by_value:.2f}" for i, scan in enumerate(self.ds.data_vars)]
                        else:
                            header_parts_2 += [f"{np.mean(self.ds[scan].sel(variable='norm').values):.2f}" for i, scan in enumerate(self.ds.data_vars)]
                else:                
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
            header += ' ,'+','.join([f"Energy Loss,{scan}" for scan in self.ds.data_vars])
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
                x_values = self.ds[scan].sel(variable='x').values
                y_values = self.ds[scan].sel(variable='y').values
                norm_values = self.ds[scan].sel(variable='norm').values
                if normalize_spectra:
                    # Normalize the y-values by the norm values
                    y_values = y_values / norm_values * divide_normalization_by_value

                # Calculate the sum of y_values for x_values < 0 and x_values > 0
                x_values, y_values, norm_values,_ = self._set_direction_energy_loss(x_values, y_values,
                                                                                  norm_values=norm_values,
                                                                                  positive_energy_loss=positive_energy_loss)
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

    def print_attributes_table(self):
        """
        Print a table of attributes for each DataArray in the dataset.
        Rows: attribute names (union of all attributes across DataArrays)
        Columns: DataArray names
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")

        # Collect all attribute names
        attr_names = set()
        for var in self.ds.data_vars:
            attr_names.update(self.ds[var].attrs.keys())
        attr_names = sorted(attr_names)
        attr_names = [attr for attr in attr_names if attr not in ("filename", "scan", "run")]

        # Build a dictionary for DataFrame construction
        data = {}
        for var in self.ds.data_vars:
            data[var] = {attr: self.ds[var].attrs.get(attr, "") for attr in attr_names}

        # Create DataFrame and display
        df = pd.DataFrame(data, index=attr_names)
        df = df.drop(index=["filename", "scan"], errors="ignore")
        display(df)

    def get_map_with_parameter(self, parameter):
        parameter_list = []
        y = []
        x = []

        for da_name in self.ds.data_vars:
            da = self.ds[da_name]
            # Extract the "energy" attribute
            par = da.attrs.get(parameter)
            parameter_list.append(par)

            # Save the RIXS spectra and corresponding x-axis
            x_values = self.ds[da_name].sel(variable='x').values
            y_values = self.ds[da_name].sel(variable='y').values
            norm_values = self.ds[da_name].sel(variable='norm').values / 1E6
            y.append(y_values/ norm_values)  # Normalize the y values
            x.append(x_values)


        par_arr = np.array(parameter_list)
        # Interpolate all y_values onto the first x_values grid
        x_arr = np.array(x[0])  # use the first x_values as the reference grid
        y_interp = []
        for i in range(len(y)):
            y_interp.append(np.interp(x_arr, x[i], y[i]))

        intensity = np.array(y_interp).T  # shape: (len(x_arr), len(energy_list))

        return x_arr, par_arr, intensity
        


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
    


def main():
    print('ciao')   

  

if __name__ == "__main__":
    main()