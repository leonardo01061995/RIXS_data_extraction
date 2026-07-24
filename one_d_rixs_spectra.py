import os
import re
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from cmcrameri import cm
from IPython.display import display
from static_functions import calculate_shift_new, calculate_shift_mccc,  _find_aligning_range, check_variations_parameters



class Generated_1D_RIXS_Spectra:
    def __init__(self, ds, energy_axis_calculated=False):
        """
        Class to handle 1D RIXS spectra generated from 2D RIXS images.
        To be used with xarray datasets generated either by extract_rixs_spectra.py or by generate_rixs_spectra.py
        Parameters
        ----------
        ds : xarray.Dataset
            The xarray Dataset containing the 1D RIXS spectra.
        """
        self.spectra_xarray = ds
        self.energy_axis_calculated = energy_axis_calculated
        check_variations_parameters(self.spectra_xarray,
                        attributes_to_exclude=["x_name", "y_name", "norm_name", 
                                            "filename", "date", "run", "scan",
                                            "pixel_row_start", "pixel_row_stop", "mirror"], 
                                threshold=0.1)


    def align_spectra(self,
                        aligning_range = None,
                        shift_postprocess = 'none',
                        correlation_batch_size=10,
                        poly_order=1,
                        plot=False,):
        """
        Process the extracted spectra to align energy shifts and calculate average spectrum.
        Parameters
        ----------
        aligning_range : tuple
            A tuple specifying the pixel row range for alignment (start, stop).
        shift_postprocess : str
            Specifies the method for post-processing shifts. Options are 'fit', 'smooth', or 'interp'.
        correlation_batch_size : int
            Number of spectra to average for correlation calculation.
        poly_order : int
            Order of the polynomial for fitting shifts if shift_postprocess is 'fit'.
        plot : bool
            Whether to plot the processed spectra.
        """

        if shift_postprocess not in ['fit', 'smooth', 'interp', 'none']:
            raise ValueError("Invalid shift_postprocess method. Choose from 'fit', 'smooth', 'interp', or 'none'.")

        # Find aligning range using the _find_aligning_range method
        print("\n-> Aligning spectra using cross-correlation.")
        if aligning_range is None:
            # Extract the first x-axis, stack all y_data arrays along a new dimension
            x_data = self.spectra_xarray[list(self.spectra_xarray.data_vars)[0]].sel(variable='x').values
            all_y_data = np.stack([self.spectra_xarray[spec].sel(variable='y').values for spec in self.spectra_xarray.data_vars], axis=0)
            avg_spectrum = np.mean(all_y_data, axis=0)

            #check if there are multiple runs. if so, enlarge the aligning range
            runs = []
            for spec_name in self.spectra_xarray.data_vars:
                run = self.spectra_xarray[spec_name].attrs.get('run', '?')
                runs.append(run)

            multiple_runs = False
            if '?' in runs:
                multiple_runs = True
            else:
                multiple_runs = len(set(runs)) > 1

            self.pixel_row_start, self.pixel_row_stop = _find_aligning_range(avg_spectrum, x_data=x_data,
                                                                             threshold=0.1, extended_range=multiple_runs)
        else:
            self.pixel_row_start = aligning_range[0]
            self.pixel_row_stop = aligning_range[1]

        # Stack all y_data arrays along a new dimension
        all_y_data = np.stack([self.spectra_xarray[spec].sel(variable='y').values for spec in self.spectra_xarray.data_vars], axis=0)
        if all_y_data.shape[0] == 1:
            print("\tOnly one spectrum found. Skipping alignment.")
            return

        #save the pixel_row_start, pixel_row_stop and sample name inside the spectra_xarrays
        for spec_name in self.spectra_xarray.data_vars:
            self.spectra_xarray[spec_name].attrs['alignment_range'] = np.array([self.pixel_row_start, self.pixel_row_stop])

        # ── Calculate shifts ──────────────────────────
        # self.shifts, self.real_shifts, self.real_shifts_batches_1round = calculate_shift_new(all_y_data,
        #                                                                                      aligning_range=(self.pixel_row_start, self.pixel_row_stop),
        #                                                                                      fit_shifts=fit_shifts,
        #                                                                                      smooth_shifts=smooth_shifts,
        #                                                                                      interp_shifts=interp_shifts,
        #                                                                                      correlation_batch_size=correlation_batch_size,
        #                                                                                      poly_order=poly_order)      
        self.shifts, self.real_shifts, self.real_shifts_batches_1round = calculate_shift_mccc(all_y_data,
                                                                                             aligning_range=(self.pixel_row_start, self.pixel_row_stop),
                                                                                             shift_postprocess=shift_postprocess,
                                                                                             correlation_batch_size=correlation_batch_size,
                                                                                             poly_order=poly_order)  

        #shifts, real_shifts and real_shifts_batches1 are in sub-pixels (points), but the x-axis
        #is in pixels
        spec_name = list(self.spectra_xarray.data_vars)[0]
        x_axis = self.spectra_xarray[spec_name].sel(variable='x').values
        subpixel_factor = x_axis[1] - x_axis[0]
        self.shifts, self.real_shifts, self.real_shifts_batches_1round = self.shifts * subpixel_factor, self.real_shifts * subpixel_factor, self.real_shifts_batches_1round * subpixel_factor


        # ── Correct shift ─────────────────────────────
        for num_spectrum, (spec_name, _) in enumerate(self.spectra_xarray.items()):          
                self.spectra_xarray[spec_name].loc[dict(variable='x')] -= self.shifts[num_spectrum]
                # print(f"{self.shifts[num_spectrum]:.2f}, ", end="")
        
        self.spectra_xarray.attrs['log'] += f"\nSpectra aligned using cross-correlation with aligning range: {self.pixel_row_start}-{self.pixel_row_stop} \
                                            , grouped in batches of size {correlation_batch_size}."
        if shift_postprocess == 'smooth':
            self.spectra_xarray.attrs['log'] += f"\nShifts smoothed using Gaussian filter with sigma = {correlation_batch_size}"
        elif shift_postprocess == 'interp':
            self.spectra_xarray.attrs['log'] += f"\nShifts interpolated using linear interpolation."
        elif shift_postprocess == 'fit':
            self.spectra_xarray.attrs['log'] += f"\nShifts fitted using polynomial of order {poly_order}."
        else:
            self.spectra_xarray.attrs['log'] += f"\nShifts applied without smoothing, interpolation, or fitting."
        
                   
        if plot:
            self.plot_spectra(True, pixel_row_start=self.pixel_row_start, pixel_row_stop=self.pixel_row_stop,
                              correlation_batch_size=correlation_batch_size)

            

    def save_to_hdf5(self, file_path_save, 
                     save_only_avg_spectrum=False,
                     normalize_spectra=False,
                     divide_normalization_by_value=1,
                     variable_names=None,
                     additional_metadata=None,
                     metadata_to_save=None):
        """
        Save the spectra to an HDF5 file.
        """

        if additional_metadata is None:
            additional_metadata = {}
        if metadata_to_save is None:
            metadata_to_save = []

        args_to_pass = dict(
            filename=file_path_save,
            normalize_spectra=normalize_spectra,
            divide_normalization_by_value=divide_normalization_by_value,
            variable_names=variable_names,
            additional_metadata=additional_metadata,
            metadata_to_save=metadata_to_save
        )
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_hdf5 is True.")
        if save_only_avg_spectrum:
            if not hasattr(self, 'avg_spectrum_xr_dataset') or self.avg_spectrum_xr_dataset is None:
                ds_avg = self.calculate_average_spectrum()
                RIXS_Spectra(ds=ds_avg).save_to_hdf5(**args_to_pass)
            else:
                RIXS_Spectra(ds=self.avg_spectrum_xr_dataset).save_to_hdf5(**args_to_pass)
            print("\tOnly the average spectrum was saved to the HDF5 file.")
        else:
            RIXS_Spectra(ds=self.spectra_xarray).save_to_hdf5(**args_to_pass)
            

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
                RIXS_Spectra(ds=ds_avg).save_to_csv_for_originlab(file_path_save, motor_names, motor_name_mapping)
            else:
                RIXS_Spectra(ds=self.avg_spectrum_xr_dataset).save_to_csv_for_originlab(file_path_save, motor_names, motor_name_mapping)

        else:
            RIXS_Spectra(ds=self.spectra_xarray).save_to_csv_for_originlab(file_path_save, motor_names, motor_name_mapping)

    def save_to_txt(self, file_path_save, save_avg_spectrum=False):
        """
        Save the spectra to a text file.
        """ 
        if file_path_save is None:
            raise ValueError("file_path_save must be defined when save_to_txt is True.")
        if save_avg_spectrum:
            ds_avg = self.calculate_average_spectrum()
            RIXS_Spectra(ds=ds_avg).save_to_txt(file_path_save, save_avg_spectrum=save_avg_spectrum)
        else:
            RIXS_Spectra(ds=self.spectra_xarray).save_to_txt(file_path_save, save_avg_spectrum=save_avg_spectrum)

    def calculate_average_spectrum(self, poisson_error=True):
        """
        Calculate the average spectrum from the extracted spectra.
        """
        if not hasattr(self, 'spectra_xarray'):
            raise ValueError("No spectra have been extracted. Please run extract_1d_runs first.")

        # Calculate the average spectrum
        runs = []
        scans = []
        # all_y_values = []
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
                                              "pixel_row_start", "pixel_row_stop","alignment_range", "mirror"]}
                
                # all_y_values = avg_spectrum.reshape(1, -1)  # Initialize all_y_values with the first spectrum's y-values
                all_y_values = [avg_spectrum.copy()]  # Initialize all_y_values with the first spectrum's y-values

            else:
                x_axis = self.spectra_xarray[spec_name].sel(variable='x').values
                spec_now = self.spectra_xarray[spec_name].sel(variable='y').values
                interp = np.interp(
                        x_axis_0, 
                        x_axis, 
                        spec_now,
                        left=0, right=0
                    )
                # all_y_values = np.vstack((all_y_values, interp))  # Append the interpolated y-values to all_y_values
                all_y_values.append(interp)

                avg_spectrum += interp
                norm += self.spectra_xarray[spec_name].sel(variable='norm').values
                runs.append(self.spectra_xarray[spec_name].attrs.get('run', '?'))
                scans.append(self.spectra_xarray[spec_name].attrs.get('scan', '?'))

        all_y_values = np.stack(all_y_values, axis=0)  # Convert the list of arrays to a 2D array

        if poisson_error:
            print("\tCalculating Poisson error for the average spectrum.")
            avg_spectrum_clamped = np.clip(avg_spectrum, a_min=0, a_max=None)  # Ensure no negative values for Poisson error calculation
            error = np.sqrt(avg_spectrum_clamped)
        else:
            print("\tCalculating error for the average spectrum as standard deviation.")
            error = np.std(all_y_values, axis=0)*np.sqrt(len(all_y_values))

        # Create the DataArray with multiple coordinates for the 'points' dimension
        data = np.stack([x_axis_0, avg_spectrum, error, norm], axis=1)
        
        avg_spectrum_xr = xr.DataArray(
            data=data,
            dims=['points', 'variable'],
            coords={
            'points': np.arange(data.shape[0]),
            'variable': ['x', 'y', 'error', 'norm']
            }
        )
        avg_spectrum_xr.attrs['log'] = self.spectra_xarray.attrs.get('log', '') + "\nAverage spectrum calculated from the extracted spectra."
        avg_spectrum_xr.attrs['mirror'] = np.mean(norm)
        avg_spectrum_xr.attrs['x_name'] = x_name
        avg_spectrum_xr.attrs['y_name'] = y_name
        avg_spectrum_xr.attrs['norm_name'] = norm_name
        avg_spectrum_xr.attrs['filename'] = filename
        avg_spectrum_xr.attrs['run'] = '_'.join(dict.fromkeys(runs))
        avg_spectrum_xr.attrs['scan'] = scans
        avg_spectrum_xr.attrs['date'] = date
        avg_spectrum_xr.attrs['alignment_range'] = np.array([self.pixel_row_start, self.pixel_row_stop])
        
        # Add other attributes
        for key, value in other_attrs.items():
            avg_spectrum_xr.attrs[key] = value

        
        self.avg_spectrum_xr_dataset = xr.Dataset()
        self.avg_spectrum_xr_dataset['avg_spectrum'] = avg_spectrum_xr.copy()
        del avg_spectrum_xr

        return self.avg_spectrum_xr_dataset.copy(deep=True)

    def plot_spectra(self, align_spectra=False, pixel_row_start=None, pixel_row_stop=None, 
                     correlation_batch_size=1):
        """
        Plot the extracted spectra.
        Spectra are grouped into consecutive batches of size `correlation_batch_size`;
        the mean spectrum of each batch is displayed (up to 5 batches, evenly spaced
        across the full spectra list).
        """
        if not hasattr(self, 'spectra_xarray'):
            raise ValueError("No spectra have been extracted. Please run extract_1d_runs or process_spectra first.")
        
        #shifts
        if align_spectra:
            plt.figure(figsize=(11,7))
            plt.subplot(2,2,1)
            plt.plot(self.real_shifts_batches_1round, 'ko-', label='Shifts (1 round)')  # 'g^-' for green triangles connected by lines
            plt.plot(self.real_shifts, 'ro-', label='Shifts (2 round)')  # 'ko-' for black circles connected by lines
            plt.plot(self.shifts, 'o-', color='orange',  label='Used Shifts')
            plt.xlabel('Image Index')
            plt.ylabel('Shift Value')
            plt.title('Real Shifts of Images')
            plt.grid()
            plt.legend()

            # Calculate integrals between pixel_row_start and pixel_row_stop for each spectrum
            integrals = []
            i0 = int(self.pixel_row_start)
            i1 = int(self.pixel_row_stop)
            for spec_name in self.spectra_xarray.data_vars:
                da = self.spectra_xarray[spec_name]
                y_vals = da.sel(variable='y').values
                integrals.append(np.sum(y_vals[i0:i1+1]))

            # Plot integrals in the second subplot
            plt.subplot(2, 2, 2)
            plt.plot(integrals, 'bo-', label='Integrated intensity')
            plt.axhline(np.mean(integrals), color='k', linestyle='--', label='Mean')
            plt.xlabel('Image Index')
            plt.ylabel('Integrated intensity (arb. units)')
            plt.title(f'Integral between pixels {i0} and {i1}')
            plt.grid()
            plt.legend()

            # plt.tight_layout()
            # plt.show()

        all_items = list(self.spectra_xarray.items())
        n_total = len(all_items)
        batch_size = max(1, correlation_batch_size)

        # Group spectra into consecutive batches of size `batch_size`
        batch_slices = [slice(start, min(start + batch_size, n_total)) for start in range(0, n_total, batch_size)]

        # Pick up to 5 batches, evenly spaced across the full list of batches
        max_batches = 5
        batches_to_plot = batch_slices[::max(1, len(batch_slices) // max_batches)]
        color_list = [cm.managua(i) for i in np.linspace(0, 1, len(batches_to_plot))]

        def _batch_mean(x_arrays, y_arrays):
            """Interpolate all y_arrays onto the first spectrum's x-axis and average them."""
            x_ref = x_arrays[0]
            y_mean = y_arrays[0].copy()
            for x_i, y_i in zip(x_arrays[1:], y_arrays[1:]):
                y_mean = y_mean + np.interp(x_ref, x_i, y_i, left=0, right=0)
            y_mean = y_mean / len(y_arrays)
            return x_ref, y_mean

        def _clean_label(name):
            """Remove 'run' and 'scan' substrings (case-insensitive) from a spectrum name to keep legends short."""
            cleaned = re.sub(r'(?i)run|scan', '', str(name))
            return cleaned.strip('_- ')

        # plt.figure(figsize=(11,3))
        # initialize a two-element list so we can assign axes by index without IndexError
        self.plot_alignment = [None, None]
        self.plot_alignment[0] = plt.subplot(2, 2, 3)
        for i, sl in enumerate(batches_to_plot):
            batch_items = all_items[sl]
            x_arrays = [spec_data.sel(variable='x').values + self.shifts[j]
                        for j, (_, spec_data) in zip(range(sl.start, sl.stop), batch_items)]
            y_arrays = [spec_data.sel(variable='y').values for (_, spec_data) in batch_items]
            x_batch, y_batch = _batch_mean(x_arrays, y_arrays)
            label = _clean_label(batch_items[0][0]) if len(batch_items) == 1 else f"{_clean_label(batch_items[0][0])}\u2013{_clean_label(batch_items[-1][0])}"
            plt.plot(x_batch, y_batch, label=label, color=color_list[i])
        spec_name = all_items[-1][0]
        if pixel_row_start is not None and pixel_row_stop is not None:
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_start], color='k', linestyle='--')
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_stop], color='k', linestyle='--')
        plt.xlabel(self.spectra_xarray[spec_name].attrs.get('x_name', 'x'))
        plt.ylabel(self.spectra_xarray[spec_name].attrs.get('y_name', 'y'))
        plt.title('Raw spectra')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        self.plot_alignment[1] = plt.subplot(2, 2, 4)
        for i, sl in enumerate(batches_to_plot):
            batch_items = all_items[sl]
            x_arrays = [spec_data.sel(variable='x').values for (_, spec_data) in batch_items]
            y_arrays = [spec_data.sel(variable='y').values for (_, spec_data) in batch_items]
            x_batch, y_batch = _batch_mean(x_arrays, y_arrays)
            label = _clean_label(batch_items[0][0]) if len(batch_items) == 1 else f"{_clean_label(batch_items[0][0])}\u2013{_clean_label(batch_items[-1][0])}"
            plt.plot(x_batch, y_batch, label=label, color=color_list[i])

        if not hasattr(self, 'avg_spectrum_xr_dataset') or self.avg_spectrum_xr_dataset is None:
            ds_avg = self.calculate_average_spectrum()
        else:
            ds_avg = self.avg_spectrum_xr_dataset.copy(deep=True)

        avg_spec = ds_avg['avg_spectrum']
        plt.plot(avg_spec.sel(variable='x'), avg_spec.sel(variable='y')/len(self.spectra_xarray.data_vars), 
                 label='Avg.', linewidth=2, color='black')
        
        # Plot vertical dashed lines for pixel_row_start and pixel_row_stop
        if pixel_row_start is not None and pixel_row_stop is not None:
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_start], color='k', linestyle='--')
            plt.axvline(x=self.spectra_xarray[spec_name].sel(variable='x')[pixel_row_stop], color='k', linestyle='--')
        plt.xlabel(self.spectra_xarray[spec_name].attrs.get('x_name', 'x'))
        plt.ylabel(self.spectra_xarray[spec_name].attrs.get('y_name', 'y'))
        plt.legend()
        plt.title('Aligned spectra')
        plt.grid()
        
        plt.tight_layout()
        # plt.show()

    
    
    def calibrate_energy(self, auto_elastic_determination=False, elastic_line_point=None, calibration=1,
                         poisson_error = True,
                         plot=False):
        """
        Set the elastic line energy point and apply calibration to the spectra.
        Parameters
        ----------
        auto_elastic_determination : bool
            Whether to automatically determine the elastic line point.
        elastic_line_point : float
            The energy point of the elastic line.
        calibration : float
            The calibration factor to be applied to the spectra (in eV/pixel)
        """
        if self.energy_axis_calculated:
            raise ValueError("Energy axis has already been calculated. Skipping...")
        
        else:
            print("\n-> Setting elastic line energy point and applying calibration...")
            if elastic_line_point is None and not auto_elastic_determination:
                raise ValueError("elastic_line_point is not defined and autodetermination is off. Please set it before calling this method.")
            
            if not hasattr(self, 'avg_spectrum_xr_dataset') or self.avg_spectrum_xr_dataset is None:
                self.calculate_average_spectrum(poisson_error=poisson_error)

            if auto_elastic_determination:
                # Automatically determine the elastic line point
                if self.pixel_row_start is None or self.pixel_row_stop is None:
                    raise ValueError("pixel_row_start and pixel_row_stop must be defined for automatic determination.")
                
                # Find the maximum point in the specified interval
                # self.calculate_average_spectrum()
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
                self.spectra_xarray[spec_name].attrs['elastic_line_point'] = elastic_line_point
                self.spectra_xarray[spec_name].attrs['calibration'] = calibration
                self.spectra_xarray[spec_name].attrs['x_name'] = 'Energy Loss (eV)'
            
            self.calculate_average_spectrum(poisson_error=poisson_error)
            self.energy_axis_calculated = True

            print(f"\tElastic line energy point set to {elastic_line_point}, calibration factor {calibration*1000:.4f} meV/pixel.")

            if plot and len(self.spectra_xarray.data_vars) > 1:
                if not hasattr(self, 'plot_alignment'):
                    raise ValueError("Alignment plot not found. Please run align_spectra with plot=True before calling this method with plot=True.")
                ax = self.plot_alignment[1]
                ax.axvline(x=elastic_line_point, color='k', linestyle='--', linewidth=1)
                try:
                    ax.figure.canvas.draw_idle()
                except Exception:
                    pass


class RIXS_Spectra:
    def __init__(self, ds=None, filepath=None, file_list=None,
                 order_by_parameter=None):
        """
        Initialize RIXS_Spectra with either an xarray.Dataset or a file path to an HDF5 file.

        Parameters
        ----------
        ds : xarray.Dataset, optional
            An xarray Dataset containing the 1D RIXS spectra.
        filepath : str, optional
            Path to the HDF5 file to be loaded.
        file_list : list of str, optional
            List of file paths to HDF5 files to be packaged into a single dataset.
        order_by_parameter : str, optional
            Parameter name to order the dataset by when packaging multiple files.
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
        Package all the spectra from different files into a single xarray.Dataset.
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
            with xr.open_dataset(filepath, engine="h5netcdf") as ds_now:
                if len(ds_now.data_vars) > 1:
                    print(f"Warning: More than one xrArray found in {filepath}.")
                for var in ds_now.data_vars:
                    self.ds[str(ii)] = ds_now[var].copy(deep=True)

        if order_by_parameter is not None:
            self._order_by_parameter(order_by_parameter)
            print(f"Ordered dataset by parameter: {order_by_parameter}")


    def _order_by_parameter(self, parameter):
        """
        Order the dataset's DataArrays by one or more attributes.

        Parameters
        ----------
        parameter : str or list of str
            Attribute name(s) to order by. If a list is given, scans are
            ordered by the first parameter; ties are broken by the second,
            then the third, and so on.
        """
        if self.ds is None:
            raise ValueError("No dataset provided.")

        # Normalize to a list of parameters
        if isinstance(parameter, str):
            parameters = [parameter]
        else:
            parameters = list(parameter)

        # Collect (scan_name, (value1, value2, ...)) pairs
        scan_attr_pairs = []
        for scan in self.ds.data_vars:
            values = []
            for param in parameters:
                attr_value = self.ds[scan].attrs.get(param)
                if attr_value is None:
                    raise ValueError(f"Parameter '{param}' to order dataset not found in attributes of scan '{scan}'.")
                values.append(attr_value)
            scan_attr_pairs.append((scan, tuple(values)))

        # Sort by the tuple of attribute values (lexicographic order)
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

        if plot:
            #calculate number of subpanels needed
            num_scans = len(self.ds.data_vars)
            num_cols = 3
            num_rows = (num_scans // num_cols) + (num_scans % num_cols > 0)
            plt.figure(figsize=(num_cols*4, num_rows*4))

        for scan in self.ds.data_vars:
            x_values = self.ds[scan].sel(variable='x').values
            y_values = self.ds[scan].sel(variable='y').values
            norm_values = self.ds[scan].sel(variable='norm').values
            error_values = self.ds[scan].sel(variable='error').values if 'error' in self.ds[scan].coords['variable'] else None
            x_values, y_values, norm_values, error_values, direction_changed = self._set_direction_energy_loss(x_values, y_values,
                                                                              norm_values=norm_values,
                                                                               error_values=error_values,
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
            self.ds[scan].loc[dict(variable='x')] = x_values_shifted 
            self.ds[scan].loc[dict(variable='y')] = y_values 
            self.ds[scan].loc[dict(variable='norm')] = norm_values 
            if error_values is not None:
                self.ds[scan].loc[dict(variable='error')] = error_values

            if plot:
                plt.subplot(num_rows, num_cols, list(self.ds.data_vars).index(scan)+1)
                plt.plot(x_values, normed_y, label='Original Data')
                plt.plot(x_fit, y_fit, 'o', label='Data for Fit')
                plt.plot(x_values, initialfit, label='Fitted Curve', color='red')
                plt.title(f"Scan {scan} - Shift: {shift:.2f}")
                plt.xlim(-resolution*3, resolution*3)
                plt.xlabel('Energy Loss (eV)')
                plt.ylabel('Intensity (arb. units)')
                plt.legend()

        plt.tight_layout()
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
    def _set_direction_energy_loss(x_values, y_values, norm_values = None, error_values=None, positive_energy_loss=True):
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
            error_values = error_values[idx] if error_values is not None else None
            direction_changed = True
        elif sum_left < sum_right and not positive_energy_loss:
            #more intensity at positive energy losses
            # Multiply x_values by -1 and flip all arrays
            x_values = -x_values
            idx = np.argsort(x_values)
            x_values = x_values[idx]
            y_values = y_values[idx]
            norm_values = norm_values[idx] if norm_values is not None else None
            error_values = error_values[idx] if error_values is not None else None
            direction_changed = True
        else:
            #no need to change direction
            direction_changed = False

        return x_values, y_values, norm_values, error_values, direction_changed


    def save_to_hdf5(self, filename,
                     variable_names=None,
                     additional_metadata=None,
                     normalize_spectra=True,
                     divide_normalization_by_value=1,
                     metadata_to_save=None):
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
        
        if additional_metadata is None:
            additional_metadata = {}
        if not isinstance(additional_metadata, dict):
            raise ValueError("additional_metadata must be a dictionary.")
        if variable_names is None:
            variable_names = []


        # for scan in self.ds.data_vars:
        #     motor_values = self.ds[scan].motor_values.values
        #     self.ds[scan].attrs['motor_values'] = motor_values.tolist()
        
        # NOTE joppli 2026-02-21: OLD VERSION
        # self.ds.to_netcdf(filename, engine='h5netcdf')
        # print(f"Dataset saved to {filename}")

        # NOTE joppli 2026-02-21: NEW VERSION BASED ON RAW_XAS_DATA.PY FUNCTIONALITY
        # Normalize metadata_to_save to a set of keys

        print(f"\n-> Saving dataset to .hdf5 format.\n\tData will {'be normalized' if normalize_spectra else 'not be normalized'}. Normalization value will be divided by {divide_normalization_by_value}.")

        if metadata_to_save is None:
            # keep all metadata: collect all attribute keys from dataset and each DataArray
            keep_keys = set()
            for var in self.ds.data_vars:
                keep_keys.update(self.ds[var].attrs.keys())
                keep_keys.update(getattr(self.ds, "attrs", {}).keys())
        elif isinstance(metadata_to_save, (list, tuple, set)):
            keep_keys = set(metadata_to_save)
        else:
            keep_keys = {str(metadata_to_save)}

        new_ds = xr.Dataset()

        for var_name in self.ds.data_vars:
            da = self.ds[var_name]
            if normalize_spectra:
                y_values = da.sel(variable='y').values
                error_values = da.sel(variable='error').values if 'error' in da.coords['variable'] else None
                norm_values = da.sel(variable='norm').values
                normalized_y = y_values / norm_values * divide_normalization_by_value
                normalized_error = error_values / norm_values * divide_normalization_by_value if error_values is not None else None
                da = da.copy(deep=True)
                da.loc[dict(variable='y')] = normalized_y
                if error_values is not None:
                    da.loc[dict(variable='error')] = normalized_error
            # deep copy the DataArray data and coords
            # Copy only selected variables if variable_names provided, otherwise copy whole DataArray
            if variable_names:
                # Normalize variable_names to list
                if not isinstance(variable_names, (list, tuple)):
                    vars_requested = [str(variable_names)]
                else:
                    vars_requested = [str(v) for v in variable_names]

                available_vars = [str(v) for v in da.coords['variable'].values]
                vars_to_copy = [v for v in vars_requested if v in available_vars]

                if len(vars_to_copy) == 0:
                    # If none of the requested variables exist, fall back to copying everything
                    print(f"\tWarning: none of requested variable_names {vars_requested} found in DataArray; copying all variables.")
                    new_da = da.copy(deep=True)
                else:
                    # Preserve the order given in vars_to_copy using positional indices,
                    # which is safe regardless of whether 'variable' is an indexed coord.
                    indices = [list(available_vars).index(v) for v in vars_to_copy]
                    new_da = da.isel(variable=indices).copy(deep=True)
            else:
                new_da = da.copy(deep=True)

            
            # filter attributes to only those requested
            new_da.attrs = {k: v for k, v in new_da.attrs.items() if k in keep_keys}
            # divide 'mirror' attribute by divide_normalization_by_value
            if 'mirror' in new_da.attrs:
                new_da.attrs['mirror'] = new_da.attrs['mirror'] / divide_normalization_by_value
            new_ds[var_name] = new_da

        # also filter dataset-level attributes if present
        new_ds.attrs = {k: v for k, v in getattr(self.ds, "attrs", {}).items() if k in keep_keys}
        for data_var in self.ds.data_vars:
            da = new_ds[data_var]
            da.attrs['normalization_divide_value'] = divide_normalization_by_value
            for k, v in additional_metadata.items():
                da.attrs[k] = v

        # new_ds.attrs['units'] = units_names
        new_ds.to_netcdf(filename, engine='h5netcdf')
        print(f"\tDataset saved to {filename}")


    def save_to_csv(self, filename, motors_dict=None,
                                    normalize_spectra=True,
                                    save_errorbars=False,
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
        motors_dict : dict
            Dictionary mapping motor names in the dataset to motor names printed in the csv file.
        normalize_spectra : bool
            If True, normalize the spectra by the normalization dataset.
        divide_normalization_by_value : float
            Value to divide the normalization dataset by when normalizing the spectra (e.g. 1E6 for mirror at ESRF)
        positive_energy_loss : bool
            If True, set the direction of energy loss to positive. If False, set it to negative.
        """

        print(f"\n-> Saving dataset to .csv format.\n\tData will {'be normalized' if normalize_spectra else 'not be normalized'}. Normalization value will be divided by {divide_normalization_by_value}.")
        has_error = all(
            'error' in self.ds[scan].coords['variable'].values
            for scan in self.ds.data_vars
        )
        if not has_error and save_errorbars:
            print("\tWarning: Not all scans have 'error' variable. Error bars will not be saved.")
            save_errorbars = False

        # Ensure the filename ends with ".csv"
        if not filename.lower().endswith('.csv'):
            base, ext = os.path.splitext(os.path.basename(filename))
            if ext.lower() != '.csv':
                print(f"\tWarning: Changing file extension to .csv for {filename}")
                filename = os.path.join(os.path.dirname(filename), base + '.csv')
            filename += '.csv'

        first_scan = list(self.ds.data_vars)[0]

        with open(filename, 'w', encoding='utf-8') as file:

            # Write the header with motor values
            # Collect motor values for all scans
            motor_values_all_scans = {motor: [] for motor in motors_dict.values()}
            for scan in self.ds.data_vars:
                for motor in motors_dict.values():
                    if motor in self.ds[scan].attrs:
                        motor_values_all_scans[motor].append(self.ds[scan].attrs[motor])
                    else:
                        motor_values_all_scans[motor].append("")

            header = ''
            # Write motor values in the header
            for metadata in motors_dict.keys():
                header_parts = [metadata]
                header_parts_1 = [' ' for _ in range(len(self.ds.data_vars))]
                header_parts_2 = []
                if metadata == 'mirror':
                        if normalize_spectra:
                            header_parts_2 += [f"{np.mean(self.ds[scan].sel(variable='norm').values)/divide_normalization_by_value:.2f}" for i, scan in enumerate(self.ds.data_vars)]
                        else:
                            header_parts_2 += [f"{np.mean(self.ds[scan].sel(variable='norm').values):.2f}" for i, scan in enumerate(self.ds.data_vars)]
                else:                
                    if motors_dict[metadata] is not None:
                        header_parts_2 += [(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                            for i, value in enumerate(motor_values_all_scans[motors_dict[metadata]])
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
            if save_errorbars:
                header += ' ,'+','.join([f"Energy Loss,{self.ds[scan].attrs['run']},Error" for scan in self.ds.data_vars])
                header += '\n'
                units = ' ,'+','.join(['(eV),(arb. units),(arb.units)'] * len(self.ds.data_vars))
            else:
                header += ' ,'+','.join([f"Energy Loss,{self.ds[scan].attrs['run']}" for scan in self.ds.data_vars])
                header += '\n'
                units = ' ,'+','.join(['(eV),(arb. units)'] * len(self.ds.data_vars))
            units += '\n'
            header += units
            file.write(f"{header}\n")

            # Stack all x and y values as adjacent columns
            all_data = []
            for scan in self.ds.data_vars:
                x_values = self.ds[scan].sel(variable='x').values
                y_values = self.ds[scan].sel(variable='y').values
                norm_values = self.ds[scan].sel(variable='norm').values
                
                error_values = self.ds[scan].sel(variable='error').values if save_errorbars else None

                if normalize_spectra:
                    # Normalize the y-values (and the errors) by the norm values
                    y_values = y_values / norm_values * divide_normalization_by_value
                    if save_errorbars:
                        error_values = error_values / norm_values * divide_normalization_by_value 

                # Calculate the sum of y_values for x_values < 0 and x_values > 0
                x_values, y_values, norm_values, error_values, _ = self._set_direction_energy_loss(x_values, y_values,
                                                                                  norm_values=norm_values,
                                                                                  error_values=error_values,
                                                                                  positive_energy_loss=positive_energy_loss)
                
                if save_errorbars:
                    # Calculate error bars as sqrt(y_values) for Poisson statistics
                    all_data.append(np.column_stack((x_values, y_values, error_values)))
                else:
                    all_data.append(np.column_stack((x_values, y_values)))

            # Concatenate all data along the second axis
            concatenated_data = np.concatenate(all_data, axis=1)

            # Write the concatenated data to the file
            for row in concatenated_data:
                file.write(' ,'+','.join(map(str, row)) + '\n')

        print(f"\tDataset saved to {filename}")



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

    def get_map_with_parameter(self, parameter, 
                               divide_normalization_by_value=1):
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
            norm_values = self.ds[da_name].sel(variable='norm').values / divide_normalization_by_value
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


