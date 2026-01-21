import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import time
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from cmcrameri import cm
from IPython.display import display
import pandas as pd



class Generated_1D_RIXS_Spectra:
    def __init__(self, ds):
        """
        Class to handle 1D RIXS spectra generated from 2D RIXS images.
        To be used with xarray datasets generated either by extract_rixs_spectra.py or by generate_rixs_spectra.py
        Parameters
        ----------
        ds : xarray.Dataset
            The xarray Dataset containing the 1D RIXS spectra.
        """
        self.spectra_xarray = ds


    def align_spectra(self,
                        pixel_row_start=None,
                        pixel_row_stop=None,
                        fit_shifts=False,
                        smooth_shifts=False,
                        correlation_batch_size=10,
                        poly_order=1,
                        plot=False,):
        """
        Process the extracted spectra to align energy shifts and calculate average spectrum.
        Parameters
        ----------
        pixel_row_start : int
            Starting pixel row for the correlation region (typically near elastic line).
        pixel_row_stop : int
            Ending pixel row for the correlation region (typically near elastic line).
        fit_shifts : bool
            Whether to fit the shifts to a polynomial function.
        smooth_shifts : bool
            Whether to smooth the shifts using a Gaussian filter.
        correlation_batch_size : int
            Number of spectra to average for correlation calculation.
        poly_order : int
            Order of the polynomial for fitting shifts if fit_shifts is True.
        plot : bool
            Whether to plot the processed spectra.
        """

      
        # Find aligning range using the _find_aligning_range method
        if pixel_row_start is None or pixel_row_stop is None:
            self.pixel_row_start, self.pixel_row_stop = self._find_aligning_range(self.spectra_xarray, threshold=0.1)
        else:
            self.pixel_row_start = pixel_row_start
            self.pixel_row_stop = pixel_row_stop

        # Stack all y_data arrays along a new dimension
        all_y_data = np.stack([self.spectra_xarray[spec].sel(variable='y').values for spec in self.spectra_xarray.data_vars], axis=0)
        if all_y_data.shape[0] == 1:
            print("Only one spectrum found. Skipping alignment.")
            return

        #save the pixel_row_start, pixel_row_stop and sample name inside the spectra_xarrays
        for spec_name in self.spectra_xarray.data_vars:
            self.spectra_xarray[spec_name].attrs['pixel_row_start'] = self.pixel_row_start
            self.spectra_xarray[spec_name].attrs['pixel_row_stop'] = self.pixel_row_stop

        ### correct the energy shifts
        _ = self._correct_shift(all_y_data, 
                                self.pixel_row_start, self.pixel_row_stop, 
                                fit_shifts=fit_shifts, 
                                smooth_shifts=smooth_shifts,
                                correlation_batch_size=correlation_batch_size, 
                                poly_order=poly_order)     
                   
        if plot:
            self.plot_spectra(True, pixel_row_start=self.pixel_row_start, pixel_row_stop=self.pixel_row_stop)

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
                RIXS_Spectra(ds=ds_avg).save_to_hdf5(file_path_save)
            else:
                RIXS_Spectra(ds=self.avg_spectrum_xr_dataset).save_to_hdf5(file_path_save)
            print("Only the average spectrum was saved to the HDF5 file.")
        else:
            RIXS_Spectra(ds=self.spectra_xarray).save_to_hdf5(file_path_save)
            

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
            raise ValueError("file_path_save must be defined when save_to_tx is True.")
        if save_avg_spectrum:
            ds_avg = self.calculate_average_spectrum()
            RIXS_Spectra(ds=ds_avg).save_to_txt(file_path_save, save_avg_spectrum=save_avg_spectrum)
        else:
            RIXS_Spectra(ds=self.spectra_xarray).save_to_txt(file_path_save, save_avg_spectrum=save_avg_spectrum)

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
        
        #shifts
        if align_spectra:
            plt.figure(figsize=(11,6))
            plt.subplot(2,1,1)
            plt.plot(self.real_shifts, 'ko-', label='Real Shifts')  # 'ko-' for black circles connected by lines
            plt.plot(self.shifts, 'ro-', label='Used Shifts')
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
            plt.subplot(2, 1, 2)
            plt.plot(integrals, 'bo-', label='Integrated intensity')
            plt.axhline(np.mean(integrals), color='k', linestyle='--', label='Mean')
            plt.xlabel('Image Index')
            plt.ylabel('Integrated intensity (arb. units)')
            plt.title(f'Integral between pixels {i0} and {i1}')
            plt.grid()
            plt.legend()

            plt.show()

        max_spectra = 10
        spectra_to_plot = list(self.spectra_xarray.items())[::max(1, len(self.spectra_xarray) // max_spectra)]
        color_list = [cm.managua(i) for i in np.linspace(0, 1, len(spectra_to_plot))]
        plt.figure(figsize=(11,3))
        plt.subplot(1, 2, 1)
        for i, (spec_name, spec_data) in enumerate(spectra_to_plot):
            plt.plot(spec_data.sel(variable='x')+self.shifts[i], 
                    spec_data.sel(variable='y'), label=spec_name, color=color_list[i])
        plt.xlabel(self.spectra_xarray[spec_name].attrs.get('x_name', 'x'))
        plt.ylabel(self.spectra_xarray[spec_name].attrs.get('y_name', 'y'))
        plt.title('Raw spectra')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        for i, (spec_name, spec_data) in enumerate(spectra_to_plot):
            plt.plot(spec_data.sel(variable='x'), 
                    spec_data.sel(variable='y'), label=spec_name, color=color_list[i])

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
        plt.show()


    @staticmethod
    def _find_aligning_range(spectra_xarray, threshold=0.05):
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

        # Extract the first x-axis, stack all y_data arrays along a new dimension
        x_data = spectra_xarray[list(spectra_xarray.data_vars)[0]].sel(variable='x').values
        all_y_data = np.stack([spectra_xarray[spec].sel(variable='y').values for spec in spectra_xarray.data_vars], axis=0)
        avg_spectrum = np.mean(all_y_data, axis=0)

        # Find where signal rises above threshold * max intensity
        threshold_value = threshold * np.max(avg_spectrum)
        # Calculate the moving average with a window of 3
        moving_avg = np.convolve(avg_spectrum, np.ones(3)/3, mode='same')
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
    
    
    def calibrate_energy(self, auto_elastic_determination=False, elastic_line_point=None, calibration=1):
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
                self.spectra_xarray[spec_name].attrs['elastic_line_point'] = elastic_line_point
                self.spectra_xarray[spec_name].attrs['calibration'] = calibration
                self.spectra_xarray[spec_name].attrs['x_name'] = 'Energy Loss (eV)'
            
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

        first_scan = list(self.ds.data_vars)[0]
        print(f"Saving dataset- Spectra will be normalized by {self.ds[first_scan].attrs['norm_name']}, divided by {divide_normalization_by_value}.\n")
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