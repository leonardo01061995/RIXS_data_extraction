import os
import re
import numpy as np
import xarray as xr


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
            #scans with progressive numbers (e.g., 123.1, 123.2) are allowed, but only the integer part is used to find the scan
            #the progressive number is stored to select the correct dataset later
            progressive_number = None
            if isinstance(scan, str) and re.match(r"^\d+\.\d+$", scan):
                x, y = scan.split('.')
                scan = x
                progressive_number = int(y)
            else:
                progressive_number = 1

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
                scan_start = scan_positions[progressive_number-1]
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
        self.normalized_data = xr.Dataset()
        
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
                self.normalized_data[scan] = xr.DataArray(
                    data=data,
                    dims=['points', 'variable'],
                    coords={
                    'points': np.arange(data.shape[0]),
                    'variable': ['x', 'y', 'norm']
                    }
                )
                self.normalized_data[scan].attrs['x_name'] = x_name
                self.normalized_data[scan].attrs['y_name'] = y_name
                self.normalized_data[scan].attrs['norm_name'] = norm_name
                
                if motor_names:
                    for motor, value in zip(motor_names, motor_values):
                        self.normalized_data[scan].attrs[motor] = value
                    if 'hu70cp' in motor_names and 'hu70ap' in motor_names:
                        self.normalized_data[scan].attrs['polarization'] = self._determine_polarization(
                            self.normalized_data[scan].attrs['hu70ap'], 
                            self.normalized_data[scan].attrs['hu70cp']
                        )

                self.normalized_data[scan].attrs['date'] = ds.attrs['date']
                self.normalized_data[scan].attrs['run'] = self.run
                self.normalized_data[scan].attrs['scan'] = ds[scan].attrs['scan']
                self.normalized_data[scan].attrs['filename'] = self.filename

        return self.normalized_data


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
        if self.normalized_data is None:
            raise ValueError("No dataset provided.")
        
        self.normalized_data.to_netcdf(filename, engine='h5netcdf')
        print(f"Dataset saved to {filename}")

    def save_to_csv(self, filename, motor_names, motor_name_mapping,
                                  metadata_in_csv=['Q', 'theta','2theta','phi','energy','polarization',
                                                      'mirror','sample','B [T]', 'T [K]',],
                                    set_metadata_value = None,
                                    normalize_spectra=False,
                                    divide_normalization_by_value=1,):
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
            Dictionary mapping motor names in the dataset to metadata names written in the csv file.
        metadata_in_csv : list of str
            List of metadata attributes to include in the CSV file.
        set_metadata_value : dict or None
            Dictionary with fixed values for specific metadata to include in the CSV file.
            If None, the default values from the dataset will be used.
            Set metadata could be either single values or lists of values (if different for each scan).
        normalize_spectra : bool
            If True, normalize the spectra by the normalization dataset.
        divide_normalization_by_value : float
            Value to divide the normalization dataset by when normalizing the spectra (e.g. 1E6 for mirror at ESRF)
        """

        # Ensure the filename ends with ".csv"
        if not filename.lower().endswith('.csv'):
            base, ext = os.path.splitext(os.path.basename(filename))
            if ext.lower() != '.csv':
                print(f"Warning: Changing file extension to .csv for {filename}")
                filename = os.path.join(os.path.dirname(filename), base + '.csv')
            filename += '.csv'

        # Extract the first scan name from self.normalized_data.data_vars
        first_scan_name = list(self.normalized_data.data_vars)[0]
        print(f"Saving dataset- Spectra will be normalized by {self.normalized_data[first_scan_name].attrs['norm_name']}, divided by {divide_normalization_by_value}.\n")
        xname = str(self.normalized_data[first_scan_name].attrs['x_name'])
        yname = str(self.normalized_data[first_scan_name].attrs['y_name'])
        norm_name = str(self.normalized_data[first_scan_name].attrs['norm_name'])

        with open(filename, 'w', encoding='utf-8') as file:

            # Write the header with motor values
            # Collect motor values for all scans
            motor_values_all_scans = {motor: [] for motor in motor_names}
            for scan in self.normalized_data.data_vars:
                for motor in motor_names:
                    if motor in self.normalized_data[scan].attrs:
                        motor_values_all_scans[motor].append(self.normalized_data[scan].attrs[motor])
                    else:
                        motor_values_all_scans[motor].append("")

            header = ''
            # Write motor values in the header
            for metadata in metadata_in_csv:
                header_parts = [metadata]
                header_parts_1 = [' ' for _ in range(len(self.normalized_data.data_vars))]
                header_parts_2 = []

                if set_metadata_value is not None and metadata in set_metadata_value:
                        if len(set_metadata_value[metadata])==1:
                            value = set_metadata_value[metadata][0]
                            header_parts_2 += [f"{value:.2f}" if isinstance(value, (int, float)) else str(value) 
                                            for _ in range(len(self.normalized_data.data_vars))]
                        else:
                            header_parts_2 += [f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                                               for value in set_metadata_value[metadata]]
                else:
                    #value is not set by user, search it in the motor values saved in the dataset
                    if metadata == 'mirror':
                        # If a fixed value is provided for this metadata, use it for all scans
                        header_parts_2 += [f"{np.mean(self.normalized_data[scan].sel(variable='norm').values)/divide_normalization_by_value:.2f}" for i, scan in enumerate(self.normalized_data.data_vars)]
                    else:
                        if metadata in motor_name_mapping:
                            #name of the origin metadata in the motor values is known
                            if motor_name_mapping[metadata] is not None:
                                header_parts_2 += [(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                                    for i, value in enumerate(motor_values_all_scans[motor_name_mapping[metadata]])
                                ]
                        else:
                            header_parts_2 += [" "] * (len(self.normalized_data.data_vars))


                for idx in range(1, len(self.normalized_data.data_vars) * 2 + 1):
                    if idx % 2 == 1:
                        header_parts.append(header_parts_1[(idx - 1) // 2])
                    else:
                        header_parts.append(header_parts_2[(idx - 1) // 2])

                header += ','.join(header_parts)  # Repeat each motor value twice
                header += '\n'

            # Write a line of "Energy Loss" and {scan} alternating
            header += ' ,'+','.join([f"{xname},{scan}" for scan in self.normalized_data.data_vars])
            header += '\n'
            units = ' ,'+','.join([' ,(arb. units)'] * len(self.normalized_data.data_vars))
            units += '\n'
            header += units
            file.write(f"{header}\n")

            # Create a DataFrame to store all scans
            data_frames = []
            # Stack all x and y values as adjacent columns
            all_data = []
            for scan in self.normalized_data.data_vars:
                x_values = self.normalized_data[scan].sel(variable='x').values
                y_values = self.normalized_data[scan].sel(variable='y').values
                norm_values = self.normalized_data[scan].sel(variable='norm').values
                if normalize_spectra:
                    # Normalize the y-values by the norm values
                    y_values = y_values / norm_values * divide_normalization_by_value

                # # Calculate the sum of y_values for x_values < 0 and x_values > 0
                # x_values, y_values, norm_values,_ = self._set_direction_energy_loss(x_values, y_values,
                #                                                                   norm_values=norm_values,
                #                                                                   positive_energy_loss=positive_energy_loss)
                all_data.append(np.column_stack((x_values, y_values)))

            # Concatenate all data along the second axis
            concatenated_data = np.concatenate(all_data, axis=1)

            # Write the concatenated data to the file
            for row in concatenated_data:
                file.write(' ,'+','.join(map(str, row)) + '\n')

        print(f"Dataset saved to {filename}")
    
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