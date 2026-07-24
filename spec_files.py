import os
import re
import numpy as np
import xarray as xr
from static_functions import _determine_polarization


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
    


    def _read_spec_file(self, scans_selected, scans_from_same_run=False, read_c_lines=True):
        """
        Extract scan data, column names, motor names/values, #C metadata and the
        date from a .spec file into an xarray.Dataset.

        Parameters
        ----------
        scans_selected : str or list
            'all' to read every scan, or a list/space-separated string of scan
            identifiers. An identifier of the form '123.2' selects the 2nd
            occurrence of scan 123.
        scans_from_same_run : bool
            If True, scans are identified by their intra-run number parsed from
            the '#S N  Scan RUN:SCAN' header; otherwise by the '#S N' number.

        Returns
        -------
        xarray.Dataset
            One DataArray per scan (dims ['points', 'datasets']), with motor and
            #C values stored in .attrs, plus a dataset-level 'date' attribute.
        """

        # ---- nested helper: parse #C metadata (unchanged from original) -------
        def _parse_c_params(scan_text):
            matches = re.findall(r"#C[ \t]?(.*?)(?=#C|\n#|\Z)", scan_text)
            params = {}
            section = None
            for raw in matches:
                line = raw.strip()
                if not line:
                    continue
                start_m = re.match(r"--\s*(.+?)\s+start\s*$", line, re.IGNORECASE)
                end_m = re.match(r"--\s*(.+?)\s+end\s*$", line, re.IGNORECASE)
                if start_m:
                    section = start_m.group(1).strip().lower().replace(' ', '_')
                    continue
                if end_m:
                    section = None
                    continue
                if line.startswith('/'):
                    key, value = 'raw_file', line
                else:
                    parts = re.split(r'\s{2,}', line)
                    if len(parts) >= 2:
                        key, value = parts[0].strip(), parts[-1].strip()
                    else:
                        key, value = line, None
                    if value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    elif value is not None:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                if key in params:
                    raise ValueError(
                        f"Duplicate #C parameter name '{key}' found while parsing scan header "
                        f"(section: {section!r}). Remove the prefix-based flattening or rename "
                        f"one of the colliding keys to resolve this."
                    )
                params[key] = value
            return params

        content = self.file_content
        if content is None:
            raise ValueError(f"File content is empty; could not read {self.filename!r}.")

        # ---- 1. Split the whole file into scan blocks in a SINGLE pass ---------
        # group(1) = the '#S' file number
        # group(2) = the intra-run scan number ('Scan RUN:SCAN'), or None if absent
        header_re = re.compile(r'^#S[ \t]+(\d+)(?:[ \t]+Scan[ \t]+\d+:(\d+))?', re.MULTILINE)
        headers = list(header_re.finditer(content))
        if not headers:
            raise ValueError(f"No '#S' scan headers found in {self.filename!r}.")

        starts = [m.start() for m in headers]
        ends = starts[1:] + [len(content)]          # each block ends where the next '#S' begins

        blocks = []                                  # [(key, start, end), ...]
        key_to_blocks = {}                           # key -> [block indices] (handles duplicates)
        for i, m in enumerate(headers):
            key = m.group(2) if scans_from_same_run else m.group(1)
            blocks.append((key, starts[i], ends[i]))
            if key is not None:
                key_to_blocks.setdefault(key, []).append(i)

        # ---- 2. Decide which blocks to process, and the label for each --------
        selected = []                                # [(label, block_index), ...]
        if scans_selected == 'all':
            for i, (key, _, _) in enumerate(blocks):
                if key is None:                      # header without the expected format
                    continue
                selected.append((key, i))
        else:
            scans = scans_selected.split() if isinstance(scans_selected, str) else scans_selected
            for scan in scans:
                progressive_number = 1
                scan_key = str(scan)
                if isinstance(scan, str) and re.match(r'^\d+\.\d+$', scan):
                    scan_key, prog = scan.split('.')
                    progressive_number = int(prog)
                idxs = key_to_blocks.get(scan_key, [])
                if not idxs:
                    print(f"Scan {scan} not found.")
                    continue
                if progressive_number > len(idxs):
                    print(f"Scan {scan}: only {len(idxs)} occurrence(s) found; using the first.")
                    progressive_number = 1
                selected.append((scan_key, idxs[progressive_number - 1]))

        # ---- 3. File-level date, extracted ONCE (not once per scan) -----------
        date_match = re.search(r'#D (.+)', content)
        date = date_match.group(1) if date_match else None

        # ---- 3b. Global motor names from the preamble (before the first #S) ---
        # Many beamlines write the '#O' motor-name block only once, in the file
        # header, and then only the per-scan '#P' value lines inside each block.
        # Parse those names once here; they serve as a fallback for any scan
        # block that has no '#O' lines of its own. A block with its own '#O'
        # still overrides this, so files that repeat '#O' per scan are unaffected.
        # Empty tokens (from trailing whitespace, e.g. '#O2 AD  ') are dropped so
        # the name count matches the '#P' value count.
        preamble = content[:starts[0]]
        global_motnames = []
        for line in re.findall(r'#O\d+[ \t]{1,2}(.+)', preamble):
            global_motnames.extend(name for name in line.split('  ') if name.strip())

        # ---- 4. Parse each selected block exactly once ------------------------
        ds = xr.Dataset()
        for label, block_idx in selected:
            _, start, end = blocks[block_idx]
            scan_text = content[start:end]           # the only slice we make, once per scan

            # column names (#L)
            col_match = re.search(r'#L\s{1,2}(.+)', scan_text)
            colnames = col_match.group(1).split('  ') if col_match else []

            # motor names (#O...) and values (#P...)
            # Prefer '#O' lines inside the scan block (some beamlines repeat them
            # per scan); fall back to the global preamble names otherwise.
            motnames = []
            for line in re.findall(r'#O\d+\s{1,2}(.+)', scan_text):
                motnames.extend(name for name in line.split('  ') if name.strip())
            if not motnames:
                motnames = global_motnames

            motvals = []
            for line in re.findall(r'#P\d+\s{1,2}(.+)', scan_text):
                motvals.extend(
                    float(v) if v != "b'ERR'" else np.nan
                    for v in re.split(r'\s{1,2}', line.strip())
                )

            # Names and values are paired positionally (#O0↔#P0, #O1↔#P1, ...).
            # A count mismatch means silent misalignment, so warn rather than
            # zip-truncate quietly.
            if motnames and len(motnames) != len(motvals):
                print(
                    f"Scan {label}: {len(motnames)} motor name(s) but "
                    f"{len(motvals)} value(s); pairing the first "
                    f"{min(len(motnames), len(motvals))} and dropping the rest."
                )

            # #C metadata
            if read_c_lines:
                cparams = _parse_c_params(scan_text)

            # numerical data (fast parser, see helper below)
            data = self._parse_data_block(scan_text)
            if data.size == 0:
                print(f"Scan {label}: no numerical data found, skipping.")
                continue

            # guard against label collisions (e.g. a file with repeated intra-run
            # scan numbers). Keep the first occurrence and warn, rather than
            # silently overwriting it.
            var_name = f'scan_{label}'
            if var_name in ds.data_vars:
                print(f"Warning: two scans map to '{var_name}'; keeping the first, ignoring the duplicate.")
                continue

            # build the DataArray for this scan
            da = xr.DataArray(
                data,
                dims=['points', 'datasets'],
                coords={'points': np.arange(data.shape[0]), 'datasets': colnames},
            )
            da.attrs.update({name: value for name, value in zip(motnames, motvals)})
            if read_c_lines:
                da.attrs.update(cparams)
            da.attrs['scan'] = label
            ds[var_name] = da

        ds.attrs['date'] = date
        return ds
 
 
    @staticmethod
    def _parse_data_block(scan_text):
        """
        Parse the numerical rows of a scan block into a 2-D float array.
    
        Fast path: split each line and let numpy do a single C-level float
        conversion. This is generally several times faster than np.loadtxt, which
        parses line-by-line in Python.
    
        Fallback: if a stray non-numeric token (e.g. 'ERR') or a ragged row makes
        the fast path fail, parse token-by-token and pad short rows with NaN so the
        result is always rectangular.
        """
        data_lines = [
            ln for ln in scan_text.split('\n')
            if ln and not ln.startswith('#') and not ln.startswith(' ')
        ]
        if not data_lines:
            return np.empty((0, 0))
    
        try:
            # np.loadtxt is C-accelerated on numpy >= 1.23 and is the fastest option.
            return np.loadtxt(data_lines, ndmin=2)
        except ValueError:
            # robust fallback for stray non-numeric tokens (e.g. 'ERR') or ragged rows
            rows = [[SpecFile._to_float(tok) for tok in ln.split()] for ln in data_lines]
            width = max(len(r) for r in rows)
            arr = np.full((len(rows), width), np.nan)
            for i, r in enumerate(rows):
                arr[i, :len(r)] = r
            return arr
 
    
    @staticmethod
    def _to_float(tok):
        try:
            return float(tok)
        except ValueError:
            return np.nan


    def _read_spec_file_old(self, scans_selected, scans_from_same_run=False):
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
        cparams_all_scans = []
        date = None

        def _parse_c_params(scan_text):
            matches = re.findall(r"#C[ \t]?(.*?)(?=#C|\n#|\Z)", scan_text)

            params = {}
            section = None
            for raw in matches:
                line = raw.strip()
                if not line:
                    continue

                start_m = re.match(r"--\s*(.+?)\s+start\s*$", line, re.IGNORECASE)
                end_m = re.match(r"--\s*(.+?)\s+end\s*$", line, re.IGNORECASE)
                if start_m:
                    section = start_m.group(1).strip().lower().replace(' ', '_')
                    continue
                if end_m:
                    section = None
                    continue

                if line.startswith('/'):
                    key, value = 'raw_file', line
                else:
                    parts = re.split(r'\s{2,}', line)
                    if len(parts) >= 2:
                        key, value = parts[0].strip(), parts[-1].strip()
                    else:
                        key, value = line, None

                    if value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    elif value is not None:
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                # NEW: no section prefix - but guard against silent collisions
                if key in params:
                    raise ValueError(
                        f"Duplicate #C parameter name '{key}' found while parsing scan header "
                        f"(section: {section!r}). Remove the prefix-based flattening or rename "
                        f"one of the colliding keys to resolve this."
                    )
                params[key] = value

            return params

        
        if scans_selected == 'all':

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
            
            # NEW: extract #C metadata for this scan (flat dict, no section prefixes)
            cparams_all_scans.append(_parse_c_params(scan_content))

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
            ds[f'scan_{scan}'].attrs.update(cparams_all_scans[i])  # NEW: merged flat, same level as motors
            ds[f'scan_{scan}'].attrs['scan'] = scan
        
        
        ds.attrs['date'] = date
        return ds


    def extract_data(self, scans, x_name, y_name, norm_name, var_names=None,
                     motors_dict=None, scans_from_same_run=False, read_c_lines=True):
        """
        Extract and normalize data based on specified x, y, and normalization datasets,
        and include specified motor names and values.

        Parameters
        ----------
        x_name, y_name, norm_name : str or list of str
            Name(s) of the dataset(s) to use as x-axis, y-axis and normalization.
        var_names : list of str, optional
            Explicit labels for the 'variable' coordinate, one per selected column,
            ordered as x-columns, then y-columns, then norm-columns. If None, labels
            default to 'x'/'y'/'norm' for a string input and to the dataset names
            themselves for a list input.
        motors_dict : dict
            Dictionary mapping motor names to their corresponding variable names.

        Returns
        -------
        xarray.Dataset
            Dataset containing the normalized data and specified motor values.
        """
        print(f"\tExtracting data for run {self.run} from scans: {scans}. X-axis: {x_name}, Y-axis: {y_name}, Normalization: {norm_name}.")
        ds = self._read_spec_file(scans_selected=scans, scans_from_same_run=scans_from_same_run, read_c_lines=read_c_lines)
        self.normalized_data = xr.Dataset()

        # Convert every name to a list of dataset names BEFORE taking any len(),
        # otherwise len() on a string counts its characters.
        def _as_list(name):
            return [name] if isinstance(name, str) else list(name)

        x_names = _as_list(x_name)
        y_names = _as_list(y_name)
        norm_names = _as_list(norm_name)
        all_names = x_names + y_names + norm_names

        # Build the 'variable' coordinate labels.
        if var_names is None:
            # Default: a string keeps its role label; a list uses the dataset names.
            x_labels = ['x'] if isinstance(x_name, str) else list(x_name)
            y_labels = ['y'] if isinstance(y_name, str) else list(y_name)
            norm_labels = ['norm'] if isinstance(norm_name, str) else list(norm_name)
            variable_labels = x_labels + y_labels + norm_labels
        else:
            variable_labels = list(var_names)
            if len(variable_labels) != len(all_names):
                raise ValueError(
                    f"var_names has {len(variable_labels)} entries but "
                    f"{len(all_names)} datasets were selected "
                    f"(x: {len(x_names)}, y: {len(y_names)}, norm: {len(norm_names)})."
                )

        for scan in ds.data_vars:
            if all(name in ds[scan].datasets for name in all_names):
                # Each selection is a length-N 1-D array; stacking on axis=1 gives (N, K),
                # K = total selected datasets (x-columns + y-columns + norm-columns).
                columns = [ds[scan].sel(datasets=name).values for name in all_names]
                data = np.stack(columns, axis=1)

                self.normalized_data[scan] = xr.DataArray(
                    data=data,
                    dims=['points', 'variable'],
                    coords={
                        'points': np.arange(data.shape[0]),
                        'variable': variable_labels
                    }
                )
                self.normalized_data[scan].attrs['x_name'] = x_name
                self.normalized_data[scan].attrs['y_name'] = y_name
                self.normalized_data[scan].attrs['norm_name'] = norm_name

                if motors_dict:
                    for motor in motors_dict.keys():
                        if motors_dict[motor] in ds[scan].attrs:
                            self.normalized_data[scan].attrs[motor] = ds[scan].attrs[motors_dict[motor]]
                        else:
                            self.normalized_data[scan].attrs[motor] = np.nan
                            print(f"\tWarning: Motor '{motors_dict[motor]}' not found in scan {scan}. Setting its value to NaN.")
                    if 'hu70cp' in ds.attrs and 'hu70ap' in ds.attrs:
                        self.normalized_data[scan].attrs['polarization'] = self._determine_polarization(
                            ds[scan].attrs['hu70ap'],
                            ds[scan].attrs['hu70cp']
                        )

                self.normalized_data[scan].attrs['date'] = ds.attrs['date']
                if self.run is not None:
                    self.normalized_data[scan].attrs['run'] = str(self.run)
                self.normalized_data[scan].attrs['scan'] = ds[scan].attrs['scan']
                self.normalized_data[scan].attrs['filename'] = self.filename

        return self.normalized_data



    def order_by_parameter(self, parameter):
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
        if self.normalized_data is None:
            raise ValueError("No dataset provided.")
        
        if additional_metadata is None:
            additional_metadata = {}
        if not isinstance(additional_metadata, dict):
            raise ValueError("additional_metadata must be a dictionary.")
        if variable_names is None:
            variable_names = []


        print(f"\n-> Saving dataset to .csv format.\n\tData will {'be normalized' if normalize_spectra else 'not be normalized'}. Normalization value will be divided by {divide_normalization_by_value}.")

        if metadata_to_save is None:
            # keep all metadata: collect all attribute keys from dataset and each DataArray
            keep_keys = set()
            for var in self.normalized_data.data_vars:
                keep_keys.update(self.normalized_data[var].attrs.keys())
                keep_keys.update(getattr(self.normalized_data, "attrs", {}).keys())
        elif isinstance(metadata_to_save, (list, tuple, set)):
            keep_keys = set(metadata_to_save)
        else:
            keep_keys = {str(metadata_to_save)}

        new_ds = xr.Dataset()

        for var_name in self.normalized_data.data_vars:
            da = self.normalized_data[var_name]
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
        new_ds.attrs = {k: v for k, v in getattr(self.normalized_data, "attrs", {}).items() if k in keep_keys}
        for data_var in self.normalized_data.data_vars:
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
            'error' in self.normalized_data[scan].coords['variable'].values
            for scan in self.normalized_data.data_vars
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

        first_scan = list(self.normalized_data.data_vars)[0]

        with open(filename, 'w', encoding='utf-8') as file:

            # Write the header with motor values
            # Collect motor values for all scans
            motor_values_all_scans = {motor: [] for motor in motors_dict.values()}
            for scan in self.normalized_data.data_vars:
                for motor in motors_dict.values():
                    if motor in self.normalized_data[scan].attrs:
                        motor_values_all_scans[motor].append(self.normalized_data[scan].attrs[motor])
                    else:
                        motor_values_all_scans[motor].append("")

            header = ''
            # Write motor values in the header
            for metadata in motors_dict.keys():
                header_parts = [metadata]
                header_parts_1 = [' ' for _ in range(len(self.normalized_data.data_vars))]
                header_parts_2 = []
                if metadata == 'mirror':
                        if normalize_spectra:
                            header_parts_2 += [f"{np.mean(self.normalized_data[scan].sel(variable='norm').values)/divide_normalization_by_value:.2f}" for i, scan in enumerate(self.normalized_data.data_vars)]
                        else:
                            header_parts_2 += [f"{np.mean(self.normalized_data[scan].sel(variable='norm').values):.2f}" for i, scan in enumerate(self.normalized_data.data_vars)]
                else:                
                    if motors_dict[metadata] is not None:
                        header_parts_2 += [(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                            for i, value in enumerate(motor_values_all_scans[motors_dict[metadata]])
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
            if save_errorbars:
                header += ' ,'+','.join([f"Energy Loss,{self.normalized_data[scan].attrs['run']},Error" for scan in self.normalized_data.data_vars])
                header += '\n'
                units = ' ,'+','.join(['(eV),(arb. units),(arb.units)'] * len(self.normalized_data.data_vars))
            else:
                header += ' ,'+','.join([f"Energy Loss,{self.normalized_data[scan].attrs['run']}" for scan in self.normalized_data.data_vars])
                header += '\n'
                units = ' ,'+','.join(['(eV),(arb. units)'] * len(self.normalized_data.data_vars))
            units += '\n'
            header += units
            file.write(f"{header}\n")

            # Stack all x and y values as adjacent columns
            all_data = []
            for scan in self.normalized_data.data_vars:
                x_values = self.normalized_data[scan].sel(variable='x').values
                y_values = self.normalized_data[scan].sel(variable='y').values
                norm_values = self.normalized_data[scan].sel(variable='norm').values
                
                error_values = self.normalized_data[scan].sel(variable='error').values if save_errorbars else None

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





