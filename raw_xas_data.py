import numpy as np
import h5py
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Raw_XAS_Data(ABC):
    """
    Abstract base class for loading raw XAS data from different facilities.
    
    This class defines the common interface that all facility-specific
    implementations must follow.
    
    Parameters
    ----------
    runs : int or list of int
        Run number(s) to be extracted.
    filepaths : str or list of str, optional
        Path(s) to the data file(s). If not provided, must specify `folder`.
    folder : str, optional
        Path to the folder containing the data files.
    
    Attributes
    ----------
    runs : list of int
        List of run numbers to extract.
    filepaths : list of str
        List of file paths to the data files.
    folder : str or None
        Path to the folder containing the data files.
    data : dict
        Dictionary containing the extracted raw data for each run.
    motor_positions : dict
        Dictionary containing motor positions for each run.
    """
    
    def __init__(self, runs, filepaths=None, folder=None):
        """Initialize the Raw_XAS_Data object."""
        # Convert runs to list if needed
        self.runs = runs if isinstance(runs, list) else [runs]
        
        # Check for duplicates in runs
        if len(self.runs) != len(set(self.runs)):
            raise ValueError("Duplicate run numbers found in the provided runs.")
        
        # Validate that either filepaths or folder is provided (but not both)
        if filepaths is None and folder is None:
            raise ValueError("Either 'filepaths' or 'folder' must be provided.")
        if filepaths is not None and folder is not None:
            raise ValueError("Cannot provide both 'filepaths' and 'folder'. Please provide only one.")
        
        self.folder = folder
        
        # Get filepaths
        if folder is not None:
            if not os.path.isdir(folder):
                raise ValueError(f"The specified folder does not exist: {folder}")
            self.filepaths = self._get_filepaths_from_runs()
        else:
            self.filepaths = self._validate_filepaths(filepaths)
        
        # Initialize data storage
        self.ds = None
    
    def _validate_filepaths(self, filepaths):
        """Validate and format the provided filepaths."""
        if isinstance(filepaths, list):
            if len(filepaths) != len(self.runs):
                raise ValueError("If filepaths is a list, it must have the same length as runs.")
            if not all(isinstance(f, str) and os.path.isfile(f) for f in filepaths):
                raise ValueError("All elements in filepaths must be valid file paths.")
            return filepaths
        elif isinstance(filepaths, str):
            if not os.path.isfile(filepaths):
                raise ValueError("filepaths must be a valid file path when provided as a string.")
            return [filepaths] * len(self.runs)
        else:
            raise ValueError("filepaths must be either a list of file paths or a single file path string.")
    
    @abstractmethod
    def _get_filepaths_from_runs(self):
        """Retrieve file paths from run numbers. Must be implemented by child classes."""
        pass
    
    @abstractmethod
    def load_data(self):
        """Load XAS data from files. Must be implemented by child classes."""
        pass
    
    def get_run_data(self, run):
        """
        Get the data for a specific run.
        
        Parameters
        ----------
        run : int
            The run number.
        
        Returns
        -------
        xarray.DataArray or None
            The data array for the specified run, or None if not found.
        """
        if self.ds is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # Find the DataArray with the matching run number
        for var_name in self.ds.data_vars:
            if self.ds[var_name].attrs.get('run') == run:
                return self.ds[var_name]
        return None
    
    def get_run_motor_positions(self, run):
        """
        Get the motor positions for a specific run.
        
        Parameters
        ----------
        run : int
            The run number.
        
        Returns
        -------
        dict or None
            Dictionary of motor positions for the specified run, or None if not found.
        """
        data_array = self.get_run_data(run)
        if data_array is None:
            return None
        
        # Extract motor positions from attributes
        motor_positions = {}
        for key, value in data_array.attrs.items():
            if key not in ['filepath', 'run']:
                motor_positions[key] = value
        return motor_positions
    
    def print_summary(self):
        """Print a summary of the loaded data."""
        print(f"\n{'='*60}")
        print(f"Raw XAS Data Summary")
        print(f"{'='*60}")
        print(f"Facility: {self.__class__.__name__}")
        print(f"Number of runs: {len(self.runs)}")
        print(f"Runs: {self.runs}")
        if self.folder is not None:
            print(f"Data folder: {self.folder}")
        
        if self.ds is not None:
            print(f"Loaded data for {len(self.ds.data_vars)} run(s)")
            
            if len(self.ds.data_vars) > 0:
                print(f"\nData structure for each run:")
                first_var = list(self.ds.data_vars)[0]
                first_data = self.ds[first_var]
                print(f"  Shape: {first_data.shape}")
                print(f"  Variables: {list(first_data.coords['variable'].values)}")
                
                # Print motor positions from first run
                motor_attrs = {k: v for k, v in first_data.attrs.items() 
                              if k not in ['filepath', 'run']}
                if motor_attrs:
                    print(f"\nMotor positions available for all runs")
                    print(f"Example motor names (run {first_data.attrs.get('run')}):")
                    for i, motor in enumerate(list(motor_attrs.keys())[:5]):
                        print(f"  - {motor}")
                    if len(motor_attrs) > 5:
                        print(f"  ... and {len(motor_attrs) - 5} more")
        else:
            print("No data loaded yet. Please call load_data() first.")
        
        print(f"{'='*60}\n")

    def average_xas(self):
        """
        Average the normalized XAS data across all runs in the dataset.

        All runs are interpolated onto the energy axis of the first run before
        averaging. The result is stored as ``self.ds['avg']`` and also returned.

        Returns
        -------
        avg_array : xarray.DataArray
            DataArray with the averaged data, with variables
            ``['Energy (eV)', 'XAS (arb. units)', 'XAS_norm (arb. units)', 'i0 (arb. units)']``.
            Motor-position attributes are copied from the first run; a warning
            is printed for any attribute that differs by more than 0.1 between
            runs. The ``'run'`` attribute contains the list of averaged run
            numbers.

        Raises
        ------
        ValueError
            If no data is loaded or if ``xas_norm`` is not present (i.e.
            ``normalize_xas()`` has not been called).
        """
        if self.ds is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Check that normalization has been performed on at least one run
        first_var = list(self.ds.data_vars)[0]
        if 'XAS_norm (arb. units)' not in self.ds[first_var].coords['variable'].values:
            raise ValueError("Normalized XAS not found. Please run normalize_xas() first.")

        # Reference energy axis from the first run
        energy0 = self.ds[first_var].sel(variable='Energy (eV)').values

        xas_list      = []
        xas_norm_list = []
        i0_list       = []
        run_labels    = []

        for ii, var_name in enumerate(self.ds.data_vars):
            da = self.ds[var_name]
            energy   = da.sel(variable='Energy (eV)').values
            xas_norm = da.sel(variable='XAS_norm (arb. units)').values
            xas_raw  = da.sel(variable='XAS (arb. units)').values
            i0       = (da.sel(variable='i0 (arb. units)').values
                        if 'i0 (arb. units)' in da.coords['variable'].values
                        else np.full_like(energy, np.nan))

            # Interpolate onto common energy axis
            xas_norm_list.append(np.interp(energy0, energy, xas_norm))
            xas_list.append(np.interp(energy0, energy, xas_raw))
            i0_list.append(np.interp(energy0, energy, i0))
            run_labels.append(da.attrs.get('run', var_name))

            # Collect attributes from first run; warn on differences
            if ii == 0:
                ref_attrs = {k: v for k, v in da.attrs.items()
                             if k not in ('filepath', 'run')}
            else:
                for key, ref_val in ref_attrs.items():
                    cur_val = da.attrs.get(key)
                    if cur_val is None:
                        continue
                    try:
                        if abs(float(cur_val) - float(ref_val)) > 0.1:
                            print(f"Warning: attribute '{key}' differs between runs "
                                  f"(run {run_labels[0]}: {ref_val}, "
                                  f"run {da.attrs.get('run', var_name)}: {cur_val})")
                    except (TypeError, ValueError):
                        if str(cur_val) != str(ref_val):
                            print(f"Warning: attribute '{key}' differs between runs "
                                  f"(run {run_labels[0]}: {ref_val}, "
                                  f"run {da.attrs.get('run', var_name)}: {cur_val})")

        avg_data = np.column_stack([
            energy0,
            np.mean(xas_list,      axis=0),
            np.mean(xas_norm_list, axis=0),
            np.mean(i0_list,       axis=0),
        ])

        avg_array = xr.DataArray(
            avg_data,
            dims=['points', 'variable'],
            coords={
                'points':   np.arange(avg_data.shape[0]),
                'variable': ['Energy (eV)', 'XAS (arb. units)', 'XAS_norm (arb. units)', 'i0 (arb. units)']
            }
        )

        # Copy attributes from first run and tag with averaged run list
        for k, v in ref_attrs.items():
            avg_array.attrs[k] = v
        avg_array.attrs['run'] = run_labels
        avg_array.attrs['filepath'] = self.ds[first_var].attrs.get('filepath', '')

        self.ds['avg'] = avg_array
        print(f"Averaged {len(run_labels)} run(s): {run_labels}")
        return avg_array


    def plot_post_edge_manipulation(self, run=None):
        """
        Plot the post-edge manipulation for a specific run.

        Parameters
        ----------
        run : int, optional
            The run number to plot. If None, the first run in the dataset is used.
        """
        if self.ds is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Select the DataArray to plot
        if run is not None:
            data_array = self.get_run_data(run)
            if data_array is None:
                raise ValueError(f"Run {run} not found in dataset.")
        else:
            first_var = list(self.ds.data_vars)[0]
            data_array = self.ds[first_var]

        # Check that normalization has been run and post-edge slope was removed
        attrs = data_array.attrs
        if "coeffs_pre_edge" not in attrs or attrs.get("coeffs_pre_edge") is None:
            raise ValueError("Pre-edge coefficients not found. Please run normalize_xas with subtract_pre_edge=True first.")
        if "coeffs_post_edge" not in attrs or attrs.get("coeffs_post_edge") is None:
            raise ValueError("Post-edge coefficients not found. Please run normalize_xas with remove_post_edge_slope=True first.")
        if "history_actions" not in attrs or "Removed post-edge slope" not in attrs["history_actions"]:
            raise ValueError("'Removed post-edge slope' not found in history. Please run normalize_xas with remove_post_edge_slope=True first.")

        # Extract data from the DataArray
        energy = data_array.sel(variable='Energy (eV)').values
        xas_norm = data_array.sel(variable='XAS_norm (arb. units)').values

        # Reconstruct the pre-normalized XAS (before post-edge removal) by re-applying pre-edge step
        coeffs_pre_edge = np.array(attrs["coeffs_pre_edge"])
        coeffs_post_edge = np.array(attrs["coeffs_post_edge"])
        energy_edge = attrs.get("energy_edge")
        step = attrs.get("step", None)

        # Reconstruct XAS after pre-edge subtraction but before post-edge slope removal
        # by re-adding the post-edge slope correction
        indices_after_edge = np.where(energy > energy_edge)[0]
        diff_curve = np.zeros_like(energy)
        diff_curve[indices_after_edge] = (
            np.polyval(coeffs_post_edge, energy[indices_after_edge]) -
            np.polyval(coeffs_post_edge, energy_edge)
        )
        divisor = step if step is not None else 1.0
        xas_before_post_edge = xas_norm + diff_curve / divisor

        indices_before_edge = np.where(energy <= energy_edge)[0]

        run_label = attrs.get('run', first_var if run is None else run)
        plt.figure(figsize=(6, 5))
        plt.plot(energy, xas_before_post_edge, label="XAS after pre-edge removal", color='k')
        plt.plot(energy[indices_after_edge],
                 np.polyval(coeffs_post_edge, energy[indices_after_edge]) / divisor,
                 label="Post-edge fit", color="#ed6262")
        plt.plot(energy[indices_before_edge],
                 np.polyval(coeffs_post_edge, energy[indices_before_edge]) / divisor,
                 label="Post-edge fit (extrapolated)", color="#ed6262", linestyle='dashed')
        plt.plot(energy, xas_norm, label="XAS with post-edge slope removed", color='#B3CDE3')

        # Mark the edge energy on the plot
        idx_edge = int(np.argmin(np.abs(energy - energy_edge)))
        y_edge = xas_norm[idx_edge]
        plt.scatter(energy[idx_edge], y_edge, color='red', marker='o', s=50, zorder=10,
                    label=f"E0 = {energy_edge:.2f} eV")
        plt.axvline(energy_edge, color='red', linestyle='--', linewidth=0.8, zorder=5)
        # Load ranges from DataArray attrs (try several possible locations)
        range_pre = data_array.attrs.get('range_pre_edge')
        range_post = data_array.attrs.get('range_post_edge')

        pre_interval = range_pre
        post_interval = range_post

        # Draw semi-transparent gray rectangles for the intervals (behind data curves)
        if pre_interval is not None:
            plt.axvspan(pre_interval[0], pre_interval[1], color='gray', alpha=0.25, zorder=0,
                        label='pre-edge range')

        if post_interval is not None:
            plt.axvspan(post_interval[0], post_interval[1], color='gray', alpha=0.25, zorder=0,
                        label='post-edge range')

        # Avoid duplicate legend entries if both rectangles added
        if pre_interval is not None and post_interval is not None:
            # Consolidate duplicate labels in legend
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = {}
            new_handles = []
            new_labels = []
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = True
                    new_handles.append(h)
                    new_labels.append(l)
            plt.legend(new_handles, new_labels)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (arb. units)")
        plt.title(f"Post-edge manipulation — run {run_label}")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _to_serializable(obj):
        """Recursively convert an object to a JSON-serializable form."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [Raw_XAS_Data._to_serializable(i) for i in obj]
        if isinstance(obj, dict):
            return {str(k): Raw_XAS_Data._to_serializable(v) for k, v in obj.items()}
        try:
            import json as _json
            _json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def _resolve_output_path(self, run, folder_out, filename_out, suffix):
        """
        Resolve the output file path for saving data.

        Parameters
        ----------
        run : int or None
            Run number. If None, the first run in the dataset is used.
        folder_out : str or None
            Output folder. If None, the source file's folder is used.
        filename_out : str or None
            Output filename. If None, a default name is generated.
        suffix : str
            File extension including the dot, e.g. '.dat' or '.json'.

        Returns
        -------
        out_file : Path
            Resolved output file path.
        data_array : xarray.DataArray
            The DataArray for the selected run.
        """
        from pathlib import Path

        if self.ds is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Select the DataArray
        if run is not None:
            data_array = self.get_run_data(run)
            if data_array is None:
                raise ValueError(f"Run {run} not found in dataset.")
        else:
            first_var = list(self.ds.data_vars)[0]
            data_array = self.ds[first_var]

        # Check that normalization has been performed
        if 'XAS_norm (arb. units)' not in data_array.coords['variable'].values:
            raise ValueError("Normalized XAS data not found. Please run normalize_xas() first.")

        # Build output path
        source_filepath = data_array.attrs.get('filepath', '')
        run_label = data_array.attrs.get('run', 'unknown')

        if source_filepath:
            _p = Path(source_filepath)
            default_stem = f"{_p.stem}_run{run_label}_processed"
            default_folder = str(_p.parent)
        else:
            default_stem = f"run{run_label}_processed"
            default_folder = str(Path.cwd())

        if folder_out is not None and filename_out is None:
            out_file = Path(folder_out) / (default_stem + suffix)
        elif filename_out is not None and folder_out is None:
            out_file = Path(default_folder) / filename_out
        elif filename_out is not None and folder_out is not None:
            out_file = Path(folder_out) / filename_out
        else:
            out_file = Path(default_folder) / (default_stem + suffix)

        return out_file, data_array

    def save_dat_file(self, run=None, folder_out=None, filename_out=None,
                      additional_metadata={}, save_multiple_files=False):
        """
        Save the XAS data to one or more .dat text files.

        Parameters
        ----------
        run : int, optional
            Run number to save. Only used when ``save_multiple_files=False``.
            If None, the first run in the dataset is used.
        folder_out : str or Path, optional
            Output folder. Defaults to the source file's folder.
        filename_out : str, optional
            Output filename. When ``save_multiple_files=True`` and this is not
            provided, names are generated automatically for each run. When
            ``save_multiple_files=False`` and not provided, a default name is
            generated from the source filepath.
        additional_header : str, optional
            Extra text prepended to the file header (before the '#' lines).
        save_multiple_files : bool, optional
            If True, one .dat file is saved per DataArray in ``self.ds``.
            If False (default), all runs are stacked as adjacent columns in a
            single file (only possible when all runs share the same number of
            points).

        Returns
        -------
        out_files : Path or list of Path
            Path (or list of paths) to the saved file(s).
        """
        from pathlib import Path

        if self.ds is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        def _build_header(attrs, additional_metadata):
            header = ""
            history = attrs.get("history_actions", None)
            if history:
                header += "# processing history: " + " | ".join(str(h) for h in history) + "\n"
            skip_attrs = {'filepath', 'run', 'history_actions'}
            for k, v in attrs.items():
                if k in skip_attrs:
                    continue
                header += f"# {k}: {self._to_serializable(v)}\n"
            for k, v in additional_metadata.items():
                header += f"# {k}: {self._to_serializable(v)}\n"
            return header

        def _extract_columns(da):
            """Return (energy, y, i0) arrays from a DataArray."""
            variables = list(da.coords['variable'].values)
            energy = da.sel(variable='Energy (eV)').values
            y = (da.sel(variable='XAS_norm (arb. units)').values
                 if 'XAS_norm (arb. units)' in variables
                 else da.sel(variable='XAS (arb. units)').values)
            y_label = 'XAS_norm (arb. units)' if 'XAS_norm (arb. units)' in variables else 'XAS (arb. units)'
            i0 = (da.sel(variable='i0 (arb. units)').values
                  if 'i0 (arb. units)' in variables
                  else np.full_like(energy, np.nan))
            return energy, y, y_label, i0

        def _resolve_folder(da):
            """Return the best output folder for a DataArray."""
            fp = da.attrs.get('filepath', '')
            return str(Path(fp).parent) if fp else str(Path.cwd())

        # ------------------------------------------------------------------ #
        # MULTIPLE FILES                                                       #
        # ------------------------------------------------------------------ #
        if save_multiple_files:
            out_files = []
            for var_name in self.ds.data_vars:
                da = self.ds[var_name]
                run_label = da.attrs.get('run', var_name)

                # Resolve output path for this run
                if folder_out is not None and filename_out is None:
                    fp = da.attrs.get('filepath', '')
                    stem = Path(fp).stem if fp else f"run_{run_label}"
                    out_file = Path(folder_out) / f"{stem}_processed.dat"
                elif filename_out is not None:
                    # Use given name with run suffix to avoid overwriting
                    base = Path(filename_out)
                    out_file = Path(folder_out or _resolve_folder(da)) / \
                               f"{base.stem}_run_{run_label}{base.suffix or '.dat'}"
                else:
                    fp = da.attrs.get('filepath', '')
                    stem = Path(fp).stem if fp else f"run_{run_label}"
                    out_file = Path(_resolve_folder(da)) / f"{stem}_run_{run_label}_processed.dat"

                energy, y, y_label, i0 = _extract_columns(da)
                data = np.column_stack((energy, y, i0))

                header = _build_header(da.attrs, additional_metadata)
                header += f"# run: {run_label}\n"
                header += f"# energy \t {y_label} \t i0"

                out_file.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(out_file, data, header=header, fmt="%18.9e",
                           delimiter="\t", comments="")
                print(f"  Data saved to: {out_file}")
                out_files.append(out_file)

            return out_files

        # ------------------------------------------------------------------ #
        # SINGLE FILE – all runs stacked as columns                           #
        # ------------------------------------------------------------------ #
        else:
            # Collect data from all runs (or just the selected one)
            if run is not None:
                da = self.get_run_data(run)
                if da is None:
                    raise ValueError(f"Run {run} not found in dataset.")
                vars_to_save = {str(run): da}
            else:
                vars_to_save = {var: self.ds[var] for var in self.ds.data_vars}

            # Check all arrays have the same length
            lengths = [self.ds[v].shape[0] for v in vars_to_save]
            if len(set(lengths)) > 1:
                raise ValueError(
                    "Cannot stack runs into a single file: DataArrays have "
                    f"different lengths: {dict(zip(vars_to_save.keys(), lengths))}. "
                    "Use save_multiple_files=True instead."
                )

            columns = []
            col_labels = []
            first_da = next(iter(vars_to_save.values()))

            for var_name, da in vars_to_save.items():
                run_label = da.attrs.get('run', var_name)
                energy, y, y_label, i0 = _extract_columns(da)
                columns.extend([energy, y, i0])
                col_labels.append(f"energy_run{run_label}\t{y_label}_run{run_label}\ti0_run{run_label}")

            data = np.column_stack(columns)

            # Resolve output path
            if folder_out is not None and filename_out is None:
                # Build filename using stem of first file plus all run labels joined by "_"
                runs_part = "_".join(
                    "".join(c if c.isalnum() else "_" for c in str(da.attrs.get('run', var_name)))
                    for var_name, da in vars_to_save.items()
                )
                fp = first_da.attrs.get('filepath', '')
                stem0 = Path(fp).stem if fp else "xas_data"
                stem = f"{stem0}_runs_{runs_part}"
                out_file = Path(folder_out) / f"{stem}_processed.dat"
            elif filename_out is not None:
                out_file = Path(folder_out or _resolve_folder(first_da)) / filename_out
            else:
                fp = first_da.attrs.get('filepath', '')
                stem = Path(fp).stem if fp else "xas_data"
                out_file = Path(_resolve_folder(first_da)) / f"{stem}_processed.dat"

            # Build header from first DataArray's attributes
            header = _build_header(first_da.attrs, additional_metadata)
            header += "# " + "\t".join(col_labels)

            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_file, data, header=header, fmt="%18.9e",
                       delimiter="\t", comments="")
            print(f"Data saved to: {out_file}")
            return out_file

    def save_json_file(self, folder_out, filename_out, 
                       run=None, 
                       additional_metadata={}, save_multiple_files=False):
        """
        Save the processed XAS data to one or more JSON files.

        Each file contains all data variables as top-level arrays and all
        DataArray attributes under a ``'metadata'`` key.

        Parameters
        ----------
        run : int, optional
            Run number to save. Only used when ``save_multiple_files=False``.
            If None, the first run in the dataset is used.
        folder_out : str or Path, optional
            Output folder. Defaults to the source file's folder.
        filename_out : str, optional
            Output filename. When ``save_multiple_files=True`` and not provided,
            names are generated automatically per run. When
            ``save_multiple_files=False`` and not provided, a default name is
            generated from the source filepath.
        additional_metadata : dict, optional
            Extra metadata entries merged into the ``'metadata'`` section.
        save_multiple_files : bool, optional
            If True, one JSON file is saved per DataArray in ``self.ds``.
            If False (default), a single JSON file is saved for the selected
            run (or the first run if ``run`` is None).

        Returns
        -------
        out_file : Path or list of Path
            Path (or list of paths) to the saved file(s).
        """
        import json
        from pathlib import Path

        if self.ds is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        def _resolve_folder(da):
            fp = da.attrs.get('filepath', '')
            return str(Path(fp).parent) if fp else str(Path.cwd())

        def _build_payload(da, additional_metadata):
            payload = {}
            metadata = {k: self._to_serializable(v) for k, v in da.attrs.items()}
            if additional_metadata:
                add_meta = self._to_serializable(additional_metadata)
                if isinstance(add_meta, dict):
                    metadata.update(add_meta)
                else:
                    metadata["additional_metadata"] = add_meta
            payload["metadata"] = metadata

            for var in da.coords['variable'].values:
                payload[str(var)] = self._to_serializable(da.sel(variable=var).values)
            return payload

        def _write(out_file, payload):
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"  Data saved to JSON file: {out_file}")

        # ------------------------------------------------------------------ #
        # MULTIPLE FILES                                                       #
        # ------------------------------------------------------------------ #
        if save_multiple_files:
            out_files = []
            for var_name in self.ds.data_vars:
                da = self.ds[var_name]
                run_label = da.attrs.get('run', var_name)

                if folder_out is not None and filename_out is None:
                    fp = da.attrs.get('filepath', '')
                    stem = Path(fp).stem if fp else f"run_{run_label}"
                    out_file = Path(folder_out) / f"{stem}_processed.json"
                elif filename_out is not None:
                    base = Path(filename_out)
                    out_file = Path(folder_out or _resolve_folder(da)) / \
                               f"{base.stem}_run_{run_label}{base.suffix or '.json'}"
                else:
                    fp = da.attrs.get('filepath', '')
                    stem = Path(fp).stem if fp else f"run_{run_label}"
                    out_file = Path(_resolve_folder(da)) / f"{stem}_run_{run_label}_processed.json"

                _write(out_file, _build_payload(da, additional_metadata))
                out_files.append(out_file)

            return out_files

        # ------------------------------------------------------------------ #
        # SINGLE FILE                                                          #
        # ------------------------------------------------------------------ #
        else:
            if run is not None:
                da = self.get_run_data(run)
                if da is None:
                    raise ValueError(f"Run {run} not found in dataset.")
            else:
                total_dict_json = {}
                for var_name in self.ds.data_vars:
                    da = self.ds[var_name]
                    dict_var = _build_payload(da, additional_metadata)
                    total_dict_json[var_name] = dict_var

            out_file = Path(folder_out or _resolve_folder(da)) / filename_out

            _write(out_file, total_dict_json)
            return out_file
        
    def save_hdf5_file(self, folder_out, filename_out, run=None,
                       variable_names = [],
                       additional_metadata={}, save_multiple_files=False,
                       metadata_to_save = []):
        """
        Save the processed XAS data to one or more HDF5 files.
        """

      
        # Build a new dataset copying all data but keeping only selected metadata keys
        if self.ds is None:
            raise ValueError("No dataset provided.")

        # if variable_names and units_names:
        #     if len(units_names) != len(variable_names):
        #         raise ValueError("Length of variable_names and units_names must match.")

        # Normalize metadata_to_save to a set of keys
        if metadata_to_save is None:
            keep_keys = set()
        elif isinstance(metadata_to_save, (list, tuple, set)):
            keep_keys = set(metadata_to_save)
        else:
            keep_keys = {str(metadata_to_save)}

        new_ds = xr.Dataset()

        for var_name in self.ds.data_vars:
            da = self.ds[var_name]
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
                    print(f"Warning: none of requested variable_names {vars_requested} found in DataArray; copying all variables.")
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
            new_ds[var_name] = new_da

        # also filter dataset-level attributes if present
        new_ds.attrs = {k: v for k, v in getattr(self.ds, "attrs", {}).items() if k in keep_keys}
        for data_var in self.ds.data_vars:
            da = new_ds[data_var]
            for k, v in additional_metadata.items():
                da.attrs[k] = v

        # new_ds.attrs['units'] = units_names
        filename = os.path.join(folder_out, filename_out)
        new_ds.to_netcdf(filename, engine='h5netcdf')
        print(f"Dataset saved to {filename}")


    def save_csv_file(self, folder_out, filename_out,
                        variable_names, units_names,
                      additional_metadata={}):
        """
        Save the processed XAS data to a CSV file.
        """

        filepath = os.path.join(folder_out, filename_out)
        header = ''
        first_var = list(self.ds.data_vars)[0]
        tot_metadata = [meta for meta in self.ds[first_var].attrs if meta not in ['filepath', 'run', 
                                                                                  'history_actions',
                                                                                  'coeffs_pre_edge', 'coeffs_post_edge',
                                                                                  'range_pre_edge', 'range_post_edge', 'energy_edge', 'step',
                                                                                  'normalization_method']]
        # tot_metadata += list(additional_metadata.keys())

        for metadata in tot_metadata:

            header_parts = ['#'+metadata]
            header_parts_1 = [' ' for _ in range(len(self.ds.data_vars))]
            header_parts_2 = []

            header_parts_2 += [(f"{self.ds[data_var].attrs.get(metadata):.2f}" if isinstance(self.ds[data_var].attrs.get(metadata), (int, float)) else str(self.ds[data_var].attrs.get(metadata)))
                        for data_var in self.ds.data_vars
                    ]

            for idx in range(1, len(self.ds.data_vars) * 2 + 1):
                if idx % len(variable_names) == 1:
                    header_parts.append(header_parts_1[(idx - 1) // 2])
                else:
                    header_parts.append(header_parts_2[(idx - 1) // 2])

            header += ','.join(header_parts)  # Repeat each motor value twice
            header += '\n'

        for k, v in additional_metadata.items():
            header_parts = [f"#{k}"]
            header_parts_1 = [' ' for _ in range(len(self.ds.data_vars))]
            header_parts_2 = []
            header_parts_2 += [(f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
                        for _ in self.ds.data_vars
                    ]
            for idx in range(1, len(self.ds.data_vars) * 2 + 1):
                if idx % len(variable_names) == 1:
                    header_parts.append(header_parts_1[(idx - 1) // 2])
                else:
                    header_parts.append(header_parts_2[(idx - 1) // 2])
            header += ','.join(header_parts) 
            header += '\n'
        

        # Create a DataFrame to store all scans
        data_frames = []
        # Stack all x and y values as adjacent columns
        all_data = []
        # Get the variable names in the dataset
        data_array_names = list(self.ds.data_vars)
        for scan in data_array_names:
            for var_name in variable_names:
                if var_name not in self.ds[scan].coords['variable'].values:
                    raise ValueError(f"Variable '{var_name}' not found in DataArray '{scan}'.")
                values = self.ds[scan].sel(variable=var_name).values
                all_data.append(values[:,np.newaxis])

        header += ' ,'
        units = ' ,'
        for scan in data_array_names:
            for var_name, unit in zip(variable_names, units_names):
                # Write a line of "Energy Loss" and {scan} alternating
                header += f"{var_name}_{scan},"  # Fix the formatting here
                units += f"{unit},"
        header += '\n'
        units += '\n'
        header += units

        # Concatenate all data along the second axis
        concatenated_data = np.concatenate(all_data, axis=1)

        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f"{header}\n")
            # Write the concatenated data to the file
            for row in concatenated_data:
                file.write(' ,'+','.join(map(str, row)) + '\n')

        print(f"Dataset saved to {filepath}")
    
        


    def normalize_xas(self,
                      normalization_method='i0 (arb. units)',
                      poly_order_i0=None,
                      subtract_pre_edge=True,
                      pre_edge_params={},
                      energy_edge=None,
                      remove_post_edge_slope=False,
                      normalize_by_mu_e0=False,
                      post_edge_params={}):
        """
        Normalize the XAS data for all runs in the dataset.
        
        This method normalizes the XAS data by dividing by the I0 data (or other methods)
        and optionally subtracting pre-edge and post-edge backgrounds.
        
        Parameters
        ----------
        normalization_method : str, optional
            The method to use for normalization. Options are:
            - 'i0 (arb. units)': Divide by I0 (default)
            - 'i0_fit': Divide by polynomial fit of I0
            - 'baseline': Divide by baseline value from pre-edge region
            Default is 'i0 (arb. units)'.
        poly_order_i0 : int, optional
            Order of polynomial to fit to I0 data if normalization_method is 'i0_fit'.
        subtract_pre_edge : bool, optional
            Whether to subtract a background from the pre-edge region. Default is True.
        pre_edge_params : dict, optional
            Dictionary containing parameters for pre-edge background subtraction:
            - 'range_pre_edge': tuple (start_index, end_index) for pre-edge region
            - 'poly_order_pre_edge': int, polynomial order (default 1)
        energy_edge : float, optional
            Energy of the absorption edge, required if normalize_by_mu_e0 or
            remove_post_edge_slope is True.
        remove_post_edge_slope : bool, optional
            Whether to remove slope after the edge. Default is False.
        normalize_by_mu_e0 : bool, optional
            Whether to normalize by mu(E0). Default is False.
        post_edge_params : dict, optional
            Dictionary containing parameters for post-edge manipulation:
            - 'range_post_edge': tuple (start_energy, end_energy) for post-edge region
            - 'poly_order_post_edge': int, polynomial order (default 1)
        
        Returns
        -------
        xarray.Dataset
            The dataset with normalized XAS data added.
        """
        if self.ds is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        if normalization_method not in ['i0 (arb. units)', 'i0+baseline', 'i0_fit']:
            raise ValueError("Invalid normalization method. Choose 'i0 (arb. units)', 'i0_fit', or 'baseline'.")
        
        updated_arrays = {}

        for var_name in self.ds.data_vars:
            data_array = self.ds[var_name]

            # Skip the averaged DataArray
            run_attr = data_array.attrs.get('run')
            if isinstance(run_attr, list) or var_name == 'avg':
                updated_arrays[var_name] = data_array
                continue

            # Extract data
            energy = data_array.sel(variable='Energy (eV)').values
            xas = data_array.sel(variable='XAS (arb. units)').values
            i0 = data_array.sel(variable='i0 (arb. units)').values if 'i0 (arb. units)' in data_array.coords['variable'].values else None
            
            # Initialize processing metadata for this run
            processing_metadata = {
                "normalization_method": normalization_method,
                "energy_edge": energy_edge,
                "step": None,
                "coeffs_pre_edge": None,
                "range_pre_edge": None,
                "coeffs_post_edge": None,
                "range_post_edge": None,
                "history_actions": ["raw xas"]
            }
            
            # Perform normalization
            if normalization_method == 'i0 (arb. units)':
                if i0 is None:
                    raise ValueError(f"I0 data not available for run {data_array.attrs.get('run')}. Cannot normalize by I0.")
                xas_norm = xas / i0
                processing_metadata["history_actions"].append("Normalized by i0")
            
            elif normalization_method == 'i0_fit':
                if i0 is None:
                    raise ValueError(f"I0 data not available for run {data_array.attrs.get('run')}.")
                if poly_order_i0 is None:
                    raise ValueError("poly_order_i0 must be provided for i0_fit normalization method.")
                poly_coeffs_i0 = np.polyfit(energy, i0, poly_order_i0)
                i0_fit = np.polyval(poly_coeffs_i0, energy)
                xas_norm = xas / i0_fit
                processing_metadata["history_actions"].append("Normalized by i0 fitted to polynomial")
            
            elif normalization_method == 'i0+baseline':
                if i0 is None:
                    raise ValueError(f"I0 data not available for run {data_array.attrs.get('run')}. Cannot normalize by I0.")
                xas_norm = xas / i0
                range_post_edge = post_edge_params.get("range_post_edge")
                # Convert energy bounds to indices if needed and validate
                if range_post_edge is not None:
                    try:
                        start, end = range_post_edge
                    except Exception:
                        raise ValueError("range_post_edge must be a tuple (start, end) of energies or indices.")

                    n1 = np.searchsorted(energy, start)
                    n2 = np.searchsorted(energy, end)

                    # Clamp to valid index range
                    n1 = max(0, min(n1, len(energy)))
                    n2 = max(0, min(n2, len(energy)))

                    if n1 >= n2:
                        raise ValueError("Invalid range_post_edge: start must be less than end after conversion to indices.")

                    range_baseline = (n1, n2)

                else:
                    raise ValueError("range_post_edge must be provided in post_edge_params for baseline normalization.")

                baseline = np.mean(xas_norm[range_baseline[0]:range_baseline[1]])
                xas_norm = xas_norm / baseline
                processing_metadata["history_actions"].append("Normalized by baseline")
            
            xas_processed = xas_norm.copy()
            
            # Subtract pre-edge background
            if subtract_pre_edge:
                range_pre_edge = pre_edge_params.get("range_pre_edge")
                if range_pre_edge is None:
                    raise ValueError("range_pre_edge must be provided in pre_edge_params.")
                poly_order_pre_edge = pre_edge_params.get("poly_order_pre_edge", 1)
                n1_pre = np.searchsorted(energy, range_pre_edge[0])
                n2_pre = np.searchsorted(energy, range_pre_edge[1])
                if n1_pre >= n2_pre:
                    raise ValueError(f"Invalid pre-edge range for run {data_array.attrs.get('run')}.")
                xas_processed, coeffs_pre_edge = self._remove_pre_edge(
                    energy, xas_processed, n1_pre, n2_pre, poly_order_pre_edge
                )
                processing_metadata["history_actions"].append("Removed pre-edge")
                processing_metadata["coeffs_pre_edge"] = coeffs_pre_edge.tolist()
                processing_metadata["range_pre_edge"] = range_pre_edge
                
                if remove_post_edge_slope:
                    range_post_edge = post_edge_params.get("range_post_edge")
                    if range_post_edge is None:
                        raise ValueError("range_post_edge must be provided in post_edge_params.")
                    if energy_edge is None:
                        raise ValueError("energy_edge must be provided to remove post-edge slope.")
                    poly_order_post_edge = post_edge_params.get("poly_order_post_edge", 1)
                    xas_processed, coeffs_post_edge = self._remove_post_edge_slope(
                        energy, xas_processed, range_post_edge,
                        poly_order_post_edge, coeffs_pre_edge, energy_edge
                    )
                    processing_metadata["history_actions"].append("Removed post-edge slope")
                    processing_metadata["coeffs_post_edge"] = coeffs_post_edge.tolist()
                    processing_metadata["range_post_edge"] = range_post_edge
                
                if normalize_by_mu_e0:
                    if energy_edge is None:
                        raise ValueError("energy_edge must be provided to normalize by mu(E0).")
                    range_post_edge = post_edge_params.get("range_post_edge")
                    if range_post_edge is None:
                        raise ValueError("range_post_edge must be provided in post_edge_params.")
                    poly_order_post_edge = post_edge_params.get("poly_order_post_edge", 1)
                    xas_processed, step = self._normalize_by_mu_e0(
                        energy, xas_processed, energy_edge,
                        coeffs_pre_edge, range_post_edge, poly_order_post_edge
                    )
                    processing_metadata["history_actions"].append("Normalized by mu(E0)")
                    processing_metadata["step"] = float(step)

            # Build a brand-new DataArray with xas_norm appended as a variable
            base_vars = [v for v in data_array.coords['variable'].values if v != 'XAS_norm (arb. units)']
            new_variables = base_vars + ['XAS_norm (arb. units)']
            base_data = np.stack(
                [data_array.sel(variable=v).values for v in base_vars] + [xas_processed],
                axis=1
            )

            updated_array = xr.DataArray(
                base_data,
                dims=['points', 'variable'],
                coords={
                    'points': np.arange(base_data.shape[0]),
                    'variable': new_variables
                },
                attrs={**data_array.attrs, **processing_metadata}
            )

            updated_arrays[var_name] = updated_array

        # Rebuild the dataset from scratch — in-place assignment on xr.Dataset
        # does not work when the 'variable' coordinate changes size.
        self.ds = xr.Dataset(updated_arrays)
        self.processing_metadata = processing_metadata
        
        n_runs = sum(1 for v in self.ds.data_vars if v != 'avg')
        print(f"Successfully normalized {n_runs} run(s).")
        return self.ds
    
    @staticmethod
    def _remove_pre_edge(energy, xas, n1, n2, poly_order_pre_edge=1):
        """
        Remove pre-edge background by fitting and subtracting a polynomial.
        
        Parameters
        ----------
        energy : numpy.ndarray
            Energy values.
        xas : numpy.ndarray
            XAS data.
        n1 : int
            Start index for pre-edge region.
        n2 : int
            End index for pre-edge region.
        poly_order_pre_edge : int, optional
            Order of polynomial to fit. Default is 1 (linear).
        
        Returns
        -------
        xas_corrected : numpy.ndarray
            XAS data with pre-edge background removed.
        coeffs : numpy.ndarray
            Polynomial coefficients.
        """
        if n1 == n2:
            raise ValueError("Pre-edge range contains no data points.")
        
        # Fit polynomial to pre-edge region
        coeffs = np.polyfit(energy[n1:n2], xas[n1:n2], poly_order_pre_edge)
        pre_edge_fit = np.polyval(coeffs, energy)
        
        # Subtract from entire spectrum
        xas_corrected = xas - pre_edge_fit
        
        return xas_corrected, coeffs
    
    @staticmethod
    def _remove_post_edge_slope(energy, xas, range_post_edge, 
                                poly_order_post_edge, coeffs_pre_edge, energy_edge):
        """
        Remove slope in the post-edge region.
        
        Parameters
        ----------
        energy : numpy.ndarray
            Energy values.
        xas : numpy.ndarray
            XAS data.
        range_post_edge : tuple
            (start_energy, end_energy) for post-edge region.
        poly_order_post_edge : int
            Order of polynomial to fit.
        coeffs_pre_edge : numpy.ndarray
            Pre-edge polynomial coefficients (not used in current implementation).
        energy_edge : float
            Energy of the absorption edge.
        
        Returns
        -------
        xas_no_slope : numpy.ndarray
            XAS data with post-edge slope removed.
        coeffs_step : numpy.ndarray
            Post-edge polynomial coefficients.
        """
        n1_step = np.searchsorted(energy, range_post_edge[0])
        n2_step = np.searchsorted(energy, range_post_edge[1])
        
        # Fit polynomial to post-edge region
        x_fit = energy[n1_step:n2_step]
        y_fit = xas[n1_step:n2_step]
        coeffs_step = np.polyfit(x_fit, y_fit, poly_order_post_edge)
        
        # Find indices after edge
        indices_after_edge = np.where(energy > energy_edge)[0]
        
        if len(indices_after_edge) > 0:
            # Remove constant term from step polynomial
            coeffs_step_no_const = coeffs_step.copy()
            coeffs_step_no_const[-1] = 0
            
            # Calculate difference curve after edge
            diff_curve = np.zeros_like(energy)
            diff_curve[indices_after_edge] = (
                np.polyval(coeffs_step, energy[indices_after_edge]) - 
                np.polyval(coeffs_step, energy_edge)
            )
            
            # Subtract from XAS
            xas_no_slope = xas - diff_curve
        else:
            xas_no_slope = xas
        
        return xas_no_slope, coeffs_step
    
    @staticmethod
    def _normalize_by_mu_e0(energy, xas, energy_edge, coeffs_pre_edge,
                           range_post_edge, poly_order_post_edge):
        """
        Normalize XAS by the step height at the edge.
        
        Parameters
        ----------
        energy : numpy.ndarray
            Energy values.
        xas : numpy.ndarray
            XAS data.
        energy_edge : float
            Energy of the absorption edge.
        coeffs_pre_edge : numpy.ndarray
            Pre-edge polynomial coefficients (not used in current implementation).
        range_post_edge : tuple
            (start_energy, end_energy) for post-edge region.
        poly_order_post_edge : int
            Order of polynomial to fit to post-edge.
        
        Returns
        -------
        xas_normalized : numpy.ndarray
            XAS data normalized by step height.
        step : float
            Step height at the edge.
        """
        n1_step = np.searchsorted(energy, range_post_edge[0])
        n2_step = np.searchsorted(energy, range_post_edge[1])
        
        # Fit polynomial to post-edge region
        x_fit = energy[n1_step:n2_step]
        y_fit = xas[n1_step:n2_step]
        coeffs_post_edge = np.polyfit(x_fit, y_fit, poly_order_post_edge)
        
        # Calculate step as value of post-edge fit at edge energy
        step = np.polyval(coeffs_post_edge, energy_edge)
        xas_normalized = xas / step
        
        return xas_normalized, step


class ESRF_XAS_Data(Raw_XAS_Data):
    """
    Class for loading XAS data from ESRF (European Synchrotron Radiation Facility).
    
    ESRF data is stored in HDF5 (.nxs) files with structure:
    - Motor positions: <run>.1/instrument/positioners/<motor_name>
    - Measurement data: <run>.1/measurement/<dataset_name>
    
    Examples
    --------
    >>> # Load with explicit filepaths
    >>> xas = ESRF_XAS_Data(runs=[1, 2, 3], filepaths='path/to/file.nxs')
    >>> xas.load_data()
    
    >>> # Load by searching folder
    >>> xas = ESRF_XAS_Data(runs=[1, 2, 3], folder='path/to/data')
    >>> xas.load_data()
    """
    
    def _get_filepaths_from_runs(self):
        """
        Retrieve ESRF file paths from run numbers.
        
        Searches for files matching ESRF naming conventions.
        """
        raise NotImplementedError(
            "ESRF filepath retrieval from run numbers is not yet implemented. "
            "Please implement based on your specific ESRF file naming convention."
        )

    
    def load_data(self):
        """Load XAS data from ESRF HDF5 files into an xarray Dataset."""
        print(f"Loading ESRF data from {len(self.filepaths)} file(s)...")
        
        self.ds = xr.Dataset()
        
        for ii, (filepath, run) in enumerate(zip(self.filepaths, self.runs)):
            print(f"  Processing run {run} from {os.path.basename(filepath)}")
            
            # Extract motor positions
            motor_pos = self._extract_motor_positions(filepath, run)
            
            # Extract XAS data
            data = self._extract_xas(filepath, run)
            
            if data is not None:
                # Create xarray DataArray with the data
                data_xr = xr.DataArray(
                    data,
                    dims=['points', 'variable'],
                    coords={
                        'points': np.arange(data.shape[0]),
                        'variable': ['Energy (eV)', 'XAS (arb. units)', 'i0 (arb. units)', 'mir']
                    }
                )
                
                # Add motor positions as attributes
                if motor_pos is not None:
                    for k, v in motor_pos.items():
                        data_xr.attrs[k] = v
                
                # Add filepath and run as attributes
                data_xr.attrs['filepath'] = filepath
                data_xr.attrs['run'] = run
                
                # Add to the dataset with a unique name
                self.ds[f'{ii}'] = data_xr.copy(deep=True)
        
        print(f"Successfully loaded {len(self.ds.data_vars)} run(s).")
        
        return self.ds
    
    def _extract_motor_positions(self, filepath, run):
        """Extract motor positions from an ESRF HDF5 file."""
        run = int(run)
        motor_positions = {}
        
        try:
            with h5py.File(filepath, 'r') as h5_file:
                group_path = f'{run}.1/instrument/positioners'
                
                if group_path not in h5_file:
                    print(f"Warning: Motor positions not found at {group_path}")
                    return None
                
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
        
        # Determine polarization
        motor_positions['polarization'] = self._determine_polarization(motor_positions)
        
        return motor_positions
    
    @staticmethod
    def _determine_polarization(motor_positions):
        """Determine X-ray polarization from undulator motor positions."""
        if 'hu70cp' not in motor_positions or 'hu70ap' not in motor_positions:
            return 'Unknown'
        
        cp = motor_positions['hu70cp']
        ap = motor_positions['hu70ap']
        
        if cp > 30 and ap > 30:
            return 'LV'
        elif -2 < cp < 2 and -2 < ap < 2:
            return 'LH'
        elif 2 <= cp <= 30 and 2 <= ap <= 30:
            return 'C+'
        elif -30 <= cp <= -2 and -30 <= ap <= -2:
            return 'C-'
        else:
            return 'Unknown'
    
    def _extract_xas(self, filepath, run):
        """Extract XAS data from an ESRF HDF5 file."""
        run = int(run)
        
        try:
            with h5py.File(filepath, 'r') as h5_file:
                group_path = f'{run}.1/measurement'
                
                if group_path not in h5_file:
                    print(f"Warning: Measurement data not found at {group_path}")
                    return None
                
                # Try different energy dataset names
                try:
                    energy = h5_file[f'{group_path}/energy_enc'][:]
                except KeyError:
                    try:
                        energy = h5_file[f'{group_path}/energy'][:]
                    except KeyError:
                        print(f"Error: Could not find energy dataset")
                        return None
                
                # Extract fluorescence and monitor data
                try:
                    xas = h5_file[f'{group_path}/ifluo_xmcd'][:]
                    i0 = h5_file[f'{group_path}/i0_xmcd'][:]
                    mir = h5_file[f'{group_path}/mir_xmcd'][:]
                except KeyError as e:
                    print(f"Error: Could not find required dataset: {e}")
                    return None
        
        except OSError as e:
            print(f"File error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        
        # Stack data into 2D array
        data = np.stack((energy, xas, i0, mir), axis=1)
        return data


class DLS_XAS_Data(Raw_XAS_Data):
    """
    Class for loading XAS data from DLS (Diamond Light Source).
    
    Examples
    --------
    >>> xas = DLS_XAS_Data(runs=466796, folder='path/to/data')
    >>> xas.load_data()
    """
    
    def _get_filepaths_from_runs(self):
        """Retrieve DLS file paths from run numbers."""
        raise NotImplementedError(
            "DLS filepath retrieval is not yet implemented."
        )
    
    def load_data(self, xas_dataset='draincurrent_c'):
        """Load XAS data from DLS HDF5 files into an xarray Dataset."""
        print(f"Loading DLS data from {len(self.filepaths)} file(s)...")

        if xas_dataset not in ['draincurrent_c', 'fy2_c', 'diff1_c']:
            raise ValueError("Invalid xas_dataset. Choose from 'draincurrent_c', 'fy2_c', or 'diff1_c'.")
        
        self.ds = xr.Dataset()

        for ii, (filepath, run) in enumerate(zip(self.filepaths, self.runs)):
            print(f"  Processing run {run} from {os.path.basename(filepath)}")

            motor_pos = {}

            with h5py.File(filepath, "r") as f:

                # Detect root group: 'entry' or 'entry1'
                if "entry1" in f:
                    root = "entry1"
                elif "entry" in f:
                    root = "entry"
                else:
                    print(f"  Error: Could not find 'entry' or 'entry1' group in {os.path.basename(filepath)}")
                    continue

                # -------------------------
                # 1. Core scan data
                # -------------------------
                try:
                    energy = f[f"{root}/diff1_c/energy"][()]
                    xas = f[f"{root}/{xas_dataset}/data"][()]
                    i0 = f[f"{root}/instrument/m4c1_c/data"][()]
                except KeyError as e:
                    print(f"  Error: Could not find required dataset for run {run}: {e}")
                    continue

                # -------------------------
                # 2. Manipulator positions
                # -------------------------
                manip = f"{root}/instrument/manipulator"
                for key in ["x", "y", "z", "th", "phi", "chi"]:
                    path = f"{manip}/{key}"
                    motor_pos[key] = f[path][()] if path in f else None

                # -------------------------
                # 3. Polarization
                # -------------------------
                if f"{root}/instrument/id/polarisation" in f:
                    pol = f[f"{root}/instrument/id/polarisation"][()]
                    pol = pol.decode() if isinstance(pol, bytes) else pol
                    motor_pos["polarization"] = pol
                else:
                    motor_pos["polarization"] = None


            # Create xarray DataArray with the data
            data = np.stack((energy, xas, i0), axis=1)
            data_xr = xr.DataArray(
                data,
                dims=['points', 'variable'],
                coords={
                    'points': np.arange(data.shape[0]),
                    'variable': ['Energy (eV)', 'XAS (arb. units)', 'i0 (arb. units)']
                }
            )

            # Add motor positions as attributes
            for k, v in motor_pos.items():
                if v is not None:
                    data_xr.attrs[k] = v

            # Add filepath and run as attributes
            data_xr.attrs['filepath'] = filepath
            data_xr.attrs['run'] = run

            self.ds[f'{ii}'] = data_xr.copy(deep=True)

        print(f"Successfully loaded {len(self.ds.data_vars)} run(s).")
        return self.ds


def main():
    """Example usage."""
    print("Raw XAS Data Classes for different facilities\n")
    print("Available classes:")
    print("  - ESRF_XAS_Data: For ESRF data")
    print("  - DLS_XAS_Data: For DLS data")
    print("\nExample usage:")
    print("  xas = ESRF_XAS_Data(runs=[1, 2, 3], folder='path/to/data')")
    print("  xas.load_data()")
    print("  xas.print_summary()")


if __name__ == "__main__":
    main()
