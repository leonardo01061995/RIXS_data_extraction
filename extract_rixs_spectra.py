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
from static_functions import _determine_polarization

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
               
        if isinstance(folder, list) and len(folder) != len(runs):
            raise ValueError("If folder is a list, its length must match the number of runs.")

        if isinstance(folder, str):
            folder = [folder]*len(runs)
        
        self.folder = folder
        print(f"\n\nFolder found:")
        for fff in  self.folder:
            print(f"\t{fff}")

        self.runs = runs if isinstance(runs, list) else [runs]
        if len(self.runs) != len(set(self.runs)):
            raise ValueError("Duplicate run numbers found in the provided runs.")
        
        # Check for duplicates in runs
        
        if isinstance(scans, str):
            self.scans = ['all']*len(self.runs)

        if len(self.runs) != len(self.scans):
            raise ValueError("The number of runs must match the number of scans provided.")
        
    


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

        for folder, run in zip(self.folder, formatted_runs):
            # Get all .spec files in directory
            files = sorted(
                (entry for entry in os.scandir(folder) if entry.name.lower().endswith('.spec')),
                key=lambda f: f.name  # Sort by file name (assuming natural order by numbers)
            )

            for file in files:
                file_path = os.path.join(folder, file)
                if run in file.name:
                    file_names.append(file_path)

        if not file_names:
            print(f"Warning: No valid .spec files found for runs {runs} \n\n")
        else:
            print(f"Found {len(file_names)} files: runs {runs}.")

        return file_names
    

    def extract_1d_runs(self,
                        x_name, y_name, norm_name, 
                        motor_names = {
                                "th": "th",
                                "chi": "chi",
                                "phi": "phi",
                                "tth": "rtth",
                                "energy": "energy",
                                "x": "xsam",
                                "y": "ysam",
                                "z": "zsam",
                                "T": "tstage",
                            },
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
        self.spec_files = self._search_runs(self.runs)

        # Initialize SpecFile objects for each found .spec file
        print(f"\n\nExtracting 1D spectra from {len(self.spec_files)} .spec files for runs {self.runs}...")
        start_time = time.time()
        spectra_xarray = xr.Dataset()
        for i, (file, run) in enumerate(zip(self.spec_files, self.runs)):
            specfile = SpecFile(file, run)
            extracted_data = specfile.extract_data(self.scans[i], x_name, y_name, norm_name, motors_dict=motor_names,
                                                   scans_from_same_run=scans_from_same_run)
            
            for scan_name, data_array in extracted_data.items():
                new_name = f"run_{run}_scan_{scan_name.split('_')[1]}"
                spectra_xarray[new_name] = data_array
                norm_values = data_array.sel(variable='norm').values
                spectra_xarray[new_name].attrs['mirror'] = float(np.mean(norm_values))

        spectra_xarray.attrs['log'] = ''

        print(f"\tExtraction completed in {time.time() - start_time:.2f} seconds.")

        return Generated_1D_RIXS_Spectra(ds=spectra_xarray)



def main():
    print('ciao')   

  

if __name__ == "__main__":
    main()