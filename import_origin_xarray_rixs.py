import os
import pandas as pd
try:
    import originpro as op
except ImportError:
    print("originpro module not found. Make sure you have Origin installed and the originpro Python package available.")
import numpy as np
import xarray as xr

def read_file(file):
    # Open the xarray dataset
    ds = xr.open_dataset(file, engine='h5netcdf')
    # Extract attributes from each DataArray and concatenate values for each key
    attr_keys = list(next(iter(ds.data_vars.values())).attrs.keys())
    metadata = {key: [] for key in attr_keys}
    for da in ds.data_vars.values():
        for key in attr_keys:
            metadata[key].append(da.attrs.get(key, None)) 

    return ds, metadata

def filter_metadata(metadata):
    # Filter out metadata keys that are not relevant for display

    #set the decimal places for each of the numeric ones
    digits = {'H [rlu]': 3, 'K [rlu]': 3, 'L [rlu]': 3, 'th': 3, 'chi': 3, 'phi': 3, 'tth': 3, 'energy': 2, 'T': 2, 'B': 2}

    filtered_metadata = {}
    for key, values in metadata.items():
        if key not in ['th', 'chi', 'phi', 'tth', 'H', 'K', 'L','mirror', 'T', 'B', 'polarization', 'energy', 'run', 'sample']:
            continue  # skip x_name and y_name as they are used for labels
        else:
            key = key + ' [rlu]' if key in ['H','K','L'] else key
            values = None if values == np.nan else values

            if key in digits.keys():
                values = [f"{float(v):.{digits[key]}f}" for v in values]
            filtered_metadata[key] = values
    return filtered_metadata

def _format_value(v):
    """Format a metadata value for display in Origin.
    
    Returns an empty string for None or NaN values,
    otherwise formats numeric values using 6 significant figures.
    """
    if v is None:
        return ''
    if isinstance(v, float) and np.isnan(v):
        return ''
    if isinstance(v, np.generic) and np.isnan(v):
        return ''
    if isinstance(v, (float, int, np.generic)):
        return f'{v:.6g}'
    
    return v

def strip_units_from_name(name):
    """Remove '(Y)' or '(Y)' suffix from a name like 'X (Y)' or 'X(Y)'."""
    import re
    return re.sub(r'\s*\(.*?\)\s*$', '', name).strip()

def get_longname_list(ds, save_error):
    # Extract long names from the dataset
    longname_list = []
    for var in ds.data_vars:
        da = ds[var]
        # Prefer explicit x_name/y_name attributes if available
        x_name = da.attrs.get('x_name', None)
        y_name = da.attrs.get('run', None) if 'run' in da.attrs.keys() else da.attrs.get('y_name', None)
        if x_name is not None and y_name is not None:
            var_names = [x_name, y_name]
            if save_error:
                error_name = da.attrs.get('error_name', None) if 'error_name' in da.attrs.keys() is not None else 'error'
                error_name = error_name + '_' + y_name
                var_names.append(error_name)
        else:
            var_names = list(da.coords['variable'].values)

        longname_list.extend([str(name) for name in var_names])
    return longname_list

def get_longname_units_list_map(longname_list, units_list):
    #For the map, only one x-column is necessary
    longname_list_map = [longname_list[0]]
    step = 3 if save_error else 2
    longname_list_map.extend(longname_list[1::step])
    units_list_map = [units_list[0]]
    units_list_map.extend(units_list[1::step])
    return longname_list_map, units_list_map


def separate_longname_from_units(longname_list):
    # Extract units from the dataset
    units_list = []
    for name in longname_list:
        if '(' in name and ')' in name:
                start = name.find('(') + 1
                end = name.find(')', start)
                if start >= end:
                    units_list.append(' ')
                else:
                    units_list.append(name[start:end])
        else:
            units_list.append(' ')

    longname_list = [str(strip_units_from_name(name)) for name in longname_list]

    return longname_list, units_list

def create_pandaframe(ds):
    # Create a pandas DataFrame from the xarray dataset
    dfs = []
    for name, da in ds.data_vars.items():
        df = da.to_dataframe(name="value").unstack("variable")
        df.columns = df.columns.get_level_values(1)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=1)

    return combined_df


def interpolate_df(df):
    # Extract the reference x column (first column of df)
    x_ref = df.iloc[:, 0].values

    # Prepare a new DataFrame: first column is x_ref, then all interpolated y columns
    new_data = {'x': x_ref}
    step = 3 if save_error else 2
    for i in range(1, df.shape[1], step):
        x = df.iloc[:, i - 1].values
        y = df.iloc[:, i].values
        # Interpolate y onto x_ref
        y_interp = np.interp(x_ref, x, y, left=0, right=0)
        new_data[f'y_{(i // step) + 1}'] = y_interp

    df_new = pd.DataFrame(new_data)
    return df_new

#### process the sheet with (x,y) data
# The file chosen by Origin import filter is placed into the fname$
# LabTalk variable. Must bring it into Python. Don't specify $ in name!

# Read the file into a pandas DataFrame.
fname = op.get_lt_str('fname')
# fname = r"C:\Users\leona\OneDrive - Universität Zürich UZH\LQMR group - PBCO_magnons\data\processed_rixs\packaged\energy_map_CALAS_37_ann1_oxygenK_LH.hdf5"
ds, metadata = read_file(fname)

data_vars = ds.data_vars.keys()
save_error = True
for scan in data_vars:
    if 'error' in ds[scan].coords['variable']:
        pass
    else:
        save_error = False
        break


df = create_pandaframe(ds)
longname_list = get_longname_list(ds, save_error)
longname_list, units_list = separate_longname_from_units(longname_list)
metadata = filter_metadata(metadata)
df_combined = interpolate_df(df)
longname_list_map, units_list_map = get_longname_units_list_map(longname_list, units_list)

# --- Create Origin worksheet and write data using from_df ---
wks = op.find_sheet()
# wks.name = os.path.basename(fname).split('.')[0]  # Set the worksheet name to the file name without extension
wkbook = wks.get_book()
wkbook.lname = os.path.basename(fname).split('.')[0] # Set the workbook name to the file name without extension, replacing "_" with " "
wks.name = 'Sheet1'
wks.from_df(df)

# save long names and units
# xy_labels = ''.join(['x' if i % 2 == 0 else 'y' for i in range(len(next(iter(metadata.values()), [])))])

if save_error:
    xy_labels = ''
    for i in range(len(longname_list)):
        if i % 3 == 0:
            xy_labels += 'x'
        elif i % 3 == 1:
            xy_labels += 'y'
        else:
            xy_labels += 'e'
else:
    xy_labels = ''.join(['x' if i % 2 == 0 else 'y' for i in range(len(longname_list))])
wks.cols_axis(xy_labels, repeat=False)


# --- Set metadata as user parameters ---
if metadata:
    for row_idx, (key, values) in enumerate(metadata.items()):
        if row_idx > 16:
            raise ValueError(f"Too many metadata keys ({len(metadata)}). Origin only supports up to 17 user parameter rows. Key '{key}' at index {row_idx} exceeds this limit.")
        wks._user_param_row(key, add=True)
        # Format numeric values using significant figures, only for 'y' columns
        formatted_values = []
        y_idx = 0
        for col_label in xy_labels:
            if col_label == 'y':
                v = values[y_idx] if y_idx < len(values) else None
                formatted_values.append(_format_value(v))
                y_idx += 1
            else:
                formatted_values.append(' ')  # empty for 'x' columns
        # print(f"Setting user parameter '{key}' with values: {formatted_values}, type {type(formatted_values[0])}, length {len(formatted_values)}")
        wks.set_labels(formatted_values, key)

wks.set_labels(longname_list, 'L')
wks.set_labels(units_list, 'U')


#for the interpolated Map
wks2 = wkbook.add_sheet('map')
wks2.from_df(df_combined)
xy_labels2 = ['x'] + ['y' for _ in longname_list_map[1:]]

if metadata:
    for row_idx, (key, values) in enumerate(metadata.items()):
        if row_idx > 16:
            raise ValueError(f"Too many metadata keys ({len(metadata)}). Origin only supports up to 17 user parameter rows. Key '{key}' at index {row_idx} exceeds this limit.")
        wks2._user_param_row(key, add=True)
        # Format numeric values using significant figures, only for 'y' columns
        formatted_values = []
        y_idx = 0
        for col_label in xy_labels2:
            if col_label == 'y' or col_label == 'e':
                v = values[y_idx] if y_idx < len(values) else None
                formatted_values.append(_format_value(v))
                y_idx += 1
            else:
                formatted_values.append(' ')  # empty for 'x' columns
        # print(f"Setting user parameter '{key}' with values: {formatted_values}, type {type(formatted_values[0])}, length {len(formatted_values)}")
        wks2.set_labels(formatted_values, key)
wks2.set_labels(longname_list_map, 'L')
wks2.set_labels(units_list_map, 'U')





