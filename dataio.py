import pickle
import inspect
import pyreadr
import pandas as pd

import numpy as np
import scipy.io as sio
import glob
from pathlib import Path


def load_data(fmri_data_path, behavioral_data_path):
    # Load pre-processed data
    result = pyreadr.read_r(fmri_data_path)
    fmri_data = result[None]  # Adjust key if necessary

    # Load behavioral data
    behavioral_data = pd.read_csv(behavioral_data_path)

    # Rename 'Subject' column to 'id' in behavioral data
    behavioral_data.rename(columns={'Subject': 'id'}, inplace=True)

    # Get columns to include in PCA
    fmri_columns = [col for col in fmri_data.columns if col.startswith('V')]

    # Subset fmri_data to only imaging and ID data
    fmri_data = fmri_data[['id'] + fmri_columns]

    # Merge fmri_data and behavioral_data on 'id'
    data = pd.merge(fmri_data, behavioral_data, on='id', how='inner')

    # Column for checking that RWLike is CPLike + IOLike
    # data['JointLike'] = data['CPLike'] + data['IOLike']

    return data, fmri_columns

def save_workspace(filename="workspace.pkl"):
    """Saves all picklable global variables to a file."""
    # Get the caller's frame (the script that called this function)
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    
    data = {}
    for name, obj in caller_globals.items():
        # Exclude built-in modules, unpicklable objects, etc.
        if not name.startswith("__") and not callable(obj) and not isinstance(obj, type):
            try:
                pickle.dumps(obj)  # Test if object is picklable
                data[name] = obj
            except (TypeError, AttributeError):
                pass  # Skip unpicklable objects
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} variables to {filename}")

def load_workspace(filename="workspace.pkl"):
    """Loads variables from a pickled file into the caller's global namespace."""
    # Get the caller's frame
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    caller_globals.update(data)
    print(f"Loaded {len(data)} variables from {filename}")


def process_roi_assignments(csv_file, ts_directory, tolerance=4.0):
    """
    Match ROI coordinates and extract network assignments for all subjects
    
    Parameters:
    csv_file: path to Power coordinates CSV
    ts_directory: directory containing *_ts.mat files
    tolerance: coordinate matching tolerance in mm
    
    Returns:
    assignments: dict with subject IDs as keys, assignments as values
    order_match: dict with subject IDs as keys, boolean match status as values
    """
    
    # Load Power atlas data
    power_data = pd.read_csv(csv_file, delimiter=';')
    power_coords = power_data[['X', 'Y', 'Z']].values
    power_assignments = power_data['Assignment'].values
    
    # Find all timeseries files
    ts_files = glob.glob(str(Path(ts_directory) / "*_ts.mat"))
    print(f"Found {len(ts_files)} timeseries files to process")
    
    assignments = {}
    order_match = {}
    
    for i, ts_file in enumerate(ts_files, 1):
        # Extract subject ID from filename
        subject_id = Path(ts_file).stem.replace('_ts', '')
        
        # Load MATLAB file
        mat_data = sio.loadmat(ts_file)
        subj_coords = np.array([mat_data['ts']['ROI'][0,0]['MNI_center'][0, i].flatten() for i in range(264)])

        # The mat data causes problems for debugging
        del mat_data

        # Match coordinates
        matches, indices = match_coordinates(subj_coords, power_coords, tolerance)
        n_matched = np.sum(matches)
        
        # Check order match
        is_order_match = np.array_equal(indices[matches], np.arange(len(indices))[matches])
        order_match[subject_id] = is_order_match
        
        # Store assignments
        assignments[subject_id] = power_assignments[indices[matches]]
        
        # Print results
        print(f"Processed subject {i}/{len(ts_files)}: {subject_id}. {n_matched}/{len(subj_coords)} ROIs matched coordinates. Order match: {'Yes' if is_order_match else 'No'}")
        if n_matched < len(subj_coords):
            print(f"  - Warning: {len(subj_coords) - n_matched} ROIs had no coordinate match")
    
    print("Processing complete!")
    return assignments, order_match

def match_coordinates(coords1, coords2, tolerance):
    """Match coordinates within tolerance"""
    indices = []
    for coord in coords1:
        distances = np.linalg.norm(coords2 - coord, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] <= tolerance:
            indices.append(min_idx)
        else:
            indices.append(-1)  # No match found
    return np.array(indices) >= 0, np.array(indices)
