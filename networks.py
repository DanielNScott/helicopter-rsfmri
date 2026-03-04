from dataio import process_roi_assignments
import pandas as pd
import numpy as np
import scipy.io as sio
import glob
from pathlib import Path

# Some network analysis defaults
networks_to_inspect = ['Default mode', 'Cingulo-opercular Task Control', 'Ventral attention', 'Visual', 'Fronto-parietal Task Control', 'Salience', 'Dorsal attention']
short_names = ['DMN', 'CO', 'VA', 'Vis', 'FPN', 'Sal', 'DAN']

def get_functional_networks(data, power_csv_file, networks_to_inspect, check_assignments, short_names, recompute=False):

    # Check if we actually have to recompute the data
    if recompute:

        # Read Power atlas data and figure out which ROIs belong to which networks
        if check_assignments:
            assignments, order_match = process_roi_assignments(power_csv_file, '../data/fmri', tolerance=5.0)

        # Regardless of whether we skipped the assignment and match check, we need the data 
        power_data = pd.read_csv(power_csv_file, delimiter=';')
        networks = power_data['System'].unique()

        # Get ROIs for each network to inspect
        rois = {}
        for net in networks_to_inspect:
            rois[net] = power_data[power_data['System'] == net]['ROI'].values - 1
            print(f"Network: {net}, ROIs: {rois[net]}")

        # Re-assemble the correlation matrices
        n_subj = data.shape[0]
        correlation_matrices, old_names, new_names = reassemble_correlations(data, n_subj)

        # Get average within-network correlations for each network
        net_corrs, flat_corrs, flatcorrcols, flatcorrnames, corrcols = get_within_network_correlations(correlation_matrices, rois, networks_to_inspect, short_names)

        # Network aggregated data
        X = pd.DataFrame(flat_corrs, columns=flatcorrnames)

        # Save the functional network data
        X.to_csv(f"../data/functional_network_corrs.csv", index=False)

    else:
        # Load the precomputed functional network data
        X = pd.read_csv(f"../data/functional_network_corrs.csv")
        flatcorrnames = X.columns.to_list()

    return X, flatcorrnames


def reassemble_correlations(data, n_subj):
    # Rename the columns of the data matrix with the ROI numbers
    old_names, new_names = [], []
    linear_index = 1
    correlation_matrix = np.zeros((n_subj, 264, 264))
    for row in np.arange(264):
        for col in np.arange(264):
            if row == col:
                correlation_matrix[:, row, col] = 1.0

            if col > row:
                roi_pair_name = f"ROI_{row}_ROI_{col}"
                old_names.append(f'V{linear_index}')
                new_names.append(roi_pair_name)

                # Save data to the correlation matrix
                correlation_matrix[:, row, col] = data[old_names[-1]].values
                correlation_matrix[:, col, row] = data[old_names[-1]].values

                linear_index += 1

    return correlation_matrix, old_names, new_names

def get_within_network_correlations(correlation_matrices, rois, networks_to_inspect, short_names):
    n_subj = correlation_matrices.shape[0]
    net_corrs  = np.zeros((n_subj, len(networks_to_inspect), len(networks_to_inspect)))
    flat_corrs = np.zeros((n_subj, len(networks_to_inspect) * (len(networks_to_inspect) + 1) // 2))

    #elements = {}

    linear_index = 0
    flatcorrcols, flatcorrnames = [], []
    for i, name1 in enumerate(networks_to_inspect):
        indices1 = rois[name1]
        for j, name2 in enumerate(networks_to_inspect):
            if j >= i:
                indices2 = rois[name2]
                net_corrs[:, i, j] = np.sum(np.triu(correlation_matrices[:, indices1, :][:,:,indices2]),axis=(1,2))/(len(indices1)*len(indices2))
                net_corrs[:, j, i] = net_corrs[:, i, j]

                flat_corrs[:, linear_index] = net_corrs[:, i, j]

                short_name_1 = short_names[i]
                short_name_2 = short_names[j]

                flatcorrcols.append(f"{name1}_x_{name2}")
                flatcorrnames.append(f"{short_name_1}_x_{short_name_2}")

                #elements[f'{name1}_x_{name2}'] = []
                #for s in range(n_subj):
                #    elements[f'{name1}_x_{name2}'].append(np.triu(correlation_matrices[:, indices1, :][:,:,indices2]).flatten())

                linear_index += 1
                
    corrcols = [f"{name1}_x_{name2}" for name1 in short_names for name2 in short_names]

    return net_corrs, flat_corrs, flatcorrcols, flatcorrnames, corrcols


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


if __name__ == "__main__":
    power_csv_file = './data/power_et_al_2011.csv'
    assignments, order_match = process_roi_assignments(power_csv_file, './data/fmri', tolerance=5.0)

    power_data = pd.read_csv(power_csv_file, delimiter=';')
    networks = power_data['System'].unique()
    networks_to_inspect = ['Default mode', 'Cingulo-opercular Task Control', 'Ventral attention', 'Visual', 'Fronto-parietal Task Control', 'Salience', 'Dorsal attention']

    # Get ROIs for each network to inspect
    for net in networks_to_inspect:
        rois = power_data[power_data['System'] == net]['ROI'].values
        print(f"Network: {net}, ROIs: {rois}")
