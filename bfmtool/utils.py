from datetime import datetime
import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
import typing as tt
from tqdm import tqdm

def file_metadata(filepath):
    filepath = Path(filepath)
    metadata_regex = r'^([A-Za-z]+)\_data\_([A-Za-z]+)\_([A-Za-z]+)\_([A-Za-z]+)\_([A-Za-z]+)\_(\d{8})\_(\d{6})\.csv$'

    match = re.search(metadata_regex, filepath.name)

    if match is None:
        return None
    
    source = match.group(1)
    activity = match.group(2)
    subject = match.group(3)
    collection_method = match.group(4)
    environment = match.group(5)

    date_str = match.group(6) 
    time_str = match.group(7) 

    datetime_str = f"{date_str}_{time_str}" 
    format_code = '%Y%m%d_%H%M%S'
    time = datetime.strptime(datetime_str, format_code)
    

    return dict(
        source = source,
        activity = activity,
        subject = subject,
        collection_method = collection_method,
        environment = environment,
        time = time,
        session_id = datetime_str
    )

ColumnInfo = tt.Dict[str, tt.Union[tt.List[str], tt.List[int]]]

def get_bfm_columns(df) -> ColumnInfo: 
    """Extracts column names and subcarrier indices.

    Args:
        df: dataframe containing bfm data

    Returns:
        A dictionary containing the following keys:
            'real_col' (List[str]): Column names for the real component.
            'real_scidx' (List[int]): Subcarrier indices for the real component.
            'imag_col' (List[str]): Column names for the imaginary component.
            'imag_scidx' (List[int]): Subcarrier indices for the imaginary component.
            'mag_col' (List[str]): Column names for the magnitude.
            'mag_scidx' (List[int]): Subcarrier indices for the magnitude.
            'phase_col' (List[str]): Column names for the phase.
            'phase_scidx' (List[int]): Subcarrier indices for the phase.
    """
    scidx_regex = r'SCIDX_(-?\d+)_'
    match_list = [re.search(scidx_regex, col) for col in df.columns]
    
    subcarrier_indices = {int(match.group(1)) for match in match_list if match is not None}
    subcarrier_indices = sorted(list(subcarrier_indices))
    
    
    real_pair = [(f'SCIDX_{scidx}_Ratio_Real', scidx) for scidx in subcarrier_indices if f'SCIDX_{scidx}_Ratio_Real' in df.columns]
    imag_pair = [(f'SCIDX_{scidx}_Ratio_Imag', scidx) for scidx in subcarrier_indices if f'SCIDX_{scidx}_Ratio_Imag' in df.columns]
    mag_pair = [(f'SCIDX_{scidx}_Ratio_Mag', scidx) for scidx in subcarrier_indices if f'SCIDX_{scidx}_Ratio_Mag' in df.columns]
    phase_pair = [(f'SCIDX_{scidx}_Ratio_Phase', scidx) for scidx in subcarrier_indices if f'SCIDX_{scidx}_Ratio_Phase' in df.columns]

    real_col = [pair[0] for pair in real_pair]
    real_scidx = [pair[1] for pair in real_pair]
    imag_col = [pair[0] for pair in imag_pair]
    imag_scidx = [pair[1] for pair in imag_pair]
    mag_col = [pair[0] for pair in mag_pair]
    mag_scidx = [pair[1] for pair in mag_pair]
    phase_col = [pair[0] for pair in phase_pair]
    phase_scidx = [pair[1] for pair in phase_pair]

    return dict(
        real_col = real_col,
        real_scidx = real_scidx,
        imag_col = imag_col,
        imag_scidx = imag_scidx,
        mag_col = mag_col,
        mag_scidx = mag_scidx,
        phase_col = phase_col,
        phase_scidx = phase_scidx,
    )


def get_df_from_dir(data_dir):
    df_list = []
    time_list = []
    data_dir = Path(data_dir)

    for file in tqdm(data_dir.rglob('bfm_data_*.csv')):
        file = Path(file)
        metadata = file_metadata(file)

        if metadata is None:
            continue

        df = pd.read_csv(file)
        df.insert(0, 'activity', metadata['activity'])
        df.insert(0, 'subject', metadata['subject'])
        df.insert(0, 'environment', metadata['environment'])
        df.insert(0, 'session_id', metadata['session_id'])

        df_list.append(df)
        time_list.append(metadata['time'])

    sorted_pairs = sorted(list(zip(df_list, time_list)), key = lambda x : x[1])
    sorted_df = [pair[0] for pair in sorted_pairs]

    df_combined = pd.concat(sorted_df).reset_index(drop = True)
    df_combined['timestamp'] = pd.to_datetime(df_combined.timestamp, unit = 's', utc = True)
    
    return df_combined


def append_mag_phase(df):
    col_dict = get_bfm_columns(df)

    assert len(col_dict['real_col']) == len(col_dict['imag_col'])
    assert np.all(np.isin(col_dict['real_scidx'], col_dict['imag_scidx']))

    subcarrier_indices = col_dict['real_scidx']

    real_data = df[col_dict['real_col']].to_numpy()
    imag_data = df[col_dict['imag_col']].to_numpy()

    mag_data = (real_data**2 + imag_data ** 2) ** (1/2)
    phase_data = np.unwrap(np.arctan2(imag_data, real_data))

    mag_col = [f'SCIDX_{scidx}_Ratio_Mag' for scidx in subcarrier_indices]
    phase_col = [f'SCIDX_{scidx}_Ratio_Phase' for scidx in subcarrier_indices]

    mag_df = pd.DataFrame(mag_data, columns = mag_col)
    phase_df = pd.DataFrame(phase_data, columns = phase_col)

    return pd.concat((df, mag_df, phase_df), axis = 1)


def filter_by_mode(df, mode_cols, group_by=None):
    """
    Filters a DataFrame to keep rows matching the mode of specified columns,
    optionally within groups.

    Args:
        df (pd.DataFrame): The input DataFrame.
        mode_cols (list): A list of column names to find the mode of (as a pair or single).
        group_by (list, optional): A list of column names to group by.
                                   If None, finds the global mode. Defaults to None.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the mode rows.
    """
    if group_by is None:
        # Case 1: Find the global mode (no grouping)
        grouping_cols = mode_cols
    else:
        # Case 2: Find the mode within groups
        grouping_cols = group_by + mode_cols

    # Count occurrences of the mode_cols within each group
    counts = df.groupby(grouping_cols).size().reset_index(name='count')

    if group_by is None:
        # Find the single global mode
        mode_identifier = counts.loc[counts['count'].idxmax()]
        # Drop the count column to prepare for merge
        mode_identifier = mode_identifier.drop('count').to_frame().T
        # Convert types to match original df for a clean merge
        mode_identifier = mode_identifier.astype(df[mode_cols].dtypes)
    else:
        # Find the mode within each group
        idx = counts.groupby(group_by)['count'].idxmax()
        mode_identifier = counts.loc[idx][grouping_cols]

    # Merge to filter the original DataFrame
    return pd.merge(df, mode_identifier, on=grouping_cols)


def median_filter_df(
        df : pd.DataFrame, 
        columns : tt.List[str], 
        window_size : int, 
        mode : str = 'nearest', 
        group_by : tt.List[str] = None
    ):
    
    if group_by is not None:
        df_list = []
        for idx, df_group in df.groupby(group_by):
            filtered_data = median_filter(
                df_group[columns],
                size = (window_size, 1),
                mode = 'nearest'
            )
            df_group[columns] = filtered_data
            df_list.append(df_group)
        
        return pd.concat(df_list, axis = 0)
    
            
