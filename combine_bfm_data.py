from bfmtool.utils import get_bfm_columns, get_df_from_dir, append_mag_phase, filter_by_mode
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--directory', '-d',
        type = str,
        help = 'Path to directory containing bfm_data',
        required = True
    )

    parser.add_argument(
        '-o', '--output',
        metavar='OUTPUT_PATH',  
        type=str,
        required=True,        
        help='Path to save the filtered output CSV file.'
    )

    parser.add_argument(
        '--to-polar', '-p',
        action='store_false',  
        help="Convert 'real' and 'imag' columns to 'magnitude' and 'phase'."
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    df = get_df_from_dir(Path(args.directory))
    col_dict = get_bfm_columns(df)

    if args.to_polar:
        df = append_mag_phase(df)
        df.drop(col_dict['real_col'] + col_dict['imag_col'], axis =1, inplace = True)

    df = filter_by_mode(df, mode_cols=['transmitter_address', 'receiver_address'], group_by = ['session_id'])
    
    df.to_csv(args.output)
    