#!/usr/bin/env python3
import os
import json
import pandas as pd
import glob
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process QMOF dataset to create id_prop.csv')
    parser.add_argument('--qmof_json', type=str, required=True, 
                        help='Path to qmof.json file containing properties')
    parser.add_argument('--structures_dir', type=str, required=True, 
                        help='Path to relaxed_structures directory containing structure files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory where id_prop.csv will be saved')
    parser.add_argument('--properties', type=str, nargs='+', default=['bandgap', 'cbm', 'vbm'],
                        help='List of properties to include in id_prop.csv')
    return parser.parse_args()

def find_structure_files(structures_dir):
    """Find all structure files in the structures directory."""
    # Look for common structure file formats
    structure_files = []
    for ext in ['*.cif', '*.vasp', '*.poscar', '*.xyz']:
        structure_files.extend(glob.glob(os.path.join(structures_dir, '**', ext), recursive=True))
    
    logger.info(f"Found {len(structure_files)} structure files")
    
    # Extract MOF IDs from filenames, handling the "qmof-" prefix
    mof_ids = {}
    for filepath in structure_files:
        filename = os.path.basename(filepath)
        # Remove extension to get the ID
        mof_id = os.path.splitext(filename)[0]
        # print(f"Extracted MOF ID: {mof_id} from file: {filename}")
        # Handle the "qmof-" prefix
        if mof_id.startswith('qmof-'):
            # Remove the "qmof-" prefix to get the pure ID
            clean_id = mof_id[5:]  # Skip the first 5 characters "qmof-"
            mof_ids[clean_id] = filepath
            # Also keep original ID for flexibility
            mof_ids[mof_id] = filepath
        else:
            mof_ids[mof_id] = filepath
    
    logger.info(f"Extracted {len(mof_ids)} unique MOF IDs")
    return mof_ids

def process_qmof_data(qmof_json, structure_files, properties):
    """Process QMOF JSON data and match with structure files."""
    # Load QMOF JSON data
    logger.info(f"Loading QMOF data from {qmof_json}")
    try:
        with open(qmof_json, 'r') as f:
            qmof_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading QMOF JSON file: {e}")
        raise
    print(qmof_data[0])
    logger.info(f"QMOF data contains {len(qmof_data)} entries")
    
    # Create id_prop dataframe by matching structures and properties
    rows = []
    matched_count = 0
    print("type of qmof_data", type(qmof_data))
    # Check for data structure format - list of dicts or dict of dicts
    if isinstance(qmof_data, list):
        # List of dictionaries format
        for entry in qmof_data: 
            mof_id = "qmof_id"
            found = False
            test_id = entry[mof_id]
            if test_id in structure_files:
                prop_values = [entry["outputs"]["pbe"][prop] for prop in properties]
                # Add to rows if all properties are available
                print(prop_values)
                if not all(pd.isna(val) for val in prop_values):
                    rows.append([test_id] + prop_values)
                    matched_count += 1
                    found = True
            if not found:
                logger.debug(f"No structure file found for MOF ID: {mof_id}")
   
    
    logger.info(f"Matched {matched_count} MOFs with structure files")
    columns = ['mof_id'] + properties
    id_prop_df = pd.DataFrame(rows, columns=columns)
    
    return id_prop_df


def main():
    """Main function to process QMOF dataset and create id_prop.csv."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find structure files
    structure_files = find_structure_files(args.structures_dir)
    
    # Print some sample structure files for debugging
    sample_size = min(5, len(structure_files))
    logger.info(f"Sample of structure files found:")
    for i, (mof_id, filepath) in enumerate(list(structure_files.items())[:sample_size]):
        logger.info(f"  {i+1}. ID: {mof_id} -> Path: {filepath}")
    
    # Process QMOF data and match with structure files
    id_prop_df = process_qmof_data(args.qmof_json, structure_files, args.properties)
    
    # Save id_prop.csv
    output_path = os.path.join(args.output_dir, 'id_prop.csv')
    id_prop_df.to_csv(output_path, index=False)
    logger.info(f"Saved id_prop.csv with {len(id_prop_df)} entries to {output_path}")
    
    # Create atom_init.json if needed
    # create_atom_init_json(args.output_dir)
    


if __name__ == "__main__":
    main()