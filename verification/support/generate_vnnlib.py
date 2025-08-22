import os
import numpy as np
import pandas as pd

""" 
cd "/home/bagir/Documents/1) Projects/2) AC verification/verification/alpha-beta-CROWN/complete_verifier"
conda activate genbab
python abcrown.py --config acopf/vrvi_with_wc.yaml

"""


def generate_vnnlib_from_excel(excel_path: str, output_path: str):
    """
    Generates a VNNLIB file by reading input bounds and output declarations
    from an Excel file with 'inputs' and 'outputs' sheets.

    Args:
        excel_path (str): The path to the Excel file containing the input bounds
                          and output declarations.
        output_path (str): The path where the VNNLIB file will be saved.
    """
    try:
        # Read the 'inputs' sheet for X_i bounds
        df_inputs = pd.read_excel(excel_path, sheet_name='inputs')
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found at {excel_path}")
    except ValueError:
        raise ValueError("The Excel file must contain a sheet named 'inputs'.")

    # Read the 'outputs' sheet for Y_i declarations
    try:
        df_outputs = pd.read_excel(excel_path, sheet_name='outputs')
    except ValueError:
        print("Warning: 'outputs' sheet not found. Proceeding without declaring output variables.")
        df_outputs = None

    # Ensure the inputs sheet has the correct columns
    if 'Input features' not in df_inputs.columns or 'Lower bound' not in df_inputs.columns or 'Upper bound' not in df_inputs.columns:
        raise ValueError("The 'inputs' sheet must contain 'Input features', 'Lower bound', and 'Upper bound' columns.")

    with open(output_path, 'w') as f:
        f.write("; VNNLIB file generated from Excel\n")

        # Part 1: Declare input variables (X_i)
        for i, row in df_inputs.iterrows():
            f.write(f"(declare-const X_{i} Real)\n")
        
        # Part 2: Declare output variables (Y_i) if the 'outputs' sheet exists
        if df_outputs is not None:
            for i, row in df_outputs.iterrows():
                f.write(f"(declare-const Y_{i} Real)\n")
                # if i == 117: # if you want to check something in the output of size n_buses
                #     break
        
        f.write("\n")

        # Part 3: Assert bounds for input variables (X_i)
        for i, row in df_inputs.iterrows():
            lower, upper = row['Lower bound'], row['Upper bound']
            # Make sure to handle potential NaN values
            if pd.notna(lower) and pd.notna(upper):
                f.write(f"; Input {i} bounds:\n")
                f.write(f"(assert (>= X_{i} {lower}))\n")
                f.write(f"(assert (<= X_{i} {upper}))\n")

        f.write("\n")
        
        # Part 4: Add property to be verified (Example: Y_0 > 100000)
        if df_outputs is not None:
            for i in range(len(df_outputs)):
                f.write(f"(assert (<= Y_0 10))\n") # assert voltage always between -2 and 2
                #f.write(f"(assert (>= Y_0 -5))\n")
                # f.write(f"(assert (<= Y_{i} 1000))\n") # assert voltage always between -2 and 2
                # f.write(f"(assert (>= Y_{i} -1000))\n")
                if i == 0:
                    break

    print(f"VNNLIB file generated: {output_path}")

if __name__ == "__main__":
    # Define a consistent base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    excel_name = 'load_bounds_case118.xlsx'
    excel_path = os.path.join(base_dir, 'verification', 'nn_models', 'bounds', excel_name)
    excel_path_ab_crown = os.path.join(base_dir, 'verification', 'alpha-beta-CROWN/complete_verifier/acopf', excel_name)
    
    # # Ensure the output directory exists
    # output_dir = os.path.join(base_dir, 'verification', 'crown_files', 'vnnlib')
    # os.makedirs(output_dir, exist_ok=True)
    
    output_name = 'vrvi_with_wc.vnnlib'
    output_path = os.path.join(base_dir, 'verification', 'crown_files', 'vnnlib', output_name) 
    output_path_ab_crown = os.path.join(base_dir, 'verification', 'alpha-beta-CROWN/complete_verifier/acopf', output_name)

    try:
        generate_vnnlib_from_excel(
            excel_path=excel_path,
            output_path=output_path
        )
        generate_vnnlib_from_excel(
            excel_path=excel_path_ab_crown,
            output_path=output_path_ab_crown
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
