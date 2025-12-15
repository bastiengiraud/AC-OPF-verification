import re
import os


"""
The case is modified by:
1) Calculating an equivalent linear cost coefficient (c'1) for each generator.
2) Aggregating generators by bus, summing their limits and dispatches, doing a weighted average for their cost.

Note: generator bases are all set to 100 MVA in the output case.

"""


# --- Constants ---
# Assuming standard system base for normalization
NEW_SBASE_MVA = 100.0 
# Threshold for identifying generators with anomalous small mBase (triggers no-scaling fix)
MBASE_THRESHOLD = 0.0 

# ----------------- PASS 1 HELPERS: NORMALIZATION ------------------

def normalize_generator_data(line_string, new_sbase_mva=NEW_SBASE_MVA):
    """
    Normalizes the generator data line, applying the mBase correction/scaling.
    Returns a dictionary of normalized values needed for aggregation/cost calculation.
    """
    clean_line = line_string.strip().rstrip(';')
    values = re.split(r'\s+', clean_line)
    
    if not values or len(values) < 10 or line_string.strip().startswith('%'):
        return None 

    try:
        data = [float(v) for v in values[:10]]
        
        bus, pg, qg, qmax, qmin, vg, current_mbase, status, pmax, pmin = data
        original_mbase = current_mbase
        
        if int(status) != 1: # Skip out-of-service units
            return None

        # Determine scaling factor
        scale_factor = 1#new_sbase_mva / current_mbase
        
        # # --- ANOMALY FIX check ---
        # if current_mbase < MBASE_THRESHOLD and current_mbase != new_sbase_mva:
        #     # If mBase is very small, we correct mBase but skip scaling P/Q limits,
        #     # assuming they are already in system MW.
        #     # print(f"  -> ANOMALY FIX: mBase corrected from {current_mbase:.2f} to 100.0 without power scaling for bus {int(bus)}.")
        #     scale_factor = 1.0 # Effective scale factor for power is 1.0 (no change)
            
        # elif current_mbase != new_sbase_mva:
        #     # Standard scaling
        #     # print(f"  -> Scaling power by {scale_factor:.4f} for bus {int(bus)} (mBase={current_mbase:.2f}).")
        #     pass
            
        # Apply scaling if needed (scale_factor will be 1.0 if anomaly fix was applied)
        pg *= 1#scale_factor
        qg *= 1#scale_factor
        qmax *= 1#scale_factor
        qmin *= 1#scale_factor
        pmax *= 1#scale_factor
        pmin *= 1#scale_factor
        
        # mBase is always updated to the new system base for the resulting line
        final_mbase = new_sbase_mva

        return {
            'bus': int(bus),
            'Pg': pg, 'Qg': qg, 'Qmax': qmax, 'Qmin': qmin, 
            'Vg': vg, 'mBase': final_mbase, 'status': int(status), 
            'Pmax': pmax, 'Pmin': pmin,
            'original_mbase': original_mbase,
            'cost_values': None # Will be filled in the cost pass
        }
        
    except ValueError as e:
        print(f"Warning: Could not parse generator data line: {line_string.strip()}. Error: {e}")
        return None

def extract_cost_data(line_string):
    """
    Extracts relevant cost coefficients from a gencost line.
    Returns a list of [c2, c1, c0] values if Type 2, or None otherwise.
    """
    clean_line = line_string.strip().rstrip(';')
    values = re.split(r'\s+', clean_line)
    
    if not values or len(values) < 7 or line_string.strip().startswith('%'):
        return None 

    try:
        cost_type = int(float(values[0]))
        # Only process Type 2 (Polynomial) costs with at least c2, c1, c0
        if cost_type == 2:
            # Return only the coefficients c2, c1, c0
            return [float(values[4]), float(values[5]), float(values[6])]
        return None
    except ValueError:
        return None

# ----------------- AGGREGATION FUNCTION (The new, separated step) -----------------

def calculate_lopf_c1(original_mbase, Pmax_pu, Pmin_pu, c2_old, c1_old, new_sbase_mva=NEW_SBASE_MVA):
    """
    Calculates the equivalent linear cost coefficient (c'1) by finding the average
    marginal cost over the generator's operating range (Pmin_pu to Pmax_pu).
    
    CRITICAL FIX: Checks for the mBase anomaly. If detected, the effective
    scaling factor for the cost coefficients is set to 1.0.
    
    Formula: C'1 = C1_new + C2_new * (Pmax_pu + Pmin_pu)
    """
    
    # C2 scales by S_old / S_new
    c2_new_pu_cost = c2_old #* physical_scale_factor * physical_scale_factor
    
    # C1 scales by S_new / S_old
    c1_new_pu_cost = c1_old #* physical_scale_factor

    # 2. Calculate the new equivalent linear coefficient (C'1)
    # The Pmax_pu and Pmin_pu passed here are already the FINAL, CORRECT PU values
    # from the normalize_generator_data function.
    
    if Pmax_pu - Pmin_pu <= 1e-6:
        # Fixed output: Use the marginal cost at Pmax_pu
        # NOTE: When using Pmax_pu here, we must use the already scaled c2_new_pu_cost and c1_new_pu_cost!
        c1_equivalent = c1_new_pu_cost + 2 * c2_new_pu_cost * Pmax_pu
    else:
        # Standard average marginal cost over the full range
        c1_equivalent = c1_new_pu_cost + c2_new_pu_cost * (Pmax_pu + Pmin_pu)
        
    return c1_equivalent

def aggregate_generators(gen_records):
    """
    Groups and aggregates generator records by bus ID.
    Sums limits/dispatches and calculates the Pmax-weighted average LOPF cost factor c'1.
    """
    print("\n--- PASS 2: AGGREGATION STEP: Grouping by bus and calculating weighted average C'1 ---")

    # Group by bus
    buses = {}
    for record in gen_records:
        bus_id = record['bus']
        if bus_id not in buses:
            buses[bus_id] = []
        buses[bus_id].append(record)

    aggregated_records = []
    
    for bus_id, units in buses.items():
        # Calculate individual LOPF C'1 for each unit first
        for unit in units:
            # cost_values is [c2, c1, c0]
            c2_old, c1_old, _ = unit['cost_values'] 
            unit['c1_lopf'] = calculate_lopf_c1(
                unit['original_mbase'], unit['Pmax'], unit['Pmin'], c2_old, c1_old
            )
            
        # Sum limits and dispatches
        total_pmax = sum(unit['Pmax'] for unit in units)
        total_pmin = sum(unit['Pmin'] for unit in units)
        
        # Calculate weighted average C'1
        # Weighting by Pmax ensures larger units have a greater influence on the final cost factor.
        if total_pmax > 1e-6:
            weighted_c1_sum = sum(unit['c1_lopf'] * unit['Pmax'] for unit in units)
            c1_lopf_agg = weighted_c1_sum / total_pmax
        else:
            # If Pmax is near zero, just take the cost of the first unit
            c1_lopf_agg = units[0]['c1_lopf'] if units else 0.0

        # Create the aggregated record
        agg_record = {
            'bus': bus_id,
            'Pg': sum(unit['Pg'] for unit in units),
            'Qg': sum(unit['Qg'] for unit in units),
            'Qmax': sum(unit['Qmax'] for unit in units),
            'Qmin': sum(unit['Qmin'] for unit in units),
            'Vg': units[0]['Vg'], 
            'mBase': NEW_SBASE_MVA,
            'status': 1,
            'Pmax': total_pmax,
            'Pmin': total_pmin,
            'c1_lopf_equivalent': c1_lopf_agg
        }
        aggregated_records.append(agg_record)
        
    return aggregated_records

# ----------------- PASS 3 HELPERS: FORMATTING -----------------

def format_aggregated_gen_line(record):
    """Formats an aggregated generator record back into an mpc.gen line."""
    # Ensure consistency in float/int types for formatting
    return (
        f" {record['bus']:4d}" +          # Bus (col 1)
        f" {record['Pg']:12.6g}" +            # Pg (col 2)
        f" {record['Qg']:12.6g}" +            # Qg (col 3)
        f" {record['Qmax']:10.4g}" +            # Qmax (col 4)
        f" {record['Qmin']:10.4g}" +            # Qmin (col 5)
        f" {record['Vg']:8.5f}" +             # Vg (col 6)
        f" {record['mBase']:8.2f}" +             # mBase (col 7) - NOW 100.00
        f" {record['status']:2d}" +          # status (col 8)
        f" {record['Pmax']:12.6g}" +            # Pmax (col 9)
        f" {record['Pmin']:12.6g}" +            # Pmin (col 10)
        ';\n'
    )

def format_aggregated_cost_line(record):
    """Formats an aggregated cost record back into an mpc.gencost line (LOPF format)."""
    # LOPF requires Type 2, n=3, with c2=0, c0=0, and the calculated c1
    new_line_content = (
        f" {2:1d}" +                          # type (col 1, must be 2)
        f" {0.0:12.6g}" +                     # startup (col 2)
        f" {0.0:12.6g}" +                     # shutdown (col 3)
        f" {3:2d}" +                          # n (col 4, number of coefficients, 3)
        f" {0.0:12.6g}" +                     # c2 (col 5) - SET TO ZERO
        f" {record['c1_lopf_equivalent']:12.6g}" + # c1 (col 6) - CALCULATED EQUIVALENT
        f" {0.0:12.6g}"                        # c0 (col 7) - SET TO ZERO
    )
    # The lines inside gencost are often indented
    return ' ' * 4 + new_line_content.strip() + ';\n'

# ----------------- MAIN PROCESSING FUNCTION -----------------

def filter_and_normalize_case(input_filepath, output_filepath):
    """
    Reads, normalizes, aggregates, and writes the output case file.
    """
    try:
        with open(input_filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return

    # --- Data Containers ---
    gen_records = []
    
    # --- Pass 1: Extract, Normalize, and Pair Data ---
    
    file_sections = {'before_gen': [], 'gen': [], 'gencost': [], 'after_gencost': []}
    current_section = 'before_gen'
    
    print("--- PASS 1: Extracting and Normalizing Generator Data ---")

    # State machine to parse file sections
    for i, line in enumerate(lines):
        if re.search(r'^\s*mpc\.gen\s*=\s*\[', line):
            current_section = 'gen'
        elif re.search(r'^\s*mpc\.gencost\s*=\s*\[', line):
            current_section = 'gencost'
        
        # Capture the entire line for section reconstruction
        if current_section == 'before_gen':
            file_sections['before_gen'].append(line)
        elif current_section == 'gen':
            file_sections['gen'].append(line)
            if re.search(r'^\]\s*;', line):
                current_section = 'between_sections'
        elif current_section == 'gencost':
            file_sections['gencost'].append(line)
            if re.search(r'^\]\s*;', line):
                current_section = 'after_gencost'
        elif current_section == 'between_sections':
            # Handle lines between 'mpc.gen' end and 'mpc.gencost' start
            if re.search(r'^\s*mpc\.gencost\s*=\s*\[', line):
                file_sections['gencost'].append(line) # already handled by state change
            else:
                 file_sections['before_gen'].append(line) # Treat as 'before_gen' for simplicity
        elif current_section == 'after_gencost':
            file_sections['after_gencost'].append(line)
    
    # Process Gen Data and create records
    raw_gen_lines = file_sections['gen'][1:-1] # Exclude start/end markers
    raw_cost_lines = file_sections['gencost'][1:-1] # Exclude start/end markers
    
    for line in raw_gen_lines:
        record = normalize_generator_data(line)
        if record:
            gen_records.append(record)

    # Pair cost data to records (assuming 1-to-1 sequential mapping)
    kept_gen_index = 0
    for i, line in enumerate(raw_cost_lines):
        cost_values = extract_cost_data(line)
        # We only care about cost data for in-service generators that were kept
        if kept_gen_index < len(gen_records):
            gen_records[kept_gen_index]['cost_values'] = cost_values
            kept_gen_index += 1
        
    # --- Pass 2: Aggregation (The separate function) ---
    
    # Filter out any records that didn't have valid cost data
    final_gen_records = [r for r in gen_records if r['cost_values'] is not None]
    
    aggregated_data = aggregate_generators(final_gen_records)
    
    print(f"Total original in-service generators processed: {len(final_gen_records)}")
    print(f"Total final aggregated generators (unique buses): {len(aggregated_data)}")

    # --- Pass 3: Construct the New File Content ---
    
    new_lines = []
    
    # 1. Start with everything before mpc.gen (includes header and between sections)
    new_lines.extend(file_sections['before_gen'])
    
    # 2. Add the new mpc.gen section
    if file_sections['gen']:
        new_lines.append(file_sections['gen'][0]) # Add mpc.gen = [
        new_gen_lines = [format_aggregated_gen_line(r) for r in aggregated_data]
        new_lines.extend(new_gen_lines)
        new_lines.append(file_sections['gen'][-1]) # Add ];
    
    # 3. Add the new mpc.gencost section
    if file_sections['gencost']:
        new_lines.append(file_sections['gencost'][0]) # Add mpc.gencost = [
        new_cost_lines = [format_aggregated_cost_line(r) for r in aggregated_data]
        new_lines.extend(new_cost_lines)
        new_lines.append(file_sections['gencost'][-1]) # Add ];
    
    # 4. Add everything after mpc.gencost
    new_lines.extend(file_sections['after_gencost'])
    
    # --- 5. Write the New File ---
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_filepath, 'w') as f:
            f.writelines(new_lines)

        print(f"\nSuccessfully processed {os.path.basename(input_filepath)}.")
        print(f"New case file saved to {output_filepath}")
        
    except Exception as e:
        print(f"\nError writing output file: {e}")







BASE_DIR = '/home/bagir/Documents/1) Projects/2) AC verification/MinMax/pglib-opf'
FILENAME = 'pglib_opf_case793_goc.m'

input_file = os.path.join(BASE_DIR, FILENAME)
output_file = os.path.join(BASE_DIR, FILENAME.replace('.m', '_cleaned.m'))

# Run the function
filter_and_normalize_case(input_file, output_file)