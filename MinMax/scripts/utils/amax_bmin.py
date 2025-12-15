import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection    
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons # Import CheckButtons widget


import matplotlib as mpl


plt.style.use(['C:/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/0) DTU Admin/5) Templates/thesis.mplstyle'])
plt.rcParams['text.usetex'] = False


from matplotlib import font_manager

font_manager.fontManager.addfont('C:/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/0) DTU Admin/5) Templates/palr45w.ttf')
plt.rcParams['font.family'] = 'Palatino' # Set the font globally
#plt.rcParams['font.family'] = 'sans-serif'




def plot_animated_ambm_approximation():
    """
    Creates an animated plot visualizing a rotating unit vector, its true magnitude,
    and its Alpha Max Beta Min (AMBM) approximation point, along with a dynamically scaling
    AMBM contour that represents the current approximation value, and a trailing line
    for the AMBM approximation point.
    """

    # 1. Define AMBM approximation parameters
    # Using parameters for guaranteed under-approximation (scaled L1 norm)
    # alpha = 0.7071 # 1/np.sqrt(2) # This is 1/sqrt(2)
    # beta = 0.7071 # 1/np.sqrt(2)  # This is 1/sqrt(2)
    
    alpha = 1  # 0.9604
    beta = np.sqrt(2) - 1 # 0.3978
    
    # alpha = 0.9604
    # beta = 0.3978
    
    # 2. Setup the static plot elements (True Unit Circle)
    fig, ax = plt.subplots(figsize=(10, 10))

    # True Unit Circle (Radius 1) - This remains static
    angles_radians_static = np.linspace(0, 2 * np.pi, 361)
    true_circle_x = np.cos(angles_radians_static)
    true_circle_y = np.sin(angles_radians_static)
    ax.plot(true_circle_x, true_circle_y, color='blue', linestyle='-', label='True Magnitude (Radius 1)', linewidth=2)

    # Set plot limits and labels
    # Limits based on the maximum possible value the purple point (and thus the red contour) will reach.
    # For alpha=beta=1/sqrt(2), the AMBM value of a unit vector ranges from 1/sqrt(2) to 1.
    # The dynamic_contour_magnitude will range from alpha*(1/sqrt(2)) = 0.5 to alpha*1 = 0.7071.
    # So, a limit of 1.1 is still appropriate to encompass the true circle and the AMBM points.
    limit = 1.1 
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Rotating Vector with Dynamic AMBM Contour')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Initialize legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()

    # 3. Initialize animated elements
    # Rotating vector (quiver)
    vec_quiver = ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green', label='Rotating Unit Vector', width=0.005)
    
    # AMBM approximation point - this point will always be on the boundary of the red diamond
    ambm_point, = ax.plot([], [], 'x', color='purple', markersize=10, label='AMBM Approx Point')

    # AMBM dynamic contour line - Initialized empty, will be updated
    # Label updated to reflect active alpha/beta values
    ambm_contour_line, = ax.plot([], [], color='red', linestyle='--',
                                 label=f'Dynamic AMBM Contour (α={alpha}, β={beta})', linewidth=2)

    # Trailing line for the AMBM approximation point
    ambm_trail_x, ambm_trail_y = [], [] # Lists to store trail coordinates
    ambm_trail_line, = ax.plot([], [], color='orange', linestyle='-', linewidth=1.5, label='AMBM Approx Trail')

    # Update legend to include animated elements
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    
    # Polygon for error shading
    ambm_error_patch = Polygon([[0, 0]], closed=True, color='red', alpha=0.2)
    ax.add_patch(ambm_error_patch)

    # 4. Define the update function for the animation
    def update(frame):
        nonlocal ambm_trail_x, ambm_trail_y # Declare as nonlocal to modify outside scope

        angle_degrees = frame
        angle_radians = np.deg2rad(angle_degrees)

        # Current unit vector components (cos_theta, sin_theta)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)

        # Update rotating vector
        vec_quiver.set_UVC(cos_theta, sin_theta)

        # Calculate AMBM approximation for the current unit vector (x,y)
        abs_current_real = np.abs(cos_theta)
        abs_current_imag = np.abs(sin_theta)
        max_val = np.maximum(abs_current_real, abs_current_imag)
        min_val = np.minimum(abs_current_real, abs_current_imag)
        
        # This is the AMBM value of the UNIT vector (magnitude 1)
        ambm_value_of_unit_vector = alpha * max_val + beta * min_val

        # Calculate the coordinates of the AMBM approximation point
        # This point lies in the direction of the current vector, scaled by its AMBM value
        ambm_point_x = cos_theta * ambm_value_of_unit_vector
        ambm_point_y = sin_theta * ambm_value_of_unit_vector
        
        # Update the AMBM approximation point
        ambm_point.set_data([ambm_point_x], [ambm_point_y])

        # --- Update the Trailing Line ---
        ambm_trail_x.append(ambm_point_x)
        ambm_trail_y.append(ambm_point_y)
        ambm_trail_line.set_data(ambm_trail_x, ambm_trail_y)
        # --- End of Trailing Line Update ---

        # --- Calculate and Scale the AMBM Contour ---
        # The target magnitude for the dynamic contour is derived such that the purple point
        # lies on its boundary after rotation.
        dynamic_contour_magnitude = alpha * ambm_value_of_unit_vector
        num_points_per_segment = 50

        # Generate base points for the diamond (aligned with axes)
        # These calculations now use dynamic_contour_magnitude
        # Handle division by zero if alpha or (alpha+beta) is zero (though not with current params)
        x_at_45_deg_dynamic = dynamic_contour_magnitude / (alpha + beta) if (alpha + beta) != 0 else 0
        x_on_axis_dynamic = dynamic_contour_magnitude / alpha if alpha != 0 else 0

        base_ambm_x_dynamic = []
        base_ambm_y_dynamic = []

        # Quadrant 1 (x >= 0, y >= 0)
        x_seg1_dynamic = np.linspace(x_on_axis_dynamic, x_at_45_deg_dynamic, num_points_per_segment)
        # Handle division by zero for beta if beta is 0 (though not with current params)
        y_seg1_dynamic = (dynamic_contour_magnitude - alpha * x_seg1_dynamic) / beta if beta != 0 else np.zeros_like(x_seg1_dynamic)
        base_ambm_x_dynamic.extend(x_seg1_dynamic.tolist())
        base_ambm_y_dynamic.extend(y_seg1_dynamic.tolist())

        y_seg2_dynamic = np.linspace(x_at_45_deg_dynamic, x_on_axis_dynamic, num_points_per_segment)
        x_seg2_dynamic = (dynamic_contour_magnitude - alpha * y_seg2_dynamic) / beta if beta != 0 else np.zeros_like(y_seg2_dynamic)
        base_ambm_x_dynamic.extend(x_seg2_dynamic.tolist())
        base_ambm_y_dynamic.extend(y_seg2_dynamic.tolist())
        
        # Mirror for other quadrants (and ensure order for continuous line)
        # Quadrant 2 (x < 0, y > 0)
        base_ambm_x_dynamic.extend([-x for x in x_seg2_dynamic[::-1]])
        base_ambm_y_dynamic.extend(y_seg2_dynamic[::-1])
        base_ambm_x_dynamic.extend([-x for x in x_seg1_dynamic[::-1]])
        base_ambm_y_dynamic.extend(y_seg1_dynamic[::-1])

        # Quadrant 3 (x < 0, y < 0)
        base_ambm_x_dynamic.extend([-x for x in x_seg1_dynamic])
        base_ambm_y_dynamic.extend([-y for y in y_seg1_dynamic])
        base_ambm_x_dynamic.extend([-x for x in x_seg2_dynamic])
        base_ambm_y_dynamic.extend([-y for y in y_seg2_dynamic])

        # Quadrant 4 (x > 0, y < 0)
        base_ambm_x_dynamic.extend(x_seg2_dynamic[::-1])
        base_ambm_y_dynamic.extend([-y for y in y_seg2_dynamic[::-1]])
        base_ambm_x_dynamic.extend(x_seg1_dynamic[::-1])
        base_ambm_y_dynamic.extend([-y for y in y_seg1_dynamic[::-1]])

        # --- Rotate the AMBM Contour to align with the vector direction ---
        cos_a = np.cos(angle_radians)
        sin_a = np.sin(angle_radians)

        x_rotated = []
        y_rotated = []
        for x_base, y_base in zip(base_ambm_x_dynamic, base_ambm_y_dynamic):
            x_r = cos_a * x_base - sin_a * y_base
            y_r = sin_a * x_base + cos_a * y_base
            x_rotated.append(x_r)
            y_rotated.append(y_r)

        ambm_contour_line.set_data(x_rotated, y_rotated)   
    
        # --- End of Dynamic AMBM Contour ---
        
        # Compute unit circle points along the trail direction
        unit_trail_x = [x / np.hypot(x, y) if np.hypot(x, y) != 0 else 0 for x, y in zip(ambm_trail_x, ambm_trail_y)]
        unit_trail_y = [y / np.hypot(x, y) if np.hypot(x, y) != 0 else 0 for x, y in zip(ambm_trail_x, ambm_trail_y)]

        # Build the polygon between the trail and the unit circle
        if len(ambm_trail_x) > 1:
            fill_x = ambm_trail_x + unit_trail_x[::-1]
            fill_y = ambm_trail_y + unit_trail_y[::-1]
            ambm_error_patch.set_xy(list(zip(fill_x, fill_y)))

        # Update title for current angle
        ax.set_title(f'Rotating Vector with Dynamic AMBM Contour (Angle: {angle_degrees}°)')

        # Return all updated artists
        return vec_quiver, ambm_point, ambm_contour_line, ambm_trail_line, ambm_error_patch, ax.title


    # 5. Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(361), interval=50, blit=False)

    plt.show()

# Run the animation
# plot_animated_ambm_approximation()











def plot_animated_ambm_approximation():
    """
    Creates an animated plot visualizing a rotating unit vector, its true magnitude,
    and its Tight Lower Bound (max of Linf and scaled L1) approximation point.
    It shows:
    - The true unit circle.
    - A rotating unit vector.
    - The combined approximation point (purple cross).
    - The dynamically scaled and rotated Linf contour (green dotted).
    - The dynamically scaled and rotated scaled L1 contour (dark green dotted).
    - A trailing line of the approximation point.
    - Shaded area representing the error.
    - A subplot showing the relative error over time.
    """

    # --- 1. Define AMBM approximation parameters for the Tight Lower Bound ---
    # These define the two components whose maximum forms the combined approximation.
    alpha_Linf = 1.0
    beta_Linf = 0.0
    # These parameters define how the max/min components are weighted for the *value* calculation
    alpha_L1_scaled_value = 1.0 / np.sqrt(2) # ~0.7071
    beta_L1_scaled_value = 1.0 / np.sqrt(2)  # ~0.7071
    # -----------------------------
    
    # # ----- over approximation
    # # These define the two components whose maximum forms the combined approximation.
    # alpha_Linf = 1.0
    # beta_Linf = 1.0
    # # These parameters define how the max/min components are weighted for the *value* calculation
    # alpha_L1_scaled_value = 1.0  # ~0.7071
    # beta_L1_scaled_value = np.sqrt(2) - 1  # ~0.7071
    # # ------------------------
    
    label_approx = r'Tight Lower Bound (max($L_\infty$, scaled $L_1$))'
    point_color = 'purple'
    trail_color = 'orange'
    patch_color = 'red' # Error patch color

    # --- 2. Setup the figure and subplots ---
    fig, (ax_polar, ax_error) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.3) # Adjust vertical spacing between subplots

    # --- Polar Plot (ax_polar) ---
    # True Unit Circle (Radius 1) - This remains static
    angles_radians_static = np.linspace(0, 2 * np.pi, 361)
    true_circle_x = np.cos(angles_radians_static)
    true_circle_y = np.sin(angles_radians_static)
    ax_polar.plot(true_circle_x, true_circle_y, color='blue', linestyle='-', label='True Magnitude (Radius 1)', linewidth=2)

    # Set plot limits and labels for polar plot
    # The combined AMBM value for a unit vector ranges from approx 0.9239 (at 22.5 deg) to 1.0 (at 0/45 deg).
    # The individual component contours will be scaled by their own values, which can be smaller.
    limit = 1.1 
    ax_polar.set_xlim([-limit, limit])
    ax_polar.set_ylim([-limit, limit])
    ax_polar.set_aspect('equal', adjustable='box')
    ax_polar.axhline(0, color='grey', lw=0.5)
    ax_polar.axvline(0, color='grey', lw=0.5)
    ax_polar.set_xlabel('Real Part')
    ax_polar.set_ylabel('Imaginary Part')
    ax_polar.set_title('Rotating Vector with Dynamic AMBM Contour')
    ax_polar.grid(True, linestyle='--', alpha=0.6)
    
    # Initialize legend for polar plot
    handles, labels = ax_polar.get_legend_handles_labels()
    ax_polar.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    
    # 3. Initialize animated elements for polar plot
    # Rotating vector (quiver)
    vec_quiver = ax_polar.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green', label='Rotating Unit Vector', width=0.005)
    
    # Combined AMBM approximation point
    ambm_point, = ax_polar.plot([], [], 'x', color=point_color, markersize=10, label=f'{label_approx} Point')

    # Two dynamic contour lines for the components of the under-approximation
    ambm_contour_line_linf, = ax_polar.plot([], [], color='green', linestyle=':',
                                     label=r'Dynamic $L_\infty$ Contour', linewidth=2)
    ambm_contour_line_l1, = ax_polar.plot([], [], color='darkgreen', linestyle=':',
                                     label=r'Dynamic Scaled $L_1$ Contour', linewidth=2)

    # Trailing line for the AMBM approximation point
    ambm_trail_x, ambm_trail_y = [], [] # Lists to store trail coordinates
    ambm_trail_line, = ax_polar.plot([], [], color=trail_color, linestyle='-', linewidth=1.5, label=f'{label_approx} Trail')

    # Update legend to include animated elements
    handles, labels = ax_polar.get_legend_handles_labels()
    ax_polar.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    
    # Polygon for error shading
    ambm_error_patch = Polygon([[0, 0]], closed=True, color=patch_color, alpha=0.2, label=f'{label_approx} Error')
    ax_polar.add_patch(ambm_error_patch)

    # --- Error Plot (ax_error) ---
    error_angles = []
    relative_error_values = []
    error_line, = ax_error.plot([], [], color=point_color, label=f'{label_approx} Relative Error') # Line for relative error

    ax_error.set_xlim([0, 360])
    # For the tight lower bound, max error is approx 0.0761 (at 22.5 deg). Min error is 0.
    ax_error.set_ylim([-0.05, 0.1]) # Adjusted to show non-negative error clearly
    ax_error.set_xlabel('Angle (degrees)')
    ax_error.set_ylabel('Relative Error (True - Approx) / True')
    ax_error.set_title('Relative Approximation Error')
    ax_error.grid(True, linestyle='--', alpha=0.6)
    ax_error.axhline(0, color='black', linestyle='-', linewidth=0.8) # Add a zero line
    ax_error.legend(loc='upper right')

    plt.tight_layout() # Re-adjust layout after adding second subplot

    # --- 4. Helper function to generate points for a single AMBM diamond contour ---
    def generate_ambm_diamond_contour(alpha_val, beta_val, magnitude, num_points_per_segment=50):
        # Handle cases where alpha or alpha+beta might be zero
        if alpha_val == 0 and beta_val == 0:
            return [0], [0]
        if magnitude == 0:
            return [0], [0] # Single point at origin for zero magnitude

        # Calculate the key points for the diamond
        # Point on axes (where min_val is 0) => alpha * M = magnitude  => M = magnitude / alpha
        on_axis_val = magnitude / alpha_val if alpha_val != 0 else float('inf')

        # Point on 45-degree lines (where min_val = max_val)
        # alpha * M + beta * M = magnitude => (alpha + beta) * M = magnitude => M = magnitude / (alpha + beta)
        at_45_deg_val = magnitude / (alpha_val + beta_val) if (alpha_val + beta_val) != 0 else float('inf')

        # Define the 8 vertices of the diamond in order for plotting a continuous line
        vertices = [
            (on_axis_val, 0),                       # 0 deg (positive X axis)
            (at_45_deg_val, at_45_deg_val),         # 45 deg (Q1 diagonal)
            (0, on_axis_val),                       # 90 deg (positive Y axis)
            (-at_45_deg_val, at_45_deg_val),        # 135 deg (Q2 diagonal)
            (-on_axis_val, 0),                      # 180 deg (negative X axis)
            (-at_45_deg_val, -at_45_deg_val),       # 225 deg (Q3 diagonal)
            (0, -on_axis_val),                      # 270 deg (negative Y axis)
            (at_45_deg_val, -at_45_deg_val),        # 315 deg (Q4 diagonal)
            (on_axis_val, 0)                        # Close the loop back to 0 deg
        ]

        contour_x = []
        contour_y = []

        # Iterate through segments defined by vertices
        for i in range(len(vertices) - 1):
            start_x, start_y = vertices[i]
            end_x, end_y = vertices[i+1]

            # Generate points along the segment
            # Ensure the first point of the next segment is not duplicated
            segment_x = np.linspace(start_x, end_x, num_points_per_segment, endpoint=False)
            segment_y = np.linspace(start_y, end_y, num_points_per_segment, endpoint=False)

            contour_x.extend(segment_x.tolist())
            contour_y.extend(segment_y.tolist())
        
        # Add the very last point to close the loop (which was excluded by endpoint=False)
        contour_x.append(vertices[-1][0])
        contour_y.append(vertices[-1][1])

        return contour_x, contour_y

    # --- 5. Define the update function for the animation ---
    def update(frame):
        nonlocal ambm_trail_x, ambm_trail_y, error_angles, relative_error_values

        angle_degrees = frame
        angle_radians = np.deg2rad(angle_degrees)

        # Current unit vector components (cos_theta, sin_theta)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        true_norm = 1.0 # For a unit vector

        # Update rotating vector
        vec_quiver.set_UVC(cos_theta, sin_theta)

        # Calculate AMBM approximation for the current unit vector (x,y)
        abs_current_real = np.abs(cos_theta)
        abs_current_imag = np.abs(sin_theta)
        max_val = np.maximum(abs_current_real, abs_current_imag)
        min_val = np.minimum(abs_current_real, abs_current_imag)
        
        # Calculate the two components of the tight lower bound for the UNIT vector
        L_inf_val_unit = alpha_Linf * max_val + beta_Linf * min_val
        # Use the separate parameters for the *value* calculation
        L1_scaled_val_unit = alpha_L1_scaled_value * max_val + beta_L1_scaled_value * min_val

        # The combined AMBM value is the maximum of the two (this is the magnitude of the purple point)
        combined_ambm_value = np.maximum(L_inf_val_unit, L1_scaled_val_unit)

        # Calculate the coordinates of the AMBM approximation point
        ambm_point_x = cos_theta * combined_ambm_value
        ambm_point_y = sin_theta * combined_ambm_value
        
        # Update the AMBM approximation point
        ambm_point.set_data([ambm_point_x], [ambm_point_y])

        # --- Update the Trailing Line ---
        ambm_trail_x.append(ambm_point_x)
        ambm_trail_y.append(ambm_point_y)
        ambm_trail_line.set_data(ambm_trail_x, ambm_trail_y)
        # --- End of Trailing Line Update ---

        # --- Calculate and Scale the Linf Contour ---
        # The contour should be scaled by its *own* approximation value for the unit vector.
        dynamic_contour_magnitude_linf = L_inf_val_unit 
        full_x_linf, full_y_linf = generate_ambm_diamond_contour(alpha_Linf, beta_Linf, dynamic_contour_magnitude_linf)
        
        x_rotated_linf = [cos_theta * x_b - sin_theta * y_b for x_b, y_b in zip(full_x_linf, full_y_linf)]
        y_rotated_linf = [sin_theta * x_b + cos_theta * y_b for x_b, y_b in zip(full_x_linf, full_y_linf)]
        ambm_contour_line_linf.set_data(x_rotated_linf, y_rotated_linf)  

        # --- Calculate and Scale the Scaled L1 Contour ---
        # The contour for L1 should be based on its standard form (alpha=1, beta=1),
        # and *then* scaled by the calculated L1_scaled_val_unit.
        # This is the key fix!
        dynamic_contour_magnitude_l1_for_plotting = L1_scaled_val_unit # Use the calculated value as the magnitude
        
        # Generate a standard L1 diamond (alpha=1, beta=1)
        # This defines the "shape" of the L1 diamond (sum of abs values)
        base_x_l1, base_y_l1 = generate_ambm_diamond_contour(1.0, 1.0, 1.0) # Generate a "unit" L1 diamond
        
        # Now, scale this base L1 diamond by the calculated L1_scaled_val_unit
        # This will make the L1 contour correctly represent the L1_scaled_value_unit
        full_x_l1 = [x * dynamic_contour_magnitude_l1_for_plotting for x in base_x_l1]
        full_y_l1 = [y * dynamic_contour_magnitude_l1_for_plotting for y in base_y_l1]
        
        x_rotated_l1 = [cos_theta * x_b - sin_theta * y_b for x_b, y_b in zip(full_x_l1, full_y_l1)]
        y_rotated_l1 = [sin_theta * x_b + cos_theta * y_b for x_b, y_b in zip(full_x_l1, full_y_l1)]
        ambm_contour_line_l1.set_data(x_rotated_l1, y_rotated_l1)  
    
        # --- End of Dynamic AMBM Contours ---
        
        # --- Update Error Patch ---
        # For under-approximation, the error patch should be between the approximation point
        # and the true unit circle point, connected to the origin.
        if combined_ambm_value != 0:
            # Origin -> Approx Point -> True Point -> Origin (for patch inside circle)
            fill_x = [0, ambm_point_x, cos_theta, 0] 
            fill_y = [0, ambm_point_y, sin_theta, 0]
            ambm_error_patch.set_xy(list(zip(fill_x, fill_y)))
        else:
            ambm_error_patch.set_xy([[0,0],[0,0],[0,0],[0,0]]) # Hide if point is at origin


        # --- Update Error Subplot ---
        error_angles.append(angle_degrees)
        
        # Relative error: (True - Approx) / True. This will be non-negative for under-approximation.
        relative_error = (true_norm - combined_ambm_value) / true_norm if true_norm != 0 else 0
        relative_error_values.append(relative_error)

        error_line.set_data(error_angles, relative_error_values)

        ax_polar.set_title(f'Rotating Vector with Dynamic AMBM Contour (Angle: {angle_degrees}°)')

        # Return all updated artists
        return vec_quiver, ambm_point, ambm_contour_line_linf, ambm_contour_line_l1, \
               ambm_trail_line, ambm_error_patch, error_line, ax_polar.title


    # 6. Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(361), interval=50, blit=False)

    plt.show()
    
    # # --- New: Separate Error Plot (0 to 90 degrees only) ---
    # # Convert to numpy for easy masking
    # error_angles_arr = np.array(error_angles)
    # relative_error_values_arr = np.array(relative_error_values)

    # # Mask for 0–90 degrees
    # mask = (error_angles_arr >= 0) & (error_angles_arr <= 90)

    # # Create new figure
    # fig2, ax_error_zoom = plt.subplots(figsize=(8, 5))
    # ax_error_zoom.plot(error_angles_arr[mask], relative_error_values_arr[mask],
    #                    color='purple', label='Relative Error (0°–90°)')
    # ax_error_zoom.set_xlim([0, 90])
    # ax_error_zoom.set_ylim([0, 0.5])  # Adjust depending on your max error
    # ax_error_zoom.set_xlabel('Angle (degrees)')
    # ax_error_zoom.set_ylabel('Relative Error')
    # ax_error_zoom.set_title('Relative Approximation Error (0°–90°)')
    # ax_error_zoom.grid(True, linestyle='--', alpha=0.6)
    # ax_error_zoom.axhline(0, color='black', linestyle='-', linewidth=0.8)
    # ax_error_zoom.legend()
    # plt.show()

# Run the animation
plot_animated_ambm_approximation()







def plot_animated_ambm_approximation():
    """
    Creates an animated plot visualizing a rotating unit vector, its true magnitude,
    and its Tight Over Bound (min of L1 and scaled Linf) approximation point.
    It shows:
    - The true unit circle.
    - A rotating unit vector.
    - The combined approximation point (purple cross).
    - The dynamically scaled and rotated L1 contour (dark green dotted).
    - The dynamically scaled and rotated scaled Linf contour (green dotted).
    - A trailing line of the approximation point.
    - Shaded area representing the error.
    - A subplot showing the relative error over time.
    """

    # --- 1. Define AMBM approximation parameters for the Tight Over Bound ---
    # Parameters for the L1 norm component (alpha=1, beta=1)
    alpha_L1_over = 1.0 # 0.9604
    beta_L1_over = 1.0 #  0.3978
    
    # Parameters for the scaled Linf component (alpha=1, beta=sqrt(2)-1)
    alpha_Linf_scaled_over = 1.0
    beta_Linf_scaled_over = np.sqrt(2) - 1 # ~0.4142

    label_approx = r'Tight Over Bound (min($L_1$, scaled $L_\infty$))' # Changed label
    point_color = 'purple'
    trail_color = 'orange'
    patch_color = 'blue' # Error patch color (changed for over-approx)

    # --- 2. Setup the figure and subplots ---
    fig, (ax_polar, ax_error) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.3) # Adjust vertical spacing between subplots

    # --- Polar Plot (ax_polar) ---
    # True Unit Circle (Radius 1) - This remains static
    angles_radians_static = np.linspace(0, 2 * np.pi, 361)
    true_circle_x = np.cos(angles_radians_static)
    true_circle_y = np.sin(angles_radians_static)
    ax_polar.plot(true_circle_x, true_circle_y, color='blue', linestyle='-', label='True Magnitude (Radius 1)', linewidth=2)

    # Set plot limits and labels for polar plot
    limit = 1.5 # Increased limit for over-approximation to be visible
    ax_polar.set_xlim([-limit, limit])
    ax_polar.set_ylim([-limit, limit])
    ax_polar.set_aspect('equal', adjustable='box')
    ax_polar.axhline(0, color='grey', lw=0.5)
    ax_polar.axvline(0, color='grey', lw=0.5)
    ax_polar.set_xlabel('Real Part')
    ax_polar.set_ylabel('Imaginary Part')
    ax_polar.set_title('Rotating Vector with Dynamic AMBM Over-Approximation Contour') # Changed title
    ax_polar.grid(True, linestyle='--', alpha=0.6)
    
    # Initialize legend for polar plot
    handles, labels = ax_polar.get_legend_handles_labels()
    ax_polar.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    
    # 3. Initialize animated elements for polar plot
    # Rotating vector (quiver)
    vec_quiver = ax_polar.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green', label='Rotating Unit Vector', width=0.005)
    
    # Combined AMBM approximation point
    ambm_point, = ax_polar.plot([], [], 'x', color=point_color, markersize=10, label=f'{label_approx} Point')

    # Two dynamic contour lines for the components of the over-approximation
    ambm_contour_line_l1_over, = ax_polar.plot([], [], color='darkgreen', linestyle=':',
                                     label=r'Dynamic $L_1$ Contour', linewidth=2) # Changed label/color
    ambm_contour_line_linf_over, = ax_polar.plot([], [], color='green', linestyle=':',
                                     label=r'Dynamic Scaled $L_\infty$ Contour', linewidth=2) # Changed label/color

    # Trailing line for the AMBM approximation point
    ambm_trail_x, ambm_trail_y = [], [] # Lists to store trail coordinates
    ambm_trail_line, = ax_polar.plot([], [], color=trail_color, linestyle='-', linewidth=1.5, label=f'{label_approx} Trail')

    # Update legend to include animated elements
    handles, labels = ax_polar.get_legend_handles_labels()
    ax_polar.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    
    # Polygon for error shading
    # For over-approximation, error is between true point and approx point, connected to origin
    ambm_error_patch = Polygon([[0, 0]], closed=True, color=patch_color, alpha=0.2, label=f'{label_approx} Error')
    ax_polar.add_patch(ambm_error_patch)

    # --- Error Plot (ax_error) ---
    error_angles = []
    relative_error_values = []
    error_line, = ax_error.plot([], [], color=point_color, label=f'{label_approx} Relative Error') # Line for relative error

    ax_error.set_xlim([0, 360])
    # For over approximation, min error is 0, max error is approx 0.4142 (at 45 deg for pure L1)
    ax_error.set_ylim([-0.05, 0.5]) # Adjusted for over-approximation error
    ax_error.set_xlabel('Angle (degrees)')
    ax_error.set_ylabel('Relative Error (Approx - True) / True') # Error calculation changed
    ax_error.set_title('Relative Approximation Error')
    ax_error.grid(True, linestyle='--', alpha=0.6)
    ax_error.axhline(0, color='black', linestyle='-', linewidth=0.8) # Add a zero line
    ax_error.legend(loc='upper right')

    plt.tight_layout() # Re-adjust layout after adding second subplot

    # --- 4. Helper function to generate points for a single AMBM diamond contour ---
    def generate_ambm_diamond_contour(alpha_val, beta_val, magnitude, num_points_per_segment=50):
        # Handle cases where alpha or alpha+beta might be zero
        if alpha_val == 0 and beta_val == 0:
            return [0], [0]
        if magnitude == 0:
            return [0], [0] # Single point at origin for zero magnitude

        # Calculate the key points for the diamond
        # Point on axes (where min_val is 0) => alpha * M = magnitude  => M = magnitude / alpha
        on_axis_val = magnitude / alpha_val if alpha_val != 0 else float('inf')

        # Point on 45-degree lines (where min_val = max_val)
        # alpha * M + beta * M = magnitude => (alpha + beta) * M = magnitude => M = magnitude / (alpha + beta)
        at_45_deg_val = magnitude / (alpha_val + beta_val) if (alpha_val + beta_val) != 0 else float('inf')

        # Define the 8 vertices of the diamond in order for plotting a continuous line
        vertices = [
            (on_axis_val, 0),                       # 0 deg (positive X axis)
            (at_45_deg_val, at_45_deg_val),         # 45 deg (Q1 diagonal)
            (0, on_axis_val),                       # 90 deg (positive Y axis)
            (-at_45_deg_val, at_45_deg_val),        # 135 deg (Q2 diagonal)
            (-on_axis_val, 0),                      # 180 deg (negative X axis)
            (-at_45_deg_val, -at_45_deg_val),       # 225 deg (Q3 diagonal)
            (0, -on_axis_val),                      # 270 deg (negative Y axis)
            (at_45_deg_val, -at_45_deg_val),        # 315 deg (Q4 diagonal)
            (on_axis_val, 0)                        # Close the loop back to 0 deg
        ]

        contour_x = []
        contour_y = []

        # Iterate through segments defined by vertices
        for i in range(len(vertices) - 1):
            start_x, start_y = vertices[i]
            end_x, end_y = vertices[i+1]

            # Generate points along the segment
            # Ensure the first point of the next segment is not duplicated
            segment_x = np.linspace(start_x, end_x, num_points_per_segment, endpoint=False)
            segment_y = np.linspace(start_y, end_y, num_points_per_segment, endpoint=False)

            contour_x.extend(segment_x.tolist())
            contour_y.extend(segment_y.tolist())
        
        # Add the very last point to close the loop (which was excluded by endpoint=False)
        contour_x.append(vertices[-1][0])
        contour_y.append(vertices[-1][1])

        return contour_x, contour_y

    # --- 5. Define the update function for the animation ---
    def update(frame):
        nonlocal ambm_trail_x, ambm_trail_y, error_angles, relative_error_values

        angle_degrees = frame
        angle_radians = np.deg2rad(angle_degrees)

        # Current unit vector components (cos_theta, sin_theta)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        true_norm = 1.0 # For a unit vector

        # Update rotating vector
        vec_quiver.set_UVC(cos_theta, sin_theta)

        # Calculate AMBM approximation for the current unit vector (x,y)
        abs_current_real = np.abs(cos_theta)
        abs_current_imag = np.abs(sin_theta)
        max_val = np.maximum(abs_current_real, abs_current_imag)
        min_val = np.minimum(abs_current_real, abs_current_imag)
        
        # Calculate the two components of the tight over bound for the UNIT vector
        # Using the new over-approximation parameters
        L1_val_unit_over = alpha_L1_over * max_val + beta_L1_over * min_val
        Linf_scaled_val_unit_over = alpha_Linf_scaled_over * max_val + beta_Linf_scaled_over * min_val

        # The combined AMBM value for over-approximation is the MINIMUM of the two
        combined_ambm_value = np.minimum(L1_val_unit_over, Linf_scaled_val_unit_over)

        # Calculate the coordinates of the AMBM approximation point
        ambm_point_x = cos_theta * combined_ambm_value
        ambm_point_y = sin_theta * combined_ambm_value
        
        # Update the AMBM approximation point
        ambm_point.set_data([ambm_point_x], [ambm_point_y])

        # --- Update the Trailing Line ---
        ambm_trail_x.append(ambm_point_x)
        ambm_trail_y.append(ambm_point_y)
        ambm_trail_line.set_data(ambm_trail_x, ambm_trail_y)
        # --- End of Trailing Line Update ---

        # --- Calculate and Scale the L1 Contour (alpha=1, beta=1) ---
        dynamic_contour_magnitude_l1_over = L1_val_unit_over 
        # For L1 contour, use alpha=1, beta=1 in generate_ambm_diamond_contour
        base_x_l1, base_y_l1 = generate_ambm_diamond_contour(1.0, 1.0, 1.0)
        full_x_l1 = [x * dynamic_contour_magnitude_l1_over for x in base_x_l1]
        full_y_l1 = [y * dynamic_contour_magnitude_l1_over for y in base_y_l1]
        
        x_rotated_l1 = [cos_theta * x_b - sin_theta * y_b for x_b, y_b in zip(full_x_l1, full_y_l1)]
        y_rotated_l1 = [sin_theta * x_b + cos_theta * y_b for x_b, y_b in zip(full_x_l1, full_y_l1)]
        ambm_contour_line_l1_over.set_data(x_rotated_l1, y_rotated_l1)

        # --- Calculate and Scale the Scaled Linf Contour (alpha=1, beta=sqrt(2)-1) ---
        dynamic_contour_magnitude_linf_scaled_over = Linf_scaled_val_unit_over
        # For scaled Linf contour, use alpha=1, beta=sqrt(2)-1 in generate_ambm_diamond_contour
        base_x_linf_scaled, base_y_linf_scaled = generate_ambm_diamond_contour(alpha_Linf_scaled_over, beta_Linf_scaled_over, 1.0)
        full_x_linf_scaled = [x * dynamic_contour_magnitude_linf_scaled_over for x in base_x_linf_scaled]
        full_y_linf_scaled = [y * dynamic_contour_magnitude_linf_scaled_over for y in base_y_linf_scaled]
        
        x_rotated_linf_scaled = [cos_theta * x_b - sin_theta * y_b for x_b, y_b in zip(full_x_linf_scaled, full_y_linf_scaled)]
        y_rotated_linf_scaled = [sin_theta * x_b + cos_theta * y_b for x_b, y_b in zip(full_x_linf_scaled, full_y_linf_scaled)]
        ambm_contour_line_linf_over.set_data(x_rotated_linf_scaled, y_rotated_linf_scaled)
    
        # --- End of Dynamic AMBM Contours ---
        
        # --- Update Error Patch ---
        # For over-approximation, the error patch should be between the true unit circle point
        # and the approximation point, connected to the origin.
        if combined_ambm_value != 0:
            fill_x = [0, cos_theta, ambm_point_x, 0] # Order changed for over-approx
            fill_y = [0, sin_theta, ambm_point_y, 0]
            ambm_error_patch.set_xy(list(zip(fill_x, fill_y)))
        else:
            ambm_error_patch.set_xy([[0,0],[0,0],[0,0],[0,0]]) # Hide if point is at origin


        # --- Update Error Subplot ---
        error_angles.append(angle_degrees)
        
        # Relative error: (Approx - True) / True. This will be non-negative for over-approximation.
        relative_error = (combined_ambm_value - true_norm) / true_norm if true_norm != 0 else 0 # Changed calculation
        relative_error_values.append(relative_error)

        error_line.set_data(error_angles, relative_error_values)

        ax_polar.set_title(f'Rotating Vector with Dynamic AMBM Over-Approximation (Angle: {angle_degrees}°)') # Changed title

        # Return all updated artists
        return vec_quiver, ambm_point, ambm_contour_line_l1_over, ambm_contour_line_linf_over, \
               ambm_trail_line, ambm_error_patch, error_line, ax_polar.title


    # 6. Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(361), interval=50, blit=False)

    plt.show()
    
    # # --- New: Separate Error Plot (0 to 90 degrees only) ---
    # # Convert to numpy for easy masking
    # error_angles_arr = np.array(error_angles)
    # relative_error_values_arr = np.array(relative_error_values)

    # # Mask for 0–90 degrees
    # mask = (error_angles_arr >= 0) & (error_angles_arr <= 90)

    # # Create new figure
    # fig2, ax_error_zoom = plt.subplots(figsize=(8, 5))
    # ax_error_zoom.plot(error_angles_arr[mask], relative_error_values_arr[mask],
    #                    color='purple', label='Relative Error (0°–90°)')
    # ax_error_zoom.set_xlim([0, 90])
    # ax_error_zoom.set_ylim([0, 0.5])  # Adjust depending on your max error
    # ax_error_zoom.set_xlabel('Angle (degrees)')
    # ax_error_zoom.set_ylabel('Relative Error')
    # ax_error_zoom.set_title('Relative Approximation Error (0°–90°)')
    # ax_error_zoom.grid(True, linestyle='--', alpha=0.6)
    # ax_error_zoom.axhline(0, color='black', linestyle='-', linewidth=0.8)
    # ax_error_zoom.legend()
    # plt.show()

# Run the animation
# plot_animated_ambm_approximation()



import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

def plot_static_over_under_errors_vertical():
    # --- Angle domain 0°–90° ---
    angles_deg = np.linspace(0, 90, 500)
    angles_rad = np.deg2rad(angles_deg)
    cos_theta = np.cos(angles_rad)
    sin_theta = np.sin(angles_rad)
    abs_real = np.abs(cos_theta)
    abs_imag = np.abs(sin_theta)
    max_val = np.maximum(abs_real, abs_imag)
    min_val = np.minimum(abs_real, abs_imag)

    # --- Over-approximation (tight upper bound) ---
    alpha, beta = 1.0, np.sqrt(2) - 1
    approx_over = alpha * max_val + beta * min_val
    rel_error_over = approx_over - 1.0
    avg_error_over = np.mean(rel_error_over)
    max_error_over = np.max(rel_error_over)

    # --- Under-approximation (tight lower bound) ---
    alpha_Linf_under, beta_Linf_under = 1.0, 0.0
    alpha_L1_under, beta_L1_under = 1.0/np.sqrt(2), 1.0/np.sqrt(2)
    L1_under = alpha_L1_under * max_val + beta_L1_under * min_val
    Linf_under = alpha_Linf_under * max_val + beta_Linf_under * min_val
    approx_under = np.maximum(L1_under, Linf_under)
    rel_error_under = 1.0 - approx_under
    avg_error_under = np.mean(rel_error_under)
    max_error_under = np.max(rel_error_under)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    tick_fontsize = 12  # increase tick font size

    # Over-approx
    ax1.plot(angles_deg, rel_error_over, color="black", lw=2)
    ax1.axhline(avg_error_over, color="blue", linestyle="--", lw=1.5)
    ax1.axhline(max_error_over, color="red", linestyle="--", lw=1.5)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 0.1)
    ax1.set_ylabel("Relative error", fontsize=12)
    ax1.set_title("Over-approximation error", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax1.yaxis.set_major_locator(MaxNLocator(4))  # fewer ticks
    ax1.tick_params(axis='both', labelsize=tick_fontsize)

    # annotate errors inside the plot
    ax1.text(40, avg_error_over + 0.002, f"avg: {avg_error_over*100:.2f}%", color="blue", fontsize=12)
    ax1.text(40, max_error_over + 0.002, f"max: {max_error_over*100:.2f}%", color="red", fontsize=12)

    # Under-approx
    ax2.plot(angles_deg, rel_error_under, color="black", lw=2)
    ax2.axhline(avg_error_under, color="blue", linestyle="--", lw=1.5)
    ax2.axhline(max_error_under, color="red", linestyle="--", lw=1.5)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 0.1)
    ax2.set_xlabel(r"$\theta$ (°)", fontsize=12)
    ax2.set_ylabel("Relative error", fontsize=12)
    ax2.set_title("Under-approximation error", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.tick_params(axis='both', labelsize=tick_fontsize)

    # annotate errors inside the plot
    ax2.text(40, avg_error_under + 0.002, f"avg: {avg_error_under*100:.2f}%", color="blue", fontsize=12)
    ax2.text(40, max_error_under + 0.002, f"max: {max_error_under*100:.2f}%", color="red", fontsize=12)

    plt.tight_layout()
    plt.show()

# Run
plot_static_over_under_errors_vertical()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FixedLocator

def plot_static_over_under_errors_quadrant(save_path = None):
    # --- Angle domain 0°–90° ---
    angles_deg = np.linspace(0, 90, 500)
    angles_rad = np.deg2rad(angles_deg)
    cos_theta = np.cos(angles_rad)
    sin_theta = np.sin(angles_rad)
    abs_real = np.abs(cos_theta)
    abs_imag = np.abs(sin_theta)
    max_val = np.maximum(abs_real, abs_imag)
    min_val = np.minimum(abs_real, abs_imag)

    # --- Over-approximation (tight upper bound) ---
    alpha, beta = 1.0, np.sqrt(2) - 1
    approx_over = alpha * max_val + beta * min_val
    rel_error_over = approx_over - 1.0
    avg_error_over = np.mean(rel_error_over)
    max_error_over = np.max(rel_error_over)

    # --- Under-approximation (tight lower bound) ---
    alpha_Linf_under, beta_Linf_under = 1.0, 0.0
    alpha_L1_under, beta_L1_under = 1.0/np.sqrt(2), 1.0/np.sqrt(2)
    L1_under = alpha_L1_under * max_val + beta_L1_under * min_val
    Linf_under = alpha_Linf_under * max_val + beta_Linf_under * min_val
    approx_under = np.maximum(L1_under, Linf_under)
    rel_error_under = 1.0 - approx_under
    avg_error_under = np.mean(rel_error_under)
    max_error_under = np.max(rel_error_under)

    # --- Colors ---
    c_over = "#d62728"   # red
    c_under = "#1f77b4"  # blue
    c_ref = "black"
    c_gray = "0.4"

    # --- Create figure with adjusted column widths ---
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6), 
        gridspec_kw={'width_ratios': [1, 1.3], 'hspace': 0.3, 'wspace': 0.05}
    )
    tick_fontsize = 13
    legend_fontsize = 12

    # ========== Over-approx quadrant ==========
    ax = axes[0,0]
    ax.plot(cos_theta, sin_theta, color=c_ref, lw=2, label="True norm = 1")
    x_over = cos_theta * approx_over
    y_over = sin_theta * approx_over
    ax.plot(x_over, y_over, color=c_over, linestyle="--", lw=2, label="Over-approx")
    ax.set_title("Over-approximation", fontsize=13)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax.legend(fontsize=legend_fontsize, loc="lower left")
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[1:])

    # ========== Under-approx quadrant ==========
    ax = axes[1,0]
    ax.plot(cos_theta, sin_theta, color=c_ref, lw=2, label="True norm = 1")
    x_under = cos_theta * approx_under
    y_under = sin_theta * approx_under
    ax.plot(x_under, y_under, color=c_under, linestyle="--", lw=2, label="Under-approx")
    ax.set_title("Under-approximation", fontsize=13)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    ax.legend(fontsize=legend_fontsize, loc="lower left")
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[1:])

    # ========== Over-approx error ==========
    ax = axes[0,1]
    ax.plot(angles_deg, rel_error_over, color=c_over, lw=2, label="Error curve")
    ax.axhline(avg_error_over, color=c_over, linestyle="--", lw=1.2, alpha=0.7,
               label=f"avg: {avg_error_over*100:.2f}%")
    ax.axhline(max_error_over, color=c_over, linestyle=":", lw=1.2, alpha=0.7,
               label=f"max: {max_error_over*100:.2f}%")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 0.1)
    ax.set_title("Error", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axhline(0, color=c_gray, lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.05, 0.10]))
    ax.legend(fontsize=legend_fontsize, loc="upper center")

    # ========== Under-approx error ==========
    ax = axes[1,1]
    ax.plot(angles_deg, rel_error_under, color=c_under, lw=2, label="Error curve")
    ax.axhline(avg_error_under, color=c_under, linestyle="--", lw=1.2, alpha=0.7,
               label=f"avg: {avg_error_under*100:.2f}%")
    ax.axhline(max_error_under, color=c_under, linestyle=":", lw=1.2, alpha=0.7,
               label=f"max: {max_error_under*100:.2f}%")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel(r"$\theta$ (°)", fontsize=13)
    ax.set_title("Error", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axhline(0, color=c_gray, lw=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.05, 0.10]))
    ax.legend(fontsize=legend_fontsize, loc="upper center")

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight")   # vector
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")   # vector
        fig.savefig(f"{save_path}.png", dpi=600, bbox_inches="tight")  # high-res raster
    plt.show()

# Run
save_path = r"C:\Users\bagir\OneDrive - Danmarks Tekniske Universitet\Dokumenter\1) Projects\6) AC-OPF verification\amax_bmin"

plot_static_over_under_errors_quadrant(save_path)








# def plot_animated_ambm_approximation():
#     """
#     Creates an animated plot visualizing a rotating unit vector, its true magnitude,
#     and its Tight Lower Bound (max of Linf and scaled L1) approximation point.
#     It shows:
#     - The true unit circle.
#     - A rotating unit vector.
#     - The combined approximation point (purple cross).
#     - The dynamically scaled and rotated combined Tight Lower Bound contour (green dotted).
#     - A trailing line of the approximation point.
#     - Shaded area representing the error.
#     - A subplot showing the relative error over time.
#     """

#     # --- 1. Define AMBM approximation parameters for the Tight Lower Bound ---
#     # These define the two components whose maximum forms the combined approximation.
#     alpha_Linf = 1.0
#     beta_Linf = 0.0
#     alpha_L1_scaled = 1.0 / np.sqrt(2) # ~0.7071
#     beta_L1_scaled = 1.0 / np.sqrt(2)  # ~0.7071
    
#     label_approx = r'Tight Lower Bound (max($L_\infty$, scaled $L_1$))'
#     point_color = 'purple'
#     trail_color = 'orange'
#     patch_color = 'red' # Error patch color

#     # --- 2. Setup the figure and subplots ---
#     fig, (ax_polar, ax_error) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
#     plt.subplots_adjust(hspace=0.3) # Adjust vertical spacing between subplots

#     # --- Polar Plot (ax_polar) ---
#     # True Unit Circle (Radius 1) - This remains static
#     angles_radians_static = np.linspace(0, 2 * np.pi, 361)
#     true_circle_x = np.cos(angles_radians_static)
#     true_circle_y = np.sin(angles_radians_static)
#     ax_polar.plot(true_circle_x, true_circle_y, color='blue', linestyle='-', label='True Magnitude (Radius 1)', linewidth=2)

#     # Set plot limits and labels for polar plot
#     # The combined AMBM value for a unit vector ranges from approx 0.9239 (at 22.5 deg) to 1.0 (at 0/45 deg).
#     limit = 1.1 
#     ax_polar.set_xlim([-limit, limit])
#     ax_polar.set_ylim([-limit, limit])
#     ax_polar.set_aspect('equal', adjustable='box')
#     ax_polar.axhline(0, color='grey', lw=0.5)
#     ax_polar.axvline(0, color='grey', lw=0.5)
#     ax_polar.set_xlabel('Real Part')
#     ax_polar.set_ylabel('Imaginary Part')
#     ax_polar.set_title('Rotating Vector with Dynamic AMBM Contour')
#     ax_polar.grid(True, linestyle='--', alpha=0.6)
    
#     # Initialize legend for polar plot
#     handles, labels = ax_polar.get_legend_handles_labels()
#     ax_polar.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    
#     # 3. Initialize animated elements for polar plot
#     # Rotating vector (quiver)
#     vec_quiver = ax_polar.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green', label='Rotating Unit Vector', width=0.005)
    
#     # Combined AMBM approximation point
#     ambm_point, = ax_polar.plot([], [], 'x', color=point_color, markersize=10, label=f'{label_approx} Point')

#     # Single dynamic contour line for the combined under-approximation (the octagon)
#     ambm_combined_contour_line, = ax_polar.plot([], [], color='green', linestyle=':',
#                                  label=f'{label_approx} Contour', linewidth=2)

#     # Trailing line for the AMBM approximation point
#     ambm_trail_x, ambm_trail_y = [], [] # Lists to store trail coordinates
#     ambm_trail_line, = ax_polar.plot([], [], color=trail_color, linestyle='-', linewidth=1.5, label=f'{label_approx} Trail')

#     # Update legend to include animated elements
#     handles, labels = ax_polar.get_legend_handles_labels()
#     ax_polar.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1))
    
#     # Polygon for error shading
#     ambm_error_patch = Polygon([[0, 0]], closed=True, color=patch_color, alpha=0.2, label=f'{label_approx} Error')
#     ax_polar.add_patch(ambm_error_patch)

#     # --- Error Plot (ax_error) ---
#     error_angles = []
#     relative_error_values = []
#     error_line, = ax_error.plot([], [], color=point_color, label=f'{label_approx} Relative Error') # Line for relative error

#     ax_error.set_xlim([0, 360])
#     # For the tight lower bound, max error is approx 0.0761 (at 22.5 deg). Min error is 0.
#     ax_error.set_ylim([-0.05, 0.1]) # Adjusted to show non-negative error clearly
#     ax_error.set_xlabel('Angle (degrees)')
#     ax_error.set_ylabel('Relative Error (True - Approx) / True')
#     ax_error.set_title('Relative Approximation Error')
#     ax_error.grid(True, linestyle='--', alpha=0.6)
#     ax_error.axhline(0, color='black', linestyle='-', linewidth=0.8) # Add a zero line
#     ax_error.legend(loc='upper right')

#     plt.tight_layout() # Re-adjust layout after adding second subplot

#     # --- 4. Helper function to generate points for the COMBINED AMBM contour (Octagon) ---
#     def generate_combined_ambm_contour(magnitude, num_points_per_segment=50):
#         if magnitude == 0:
#             return [0], [0]

#         # The combined contour is the outer envelope of the L_inf square and the scaled L_1 diamond.
#         # This forms an octagon.
#         # The equation for the combined contour is:
#         # max(max(|x|,|y|), (1/sqrt(2))*(|x|+|y|)) = magnitude

#         # Let tan_22_5 = tan(22.5 degrees) = sqrt(2) - 1
#         tan_22_5 = np.sqrt(2) - 1

#         # Vertices of the octagon in the first quadrant (x >= 0, y >= 0):
#         # V1: (magnitude, 0) - from L_inf
#         # V2: (magnitude, magnitude * tan_22_5) - intersection of x=magnitude and (1/sqrt(2))(x+y)=magnitude
#         # V3: (magnitude * tan_22_5, magnitude) - intersection of y=magnitude and (1/sqrt(2))(x+y)=magnitude
#         # V4: (0, magnitude) - from L_inf

#         v1_x, v1_y = magnitude, 0
#         v2_x, v2_y = magnitude, magnitude * tan_22_5
#         v3_x, v3_y = magnitude * tan_22_5, magnitude
#         v4_x, v4_y = 0, magnitude

#         # Generate points for each segment in Quadrant 1
#         q1_x = []
#         q1_y = []

#         # Segment 1: V1 to V2 (along x=magnitude line)
#         q1_x.extend(np.linspace(v1_x, v2_x, num_points_per_segment, endpoint=False).tolist())
#         q1_y.extend(np.linspace(v1_y, v2_y, num_points_per_segment, endpoint=False).tolist())

#         # Segment 2: V2 to V3 (along (1/sqrt(2))(x+y)=magnitude line)
#         # This segment is part of the scaled L1 diamond.
#         # x + y = magnitude * sqrt(2)
#         # We need to generate points between V2 and V3.
#         x_seg2 = np.linspace(v2_x, v3_x, num_points_per_segment, endpoint=False)
#         y_seg2 = (magnitude * np.sqrt(2) - x_seg2)
#         q1_x.extend(x_seg2.tolist())
#         q1_y.extend(y_seg2.tolist())

#         # Segment 3: V3 to V4 (along y=magnitude line)
#         q1_x.extend(np.linspace(v3_x, v4_x, num_points_per_segment, endpoint=False).tolist())
#         q1_y.extend(np.linspace(v3_y, v4_y, num_points_per_segment, endpoint=False).tolist())
        
#         # Add the last point of Q1 (V4)
#         q1_x.append(v4_x)
#         q1_y.append(v4_y)

#         # Mirror for other quadrants to get the full octagon
#         full_x = []
#         full_y = []

#         # Quadrant 1
#         full_x.extend(q1_x)
#         full_y.extend(q1_y)

#         # Quadrant 2 (mirror Q1 across Y-axis, then reverse order for continuous path)
#         full_x.extend([-x for x in q1_x[::-1]])
#         full_y.extend(q1_y[::-1])

#         # Quadrant 3 (mirror Q1 across origin, then reverse order for continuous path)
#         full_x.extend([-x for x in q1_x])
#         full_y.extend([-y for y in q1_y])

#         # Quadrant 4 (mirror Q1 across X-axis, then reverse order for continuous path)
#         full_x.extend(q1_x[::-1])
#         full_y.extend([-y for y in q1_y[::-1]])

#         # Close the loop (add the first point again)
#         full_x.append(full_x[0])
#         full_y.append(full_y[0])

#         return full_x, full_y

#     # --- 5. Define the update function for the animation ---
#     def update(frame):
#         nonlocal ambm_trail_x, ambm_trail_y, error_angles, relative_error_values

#         angle_degrees = frame
#         angle_radians = np.deg2rad(angle_degrees)

#         # Current unit vector components (cos_theta, sin_theta)
#         cos_theta = np.cos(angle_radians)
#         sin_theta = np.sin(angle_radians)
#         true_norm = 1.0 # For a unit vector

#         # Update rotating vector
#         vec_quiver.set_UVC(cos_theta, sin_theta)

#         # Calculate AMBM approximation for the current unit vector (x,y)
#         abs_current_real = np.abs(cos_theta)
#         abs_current_imag = np.abs(sin_theta)
#         max_val = np.maximum(abs_current_real, abs_current_imag)
#         min_val = np.minimum(abs_current_real, abs_current_imag)
        
#         # Calculate the two components of the tight lower bound for the UNIT vector
#         L_inf_val_unit = alpha_Linf * max_val + beta_Linf * min_val
#         L1_scaled_val_unit = alpha_L1_scaled * max_val + beta_L1_scaled * min_val

#         # The combined AMBM value is the maximum of the two (this is the magnitude of the purple point)
#         combined_ambm_value = np.maximum(L_inf_val_unit, L1_scaled_val_unit)

#         # Calculate the coordinates of the AMBM approximation point
#         ambm_point_x = cos_theta * combined_ambm_value
#         ambm_point_y = sin_theta * combined_ambm_value
        
#         # Update the AMBM approximation point
#         ambm_point.set_data([ambm_point_x], [ambm_point_y])

#         # --- Update the Trailing Line ---
#         ambm_trail_x.append(ambm_point_x)
#         ambm_trail_y.append(ambm_point_y)
#         ambm_trail_line.set_data(ambm_trail_x, ambm_trail_y)
#         # --- End of Trailing Line Update ---

#         # --- Calculate and Scale the Combined Octagon Contour ---
#         # The contour should be scaled by the combined_ambm_value
#         dynamic_contour_magnitude = combined_ambm_value
#         full_x_combined, full_y_combined = generate_combined_ambm_contour(dynamic_contour_magnitude)
        
#         x_rotated_combined = [cos_theta * x_b - sin_theta * y_b for x_b, y_b in zip(full_x_combined, full_y_combined)]
#         y_rotated_combined = [sin_theta * x_b + cos_theta * y_b for x_b, y_b in zip(full_x_combined, full_y_combined)]
#         ambm_combined_contour_line.set_data(x_rotated_combined, y_rotated_combined)   
    
#         # --- End of Dynamic AMBM Contour ---
        
#         # --- Update Error Patch ---
#         # For under-approximation, the error patch should be between the approximation point
#         # and the true unit circle point, connected to the origin.
#         if combined_ambm_value != 0:
#             # Origin -> Approx Point -> True Point -> Origin (for patch inside circle)
#             fill_x = [0, ambm_point_x, cos_theta, 0] 
#             fill_y = [0, ambm_point_y, sin_theta, 0]
#             ambm_error_patch.set_xy(list(zip(fill_x, fill_y)))
#         else:
#             ambm_error_patch.set_xy([[0,0],[0,0],[0,0],[0,0]]) # Hide if point is at origin


#         # --- Update Error Subplot ---
#         error_angles.append(angle_degrees)
        
#         # Relative error: (True - Approx) / True. This will be non-negative for under-approximation.
#         relative_error = (true_norm - combined_ambm_value) / true_norm if true_norm != 0 else 0
#         relative_error_values.append(relative_error)

#         error_line.set_data(error_angles, relative_error_values)

#         ax_polar.set_title(f'Rotating Vector with Dynamic AMBM Contour (Angle: {angle_degrees}°)')

#         # Return all updated artists
#         return vec_quiver, ambm_point, ambm_combined_contour_line, \
#                ambm_trail_line, ambm_error_patch, error_line, ax_polar.title


#     # 6. Create the animation
#     ani = animation.FuncAnimation(fig, update, frames=range(361), interval=50, blit=False)

#     plt.show()

# # Run the animation
# plot_animated_ambm_approximation()















# def plot_animated_ambm_approximation_extended():
#     """
#     Creates an animated plot visualizing a rotating unit vector, its true magnitude,
#     and two different Alpha Max Beta Min (AMBM) approximations.
#     It shows:
#     - The true unit circle.
#     - A rotating unit vector.
#     - For each approximation:
#         - Its approximation point (scaled vector).
#         - Its dynamically scaled and rotated AMBM contour.
#         - A trailing line of its approximation point.
#         - Shaded area representing the error.
#     - A subplot showing the relative error of both approximations vs. angle.
#     - Interactive checkboxes to toggle visibility of each approximation.
#     """

#     # 1. Define AMBM approximation parameters for two different approximations
    
#     # Approximation 1: Optimal Closest Approximation (not guaranteed lower bound)
#     alpha1 = 0.9604
#     beta1 = 0.3978
#     label1 = r'Opt. Approx ($\alpha=0.9604, \beta=0.3978$)'
#     color1 = 'purple'
#     contour_color1 = 'red'
#     trail_color1 = 'orange'
#     patch_color1 = 'red'

#     # Approximation 2: Tightest Guaranteed Under-Approximation (composite)
#     # This is effectively max(L_inf_norm, scaled_L1_norm)
#     # We will calculate its value directly in the update function.
#     # For plotting its *dynamic contour*, we'll use effective alpha/beta based on which
#     # component is chosen by the max() function for the current angle.
#     alpha2_Linf = 1.0
#     beta2_Linf = 0.0
#     alpha2_L1_scaled = 1.0 / np.sqrt(2) # ~0.7071
#     beta2_L1_scaled = 1.0 / np.sqrt(2)  # ~0.7071
#     label2 = r'Tight Lower Bound (max($L_\infty$, scaled $L_1$))'
#     color2 = 'cyan'
#     contour_color2 = 'green'
#     trail_color2 = 'blue'
#     patch_color2 = 'cyan'

#     # 2. Setup the figure and subplots
#     fig, (ax_polar, ax_error) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
#     plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95, hspace=0.3) # Adjust layout for widgets

#     # --- Polar Plot (ax_polar) ---
#     # True Unit Circle (Radius 1) - This remains static
#     angles_radians_static = np.linspace(0, 2 * np.pi, 361)
#     true_circle_x = np.cos(angles_radians_static)
#     true_circle_y = np.sin(angles_radians_static)
#     ax_polar.plot(true_circle_x, true_circle_y, color='blue', linestyle='-', label='True Magnitude (Radius 1)', linewidth=2)

#     # Set plot limits and labels
#     limit = 1.1 
#     ax_polar.set_xlim([-limit, limit])
#     ax_polar.set_ylim([-limit, limit])
#     ax_polar.set_aspect('equal', adjustable='box')
#     ax_polar.axhline(0, color='grey', lw=0.5)
#     ax_polar.axvline(0, color='grey', lw=0.5)
#     ax_polar.set_xlabel('Real Part')
#     ax_polar.set_ylabel('Imaginary Part')
#     ax_polar.set_title('Rotating Vector with Dynamic AMBM Contours')
#     ax_polar.grid(True, linestyle='--', alpha=0.6)
    
#     # --- Initialize animated elements for Approx 1 ---
#     vec_quiver = ax_polar.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green', label='Rotating Unit Vector', width=0.005)
    
#     ambm_point1, = ax_polar.plot([], [], 'x', color=color1, markersize=10, label=f'{label1} Point')
#     ambm_contour_line1, = ax_polar.plot([], [], color=contour_color1, linestyle='--',
#                                          label=f'{label1} Contour', linewidth=2)
#     ambm_trail_x1, ambm_trail_y1 = [], []
#     ambm_trail_line1, = ax_polar.plot([], [], color=trail_color1, linestyle='-', linewidth=1.5, label=f'{label1} Trail')
#     ambm_error_patch1 = Polygon([[0, 0]], closed=True, color=patch_color1, alpha=0.2, label=f'{label1} Error')
#     ax_polar.add_patch(ambm_error_patch1)

#     # --- Initialize animated elements for Approx 2 ---
#     ambm_point2, = ax_polar.plot([], [], 'o', color=color2, markersize=8, label=f'{label2} Point')
#     ambm_contour_line2, = ax_polar.plot([], [], color=contour_color2, linestyle=':',
#                                          label=f'{label2} Contour', linewidth=2)
#     ambm_trail_x2, ambm_trail_y2 = [], []
#     ambm_trail_line2, = ax_polar.plot([], [], color=trail_color2, linestyle='-', linewidth=1.5, label=f'{label2} Trail')
#     ambm_error_patch2 = Polygon([[0, 0]], closed=True, color=patch_color2, alpha=0.2, label=f'{label2} Error')
#     ax_polar.add_patch(ambm_error_patch2)

#     # Update legend for polar plot (initial state)
#     handles_polar, labels_polar = ax_polar.get_legend_handles_labels()
#     ax_polar.legend(handles_polar, labels_polar, loc='upper left', bbox_to_anchor=(1,1))
    
#     # --- Error Plot (ax_error) ---
#     error_angles = []
#     error_values1 = []
#     error_values2 = []

#     error_line1, = ax_error.plot([], [], color=color1, label=f'{label1} Relative Error')
#     error_line2, = ax_error.plot([], [], color=color2, label=f'{label2} Relative Error')

#     ax_error.set_xlim([0, 360])
#     ax_error.set_ylim([-0.35, 0.35]) # Adjust limits based on expected error range (e.g., +/- 30%)
#     ax_error.set_xlabel('Angle (degrees)')
#     ax_error.set_ylabel('Relative Error (True - Approx) / True')
#     ax_error.set_title('Relative Approximation Error')
#     ax_error.grid(True, linestyle='--', alpha=0.6)
#     ax_error.legend(loc='upper right')

#     plt.tight_layout()

#     # --- Checkbox Widgets for Toggling Visibility ---
#     # Create an axes for the checkboxes
#     ax_checkbox = fig.add_axes([0.01, 0.8, 0.1, 0.15]) # [left, bottom, width, height]
#     ax_checkbox.set_axis_off() # Hide the axes frame

#     check_labels = [label1, label2]
#     initial_visibility = [True, True] # Both visible initially

#     check = CheckButtons(ax_checkbox, check_labels, initial_visibility)

#     # Store artists for easy toggling
#     artists_approx1 = [ambm_point1, ambm_contour_line1, ambm_trail_line1, ambm_error_patch1, error_line1]
#     artists_approx2 = [ambm_point2, ambm_contour_line2, ambm_trail_line2, ambm_error_patch2, error_line2]

#     def toggle_visibility(label):
#         if label == label1:
#             for artist in artists_approx1:
#                 artist.set_visible(not artist.get_visible())
#         elif label == label2:
#             for artist in artists_approx2:
#                 artist.set_visible(not artist.get_visible())
#         fig.canvas.draw_idle() # Redraw the canvas to reflect changes

#     check.on_clicked(toggle_visibility)


#     # 4. Define the update function for the animation
#     def update(frame):
#         nonlocal ambm_trail_x1, ambm_trail_y1, ambm_trail_x2, ambm_trail_y2, \
#                    error_angles, error_values1, error_values2

#         angle_degrees = frame
#         angle_radians = np.deg2rad(angle_degrees)

#         # Current unit vector components (cos_theta, sin_theta)
#         cos_theta = np.cos(angle_radians)
#         sin_theta = np.sin(angle_radians)
#         true_norm = 1.0 # For a unit vector

#         # Update rotating vector
#         vec_quiver.set_UVC(cos_theta, sin_theta)

#         # --- Approximation 1 Calculations (Optimal Closest AMBM) ---
#         abs_real1 = np.abs(cos_theta)
#         abs_imag1 = np.abs(sin_theta)
#         max_val1 = np.maximum(abs_real1, abs_imag1)
#         min_val1 = np.minimum(abs_real1, abs_imag1)
        
#         ambm_value1 = alpha1 * max_val1 + beta1 * min_val1
        
#         ambm_point_x1 = cos_theta * ambm_value1
#         ambm_point_y1 = sin_theta * ambm_value1
        
#         ambm_point1.set_data([ambm_point_x1], [ambm_point_y1])

#         ambm_trail_x1.append(ambm_point_x1)
#         ambm_trail_y1.append(ambm_point_y1)
#         ambm_trail_line1.set_data(ambm_trail_x1, ambm_trail_y1)

#         # Dynamic contour for Approx 1
#         dynamic_contour_magnitude1 = alpha1 * ambm_value1 # Correct scaling for rotated diamond
#         x_at_45_deg1 = dynamic_contour_magnitude1 / (alpha1 + beta1) if (alpha1 + beta1) != 0 else 0
#         x_on_axis1 = dynamic_contour_magnitude1 / alpha1 if alpha1 != 0 else 0

#         base_ambm_x1 = []
#         base_ambm_y1 = []
#         num_points_per_segment = 50
#         for x_val in np.linspace(x_on_axis1, x_at_45_deg1, num=num_points_per_segment):
#             base_ambm_x1.append(x_val)
#             base_ambm_y1.append((dynamic_contour_magnitude1 - alpha1 * x_val) / beta1 if beta1 != 0 else 0)
#         for y_val in np.linspace(x_at_45_deg1, x_on_axis1, num=num_points_per_segment):
#             base_ambm_x1.append((dynamic_contour_magnitude1 - alpha1 * y_val) / beta1 if beta1 != 0 else 0)
#             base_ambm_y1.append(y_val)
        
#         # Mirror for other quadrants
#         full_x1 = base_ambm_x1 + [-x for x in base_ambm_x1[::-1]] + [-x for x in base_ambm_x1] + base_ambm_x1[::-1]
#         full_y1 = base_ambm_y1 + base_ambm_y1[::-1] + [-y for y in base_ambm_y1] + [-y for y in base_ambm_y1[::-1]]
        
#         x_rotated1 = [cos_theta * x_b - sin_theta * y_b for x_b, y_b in zip(full_x1, full_y1)]
#         y_rotated1 = [sin_theta * x_b + cos_theta * y_b for x_b, y_b in zip(full_x1, full_y1)]
#         ambm_contour_line1.set_data(x_rotated1, y_rotated1)

#         # --- Approximation 2 Calculations (Tightest Guaranteed Under-Approximation) ---
#         abs_real2 = np.abs(cos_theta)
#         abs_imag2 = np.abs(sin_theta)
#         max_val2 = np.maximum(abs_real2, abs_imag2)
#         min_val2 = np.minimum(abs_real2, abs_imag2)

#         # Calculate the two components
#         L_inf_val = alpha2_Linf * max_val2 + beta2_Linf * min_val2 # M
#         L1_scaled_val = alpha2_L1_scaled * max_val2 + beta2_L1_scaled * min_val2 # (1/sqrt(2))*(M+m)

#         # The combined AMBM value is the maximum of the two
#         combined_ambm_value = np.maximum(L_inf_val, L1_scaled_val)

#         ambm_point_x2 = cos_theta * combined_ambm_value
#         ambm_point_y2 = sin_theta * combined_ambm_value
        
#         ambm_point2.set_data([ambm_point_x2], [ambm_point_y2])

#         ambm_trail_x2.append(ambm_point_x2)
#         ambm_trail_y2.append(ambm_point_y2)
#         ambm_trail_line2.set_data(ambm_trail_x2, ambm_trail_y2)

#         # Dynamic contour for Approx 2 (using effective alpha/beta for current angle)
#         # This is a simplification; the true combined contour is more complex.
#         # We determine which component is active for the current angle's AMBM value
#         if L_inf_val >= L1_scaled_val: # L_inf is dominant or equal
#             current_alpha2 = alpha2_Linf
#             current_beta2 = beta2_Linf
#         else: # Scaled L1 is dominant
#             current_alpha2 = alpha2_L1_scaled
#             current_beta2 = beta2_L1_scaled
        
#         dynamic_contour_magnitude2 = current_alpha2 * combined_ambm_value # Correct scaling
        
#         x_at_45_deg2 = dynamic_contour_magnitude2 / (current_alpha2 + current_beta2) if (current_alpha2 + current_beta2) != 0 else 0
#         x_on_axis2 = dynamic_contour_magnitude2 / current_alpha2 if current_alpha2 != 0 else 0

#         base_ambm_x2 = []
#         base_ambm_y2 = []
#         for x_val in np.linspace(x_on_axis2, x_at_45_deg2, num=num_points_per_segment):
#             base_ambm_x2.append(x_val)
#             base_ambm_y2.append((dynamic_contour_magnitude2 - current_alpha2 * x_val) / current_beta2 if current_beta2 != 0 else 0)
#         for y_val in np.linspace(x_at_45_deg2, x_on_axis2, num=num_points_per_segment):
#             base_ambm_x2.append((dynamic_contour_magnitude2 - current_alpha2 * y_val) / current_beta2 if current_beta2 != 0 else 0)
#             base_ambm_y2.append(y_val)
        
#         # Mirror for other quadrants
#         full_x2 = base_ambm_x2 + [-x for x in base_ambm_x2[::-1]] + [-x for x in base_ambm_x2] + base_ambm_x2[::-1]
#         full_y2 = base_ambm_y2 + base_ambm_y2[::-1] + [-y for y in base_ambm_y2] + [-y for y in base_ambm_y2[::-1]]
        
#         x_rotated2 = [cos_theta * x_b - sin_theta * y_b for x_b, y_b in zip(full_x2, full_y2)]
#         y_rotated2 = [sin_theta * x_b + cos_theta * y_b for x_b, y_b in zip(full_x2, full_y2)]
#         ambm_contour_line2.set_data(x_rotated2, y_rotated2)


#         # --- Update Error Patches ---
#         # Error patch 1
#         current_unit_x = cos_theta
#         current_unit_y = sin_theta
#         if ambm_value1 != 0:
#             fill_x1 = [ambm_point_x1, current_unit_x, 0, ambm_point_x1] # Approx point, true unit point, origin, approx point
#             fill_y1 = [ambm_point_y1, current_unit_y, 0, ambm_point_y1]
#             ambm_error_patch1.set_xy(list(zip(fill_x1, fill_y1)))
#         else:
#             ambm_error_patch1.set_xy([[0,0],[0,0],[0,0],[0,0]]) # Hide if point is at origin

#         # Error patch 2
#         if combined_ambm_value != 0:
#             fill_x2 = [ambm_point_x2, current_unit_x, 0, ambm_point_x2]
#             fill_y2 = [ambm_point_y2, current_unit_y, 0, ambm_point_y2]
#             ambm_error_patch2.set_xy(list(zip(fill_x2, fill_y2)))
#         else:
#             ambm_error_patch2.set_xy([[0,0],[0,0],[0,0],[0,0]]) # Hide if point is at origin


#         # --- Update Error Subplot ---
#         error_angles.append(angle_degrees)
        
#         relative_error1 = (true_norm - ambm_value1) / true_norm if true_norm != 0 else 0
#         relative_error2 = (true_norm - combined_ambm_value) / true_norm if true_norm != 0 else 0
        
#         error_values1.append(relative_error1)
#         error_values2.append(relative_error2)

#         error_line1.set_data(error_angles, error_values1)
#         error_line2.set_data(error_angles, error_values2)

#         ax_polar.set_title(f'Rotating Vector with Dynamic AMBM Contours (Angle: {angle_degrees}°)')

#         # Return all updated artists
#         # Only return artists that are currently visible to optimize blitting if it were enabled
#         # For blit=False, returning all is fine.
#         return vec_quiver, ambm_point1, ambm_contour_line1, ambm_trail_line1, ambm_error_patch1, \
#                ambm_point2, ambm_contour_line2, ambm_trail_line2, ambm_error_patch2, \
#                error_line1, error_line2, ax_polar.title


#     # 5. Create the animation
#     # blit=False is crucial for widgets and patches to update correctly in some environments
#     ani = animation.FuncAnimation(fig, update, frames=range(361), interval=50, blit=False)

#     plt.show()

# # Run the animation
# plot_animated_ambm_approximation_extended()