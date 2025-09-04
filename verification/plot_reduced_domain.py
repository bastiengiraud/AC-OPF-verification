import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.style.use(['/mnt/c/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/0) DTU Admin/5) Templates/thesis.mplstyle'])
plt.rcParams['text.usetex'] = False

from matplotlib import font_manager

font_manager.fontManager.addfont('/mnt/c/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/0) DTU Admin/5) Templates/palr45w.ttf')
plt.rcParams['font.family'] = 'Palatino' # Set the font globally
#plt.rcParams['font.family'] = 'sans-serif'

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_files = [
    "model_verification_results_14_buses.xlsx",
    "model_verification_results_57_buses.xlsx",
    "model_verification_results_118_buses.xlsx",
]

# Deltas must be consistent across files
deltas = [0.8, 0.9, 1.0, 1.1, 1.2]  # adapt if needed

# --- Helper: Load results ---
def load_results(filepath):
    df = pd.read_excel(filepath, index_col=0)
    return df

import re

def split_col(col):
    match = re.match(r"(.+\.pt)_(\d+\.\d+)", col)
    if match:
        return match.group(1), float(match.group(2))
    else:
        return col, None


# --- Main plotting function ---
def plot_group(excel_files, group_name, save_name, model_filter):
    mpl.rcParams.update({
        'font.size': 14,        # tick labels, axis labels
        'axes.titlesize': 14,   # subplot titles
        'axes.labelsize': 14,   # axis labels
        'legend.fontsize': 12,  # legend
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    fig, axes = plt.subplots(len(excel_files), 1, figsize=(6.5, 5))
    
    metric_labels = {
        "Pg tot Max Violation": r"$\nu_{P_g}^{\mathrm{max}}$",
        "Pg tot Avg Violation": r"$\nu_{P_g}^{\mathrm{avg}}$",
        "Qg tot Max Violation": r"$\nu_{Q_g}^{\mathrm{max}}$",
        "Qg tot Avg Violation": r"$\nu_{Q_g}^{\mathrm{avg}}$",
        "Vm tot Max Violation": r"$\nu_{V_m}^{\mathrm{max}}$",
        "Vm tot Avg Violation": r"$\nu_{V_m}^{\mathrm{avg}}$",
        "Ibr tot Max Violation": r"$\nu_{l}^{\mathrm{max}}$",
        "Ibal tot Max Violation": r"$\nu_{bal}^{\mathrm{max}}$",
    }


    for i, file in enumerate(excel_files):
        filepath = os.path.join(script_dir, file)
        df = load_results(filepath)
        df.index = df.index.astype(str).str.strip()

        # Get only "Max Violation" metrics
        metrics_to_plot = [m for m in df.index if "tot Max" in m]

        # Parse model names + deltas
        parsed = [split_col(c) for c in df.columns]
        model_names = sorted(set(m for m, d in parsed if d is not None))
        deltas = sorted(set(d for m, d in parsed if d is not None))

        # Filter models by user-defined rule
        model_names = [m for m in model_names if model_filter(m)]

        # Extract system size (bus count) from filename
        system_size = (
            os.path.basename(file)
            .replace("model_verification_results_", "")
            .replace("_buses.xlsx", "")
        )

        ax = axes[i]
        for model in model_names:
            model_data = df[[col for col in df.columns if col.startswith(model)]]
            model_data.columns = deltas

            for metric in metrics_to_plot:
                if metric in model_data.index:
                    series = model_data.loc[metric, :]
                    # Skip plotting if all values are NaN or 0
                    if series.isna().all() or (series == 0).all():
                        continue
                    ax.plot(deltas, series, marker='o', markersize=6, linewidth=2, label=metric_labels.get(metric, metric))


        ax.set_title(f"{system_size}-bus system", fontsize=12, fontweight="bold")
        ax.set_ylabel(r'Guarantee $\nu$ (%)')
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(deltas) 
        # ax.legend(fontsize=12, loc="upper right", ncol=5, frameon=True)
        
        # Only add horizontal legend to the first subplot
        if i == 2:
            ax.legend(
                fontsize=12,
                loc='upper right',
                # bbox_to_anchor=(0.5, 1.15),
                ncol=5,
                frameon=True
            )

    axes[-1].set_xlabel(r'$\delta$ Factor')
    # plt.suptitle(group_name, fontsize=14, fontweight="bold")  # group title on top
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
    outpath = os.path.join(script_dir, save_name)
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"✅ Saved {group_name} plot to {outpath}")


# --- Generate the two figures ---
plot_group(
    excel_files,
    group_name="Power (pg_vm_True)",
    save_name="plots/pg_vm_true_results.png",
    model_filter=lambda m: "True_pg_vm" in m
)

plot_group(
    excel_files,
    group_name="Voltage (vr_vi_True)",
    save_name="plots/vr_vi_true_results.png",
    model_filter=lambda m: "True_vr_vi" in m
)