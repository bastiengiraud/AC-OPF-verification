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
    #"model_verification_results_14_buses.xlsx",
    "rahul_verification_results_57_buses.xlsx",
    "rahul_verification_results_118_buses.xlsx",
    "rahul_verification_results_300_buses.xlsx",
    "rahul_verification_results_793_buses.xlsx",
]

# Deltas must be consistent across files
deltas = [0.0, 0.05, 0.1, 0.15, 0.20]  

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


# # --- Main plotting function ---
# def plot_group(excel_files, group_name, save_name, model_filter):
#     mpl.rcParams.update({
#         'font.size': 14,        # tick labels, axis labels
#         'axes.titlesize': 14,   # subplot titles
#         'axes.labelsize': 14,   # axis labels
#         'legend.fontsize': 12,  # legend
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12
#     })
#     fig, axes = plt.subplots(len(excel_files), 1, figsize=(6.5, 6))
    
#     metric_labels = {
#         "Pg tot Max Violation": r"$\nu_{P_g}^{\mathrm{max}}$",
#         "Pg tot Avg Violation": r"$\nu_{P_g}^{\mathrm{avg}}$",
#         "Qg tot Max Violation": r"$\nu_{Q_g}^{\mathrm{max}}$",
#         "Qg tot Avg Violation": r"$\nu_{Q_g}^{\mathrm{avg}}$",
#         "Vmg tot Max Violation": r"$\nu_{V_m,g}^{\mathrm{max}}$",
#         "Vmg tot Avg Violation": r"$\nu_{V_m,g}^{\mathrm{avg}}$",
#         "Ibr tot Max Violation": r"$\nu_{l}^{\mathrm{max}}$",
#         "Ibal tot Max Violation": r"$\nu_{bal}^{\mathrm{max}}$",
#     }
    
#     metric_colors = {
#         "Pg tot Max Violation": "tab:blue",
#         "Qg tot Max Violation": "tab:orange",
#         "Vmg tot Max Violation": "tab:green",
#         "Ibr tot Max Violation": "tab:red",
#         "Ibal tot Max Violation": "tab:purple",
#         #"Qg tot Max Violation": "tab:brown",
#         # Add more if needed
#     }


#     for i, file in enumerate(excel_files):
#         filepath = os.path.join(script_dir, file)
#         df = load_results(filepath)
#         df.index = df.index.astype(str).str.strip()

#         # Get only "Max Violation" metrics
#         metrics_to_plot = [
#             m for m in df.index 
#             if "tot Max" in m and m != "Vm tot Max Violation"
#         ]


#         # Parse model names + deltas
#         parsed = [split_col(c) for c in df.columns]
#         model_names = sorted(set(m for m, d in parsed if d is not None))
#         deltas = sorted(set(d for m, d in parsed if d is not None))

#         # Filter models by user-defined rule
#         model_names = [m for m in model_names if model_filter(m)]

#         # Extract system size (bus count) from filename
#         system_size = (
#             os.path.basename(file)
#             .replace("rahul_verification_results_", "")
#             .replace("_buses.xlsx", "")
#         )

#         ax = axes[i]
#         for model in model_names:
#             model_data = df[[col for col in df.columns if col.startswith(model)]]
#             model_data.columns = deltas            

#             for metric in metrics_to_plot:
#                 if metric in model_data.index:
#                     series = model_data.loc[metric, :]

#                     # Skip plotting if all values are NaN or 0
#                     color = metric_colors.get(metric, None)  # consistent metric color

#                     if series.isna().all():
#                         continue

#                     if (series == 0).all():
#                         # plot a flat line at 0 in the metric’s color
#                         ax.plot(
#                             deltas, [0] * len(deltas),
#                             marker='o', markersize=6,
#                             color=color, linewidth=2,
#                             label=metric_labels.get(metric, metric)
#                         )
#                     else:
#                         ax.plot(
#                             deltas, series,
#                             marker='o', markersize=6, linewidth=2,
#                             color=color,
#                             label=metric_labels.get(metric, metric)
#                         )


#         ax.set_title(f"{system_size}-bus system", fontsize=12, fontweight="bold")
#         #ax.set_ylabel(r'Guarantee $\nu$ (%)')
#         ax.grid(True, linestyle="--", alpha=0.6)
#         ax.set_xticks(deltas) 
#         # ax.legend(fontsize=12, loc="upper right", ncol=5, frameon=True)
        
#         # Only add horizontal legend to the first subplot
#         if i == 2:
#             ax.legend(
#                 fontsize=12,
#                 loc='upper right',
#                 # bbox_to_anchor=(0.5, 1.15),
#                 ncol=5,
#                 frameon=True
#             )

#     axes[-1].set_xlabel(r'$\delta$ Factor')
#     fig.text(0.0, 0.5, r'Guarantee $\nu$ (%)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
#     # plt.suptitle(group_name, fontsize=14, fontweight="bold")  # group title on top
#     plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
#     outpath = os.path.join(script_dir, save_name)
#     plt.savefig(outpath, dpi=300)
#     plt.close()
#     print(f"✅ Saved {group_name} plot to {outpath}")

def plot_group(excel_files, group_name, save_name, model_filter):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import os

    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    fig, axes = plt.subplots(len(excel_files), 1, figsize=(7, 4.5),
                             sharex=True)

    metric_labels = {
        "Pg tot Max Violation": r"$\nu_{P_g}^{\mathrm{max}}$",
        "Pg tot Avg Violation": r"$\nu_{P_g}^{\mathrm{avg}}$",
        "Qg tot Max Violation": r"$\nu_{Q_g}^{\mathrm{max}}$",
        "Qg tot Avg Violation": r"$\nu_{Q_g}^{\mathrm{avg}}$",
        "Vmg tot Max Violation": r"$\nu_{V_{m,g}}^{\mathrm{max}}$",
        "Vmg tot Avg Violation": r"$\nu_{V_{m,g}}^{\mathrm{avg}}$",
        "Ibr tot Max Violation": r"$\nu_{l}^{\mathrm{max}}$",
        "Ibal tot Max Violation": r"$\nu_{bal}^{\mathrm{max}}$",
    }

    metric_colors = {
        "Pg tot Max Violation": "tab:blue",
        "Qg tot Max Violation": "tab:orange",
        "Vmg tot Max Violation": "tab:green",
        "Ibr tot Max Violation": "tab:red",
        "Ibal tot Max Violation": "tab:purple",
    }

    legend_handles = []

    for i, file in enumerate(excel_files):
        filepath = os.path.join(script_dir, file)
        df = load_results(filepath)
        df.index = df.index.astype(str).str.strip()

        metrics_to_plot = [
            m for m in df.index if "tot Max" in m and m != "Vm tot Max Violation"
        ]

        parsed = [split_col(c) for c in df.columns]
        model_names = sorted(set(m for m, d in parsed if d is not None))
        deltas = sorted(set(d for m, d in parsed if d is not None))
        model_names = [m for m in model_names if model_filter(m)]

        system_size = (
            os.path.basename(file)
            .replace("rahul_verification_results_", "")
            .replace("_buses.xlsx", "")
        )

        ax = axes[i]
        for model in model_names:
            model_data = df[[col for col in df.columns if col.startswith(model)]]
            model_data.columns = deltas

            for metric in metrics_to_plot:
                if metric in model_data.index:
                    series = model_data.loc[metric, :]
                    color = metric_colors.get(metric, None)
                    label = metric_labels.get(metric, metric)

                    if series.isna().all():
                        continue

                    (line,) = ax.plot(
                        deltas,
                        series if not (series == 0).all() else [0] * len(deltas),
                        marker="x",
                        markersize=5,
                        linewidth=2,
                        color=color,
                        label=label,
                    )
                    if label not in [l.get_label() for l in legend_handles]:
                        legend_handles.append(line)

        ax.text(
            0.95,
            0.85,
            f"{system_size}-bus",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            fontweight="bold",
            # bbox=dict(
            # facecolor='lightgray',     # background color of the box
            # edgecolor='black',     # box border color
            # boxstyle='round,pad=0.3',  # rounded box with padding
            # alpha=0.5,              # transparency (1 = opaque)
            # pad=0.2
            # ),
        )

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(deltas)

    axes[-1].set_xlabel(r'$\delta$ Factor')
    fig.text(0.0, 0.5, r'Guarantee $\nu$ (%)', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    # --- Legend in bottom-left corner of top subplot ---
    axes[0].legend(
        handles=legend_handles,
        loc='lower left',
        ncol=5,                        # number of columns
        frameon=True,
        bbox_to_anchor=(0.0, -0.02),   # position below axis
        labelspacing=0.1,               # vertical spacing between entries
        columnspacing=0.6,              # horizontal spacing between columns
        handlelength=0.8,               # length of the line markers
        handleheight=0.3,               # height of the line markers
        markerscale=0.6                  # scale of marker relative to line
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.2)
    outpath = os.path.join(script_dir, save_name)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
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