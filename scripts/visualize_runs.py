import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import numpy as np

########################################
# Configuration
########################################
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "Faber_UR_NLP_disaster_tweets"

RUN_IDS = [
    "1a0b86fc-f640-4731-a369-a914e77cb2aa",
    "34e209e0-9400-412c-9888-67cbaa5373b4"
]

metrics_to_plot = ["accuracy", "f1", "precision", "recall", "loss"]

########################################
# Connect to the database
########################################
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)
cursor = conn.cursor(dictionary=True)

########################################
# Fetch data
########################################
if not RUN_IDS:
    cursor.execute("SELECT run_id FROM runs ORDER BY start_time DESC LIMIT 10;")
    RUN_IDS = [row["run_id"] for row in cursor.fetchall()]

print("Comparing runs:")
for r in RUN_IDS:
    print(f"  {r}")

all_hparams = {}
all_data = {}

for run_id in RUN_IDS:
    # Hyperparameters
    cursor.execute("""
        SELECT param_name, param_value
        FROM hyperparameters
        WHERE run_id = %s
    """, (run_id,))
    hparams = cursor.fetchall()
    hparam_dict = {}
    for hp in hparams:
        val = hp['param_value']
        try:
            val = json.loads(val)
        except:
            pass
        hparam_dict[hp['param_name']] = val
    all_hparams[run_id] = hparam_dict

    # Metrics
    cursor.execute("""
        SELECT step, metric_name, metric_value, phase
        FROM metrics
        WHERE run_id = %s
        ORDER BY step ASC
    """, (run_id,))
    mrows = cursor.fetchall()
    df = pd.DataFrame(mrows)
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors='coerce')
    all_data[run_id] = df

conn.close()

########################################
# Identify differing hyperparameters
########################################
all_param_names = set()
for h in all_hparams.values():
    all_param_names.update(h.keys())

differing_params = []
for param in all_param_names:
    values = [all_hparams[r].get(param, None) for r in RUN_IDS]
    # Convert to JSON strings for comparison
    try:
        json_values = [json.dumps(v, sort_keys=True) for v in values]
    except:
        json_values = [str(v) for v in values]
    if len(set(json_values)) > 1:
        differing_params.append(param)

print("Differing hyperparameters among runs:", differing_params)

def format_param_value(val):
    if isinstance(val, list):
        return f"list={len(val)}"
    elif isinstance(val, (float, int)):
        return str(val)
    else:
        return str(val)

run_labels = {}
for run_id in RUN_IDS:
    label_parts = []
    for p in differing_params:
        val = all_hparams[run_id].get(p, None)
        label_parts.append(f"{p}={format_param_value(val)}")
    run_label = ", ".join(label_parts) if label_parts else run_id
    run_labels[run_id] = run_label

for rid, lbl in run_labels.items():
    print(f"Run: {rid} Label: {lbl}")

########################################
# Prepare directory for plots
########################################
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("plots", "runs", timestamp)
os.makedirs(out_dir, exist_ok=True)

########################################
# First Figure: Line plots over steps
########################################
fig, axes = plt.subplots(len(metrics_to_plot), 2, figsize=(12, 4*len(metrics_to_plot)), sharex=False)
if len(metrics_to_plot) == 1:
    axes = np.array([axes])  # Ensure axes is 2D

# Determine unified y-limits for the line plots
metric_phase_stats = {}
for metric in metrics_to_plot:
    metric_phase_stats[metric] = {"train": {"min": float('inf'), "max": float('-inf')},
                                  "eval": {"min": float('inf'), "max": float('-inf')}}
    for run_id, df in all_data.items():
        for phase in ["train", "eval"]:
            mdf = df[(df["metric_name"] == metric) & (df["phase"] == phase)]
            if not mdf.empty:
                cur_min = mdf["metric_value"].min()
                cur_max = mdf["metric_value"].max()
                if cur_min < metric_phase_stats[metric][phase]["min"]:
                    metric_phase_stats[metric][phase]["min"] = cur_min
                if cur_max > metric_phase_stats[metric][phase]["max"]:
                    metric_phase_stats[metric][phase]["max"] = cur_max

for metric in metrics_to_plot:
    global_min = min(metric_phase_stats[metric]["train"]["min"], metric_phase_stats[metric]["eval"]["min"])
    global_max = max(metric_phase_stats[metric]["train"]["max"], metric_phase_stats[metric]["eval"]["max"])
    margin = (global_max - global_min) * 0.05
    if margin == 0:
        margin = 0.01
    global_min -= margin
    global_max += margin
    metric_phase_stats[metric]["unified_min"] = global_min
    metric_phase_stats[metric]["unified_max"] = global_max

for i, metric in enumerate(metrics_to_plot):
    for j, phase in enumerate(["train", "eval"]):
        ax = axes[i, j]
        for run_id, df in all_data.items():
            mdf = df[(df["metric_name"] == metric) & (df["phase"] == phase)].dropna(subset=["metric_value"])
            if not mdf.empty:
                ax.plot(mdf["step"], mdf["metric_value"], label=run_labels[run_id])
        ax.set_title(f"{phase.capitalize()} {metric}")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_ylim(metric_phase_stats[metric]["unified_min"], metric_phase_stats[metric]["unified_max"])
        if i == 0 and j == 1:
            ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "line_plots.png"))
plt.close()

########################################
# Second Figure: Bar plots for best values
########################################
fig, axes = plt.subplots(len(metrics_to_plot), 2, figsize=(12, 4*len(metrics_to_plot)), sharex=False)
if len(metrics_to_plot) == 1:
    axes = np.array([axes])  # Ensure axes is 2D

# Compute best values
best_values = {}
for run_id, df in all_data.items():
    best_values[run_id] = {}
    for metric in metrics_to_plot:
        for phase in ["train", "eval"]:
            mdf = df[(df["metric_name"] == metric) & (df["phase"] == phase)]
            if mdf.empty:
                best_values[run_id][(metric, phase)] = None
            else:
                if metric == "loss":
                    val = mdf["metric_value"].min()
                else:
                    val = mdf["metric_value"].max()
                best_values[run_id][(metric, phase)] = val

# Determine y-limits for bar plots
for metric in metrics_to_plot:
    values_train = [best_values[r][(metric,"train")] for r in RUN_IDS if best_values[r][(metric,"train")] is not None]
    values_eval = [best_values[r][(metric,"eval")] for r in RUN_IDS if best_values[r][(metric,"eval")] is not None]
    all_vals = values_train + values_eval
    if len(all_vals) == 0:
        all_vals = [0, 1]

    min_val = min(all_vals)
    max_val = max(all_vals)
    margin = (max_val - min_val) * 0.05
    if margin == 0:
        margin = 0.01
    min_val -= margin
    max_val += margin
    metric_phase_stats[metric]["bar_min"] = min_val
    metric_phase_stats[metric]["bar_max"] = max_val

# Use a colormap to differentiate runs
color_map = plt.get_cmap("tab10")

# We will show legend on the top-right plot (similar to line plots)
legend_handles = []
legend_labels = []
added_labels = set()

for i, metric in enumerate(metrics_to_plot):
    for j, phase in enumerate(["train", "eval"]):
        ax = axes[i, j]
        # Plot bars at integer positions
        for k, run_id in enumerate(RUN_IDS):
            val = best_values[run_id].get((metric, phase), None)
            if val is not None:
                c = color_map(k)
                bar = ax.bar(k, val, color=c)
                # Prepare legend entries only once per run_id
                lbl = run_labels[run_id]
                if lbl not in added_labels:
                    legend_handles.append(bar)
                    legend_labels.append(lbl)
                    added_labels.add(lbl)

        ax.set_title(f"{phase.capitalize()} {metric} (Best)")
        ax.set_ylabel(metric)
        ax.set_ylim(metric_phase_stats[metric]["bar_min"], metric_phase_stats[metric]["bar_max"])
        ax.set_xticks(range(len(RUN_IDS)))
        # No labels on the x-axis since we're using a legend
        ax.set_xticklabels(["" for _ in RUN_IDS])

# Add legend to one of the axes (e.g., top-right subplot)
# We'll just pick the top-right subplot (0, 1) for convenience
axes[0, 1].legend(legend_handles, legend_labels, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "bar_plots.png"))
plt.close()

print(f"Plots saved to {out_dir}")
