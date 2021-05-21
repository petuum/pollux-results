import argparse
import dateutil.parser
import glob
import json
import os
import pandas
from datetime import datetime, timedelta
import seaborn
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("policy", nargs="+")
parser.add_argument("--value", choices=["efficiency", "allocation"], required=True)
args = parser.parse_args()

def parse_efficiency(policy):
    records = []
    min_ts = None
    with open(os.path.join(os.path.dirname(__file__), policy, policy + ".log")) as f:
        for line in f:
            rec = json.loads(line)
            if min_ts is None and rec["submitted_jobs"]:
                min_ts = float(rec["timestamp"])
            for job in rec["submitted_jobs"]:
                if job.get("phase") not in ("Running", "Starting", "Stopping"):
                    continue
                if job.get("train", {}).get("gradParams") is None:
                    continue
                scale = job["train"]["batchSize"] / job["train"]["initBatchSize"]
                sqr = job["train"]["gradParams"]["norm"]
                var = job["train"]["gradParams"]["var"]
                gain = (var + sqr) / (var / scale + sqr)
                num_gpus = len(job.get("allocation", []))
                records.append({
                    "efficiency": gain / scale * num_gpus,
                    "num_gpus": num_gpus,
                    "time": timedelta(seconds=float(rec["timestamp"]) - min_ts)
                })
    return pandas.DataFrame.from_records(records)

def policy_name(policy):
    name = policy.split("-")[0].capitalize()
    if name == "Optimus":
        name += "+Oracle+TunedJobs"
    elif name == "Tiresias":
        name += "+TunedJobs"
    else:
        name += " (p = -1)"
    return name

if args.value == "efficiency":
    df_list = []
    for policy in args.policy:
        df = parse_efficiency(policy)
        df = df.set_index("time").resample("30min").sum().reset_index()
        df["policy"] = policy_name(policy)
        df_list.append(df)
    df = pandas.concat(df_list)
    df["efficiency"] = df["efficiency"] / df["num_gpus"] * 100
else:
    df_list = []
    for policy in args.policy:
        df = parse_efficiency(policy)
        df = df.groupby("time").sum().reset_index()
        df = df.set_index("time").resample("30min").mean().reset_index()
        df["policy"] = policy_name(policy)
        df_list.append(df)
    df = pandas.concat(df_list)

df["hours"] = df.time.dt.total_seconds() / 3600
print(df)

seaborn.set_style("whitegrid")
seaborn.set_style("whitegrid", {"font.family": "serif"})
matplotlib.rcParams['legend.labelspacing'] = 0.4
matplotlib.rcParams['legend.columnspacing'] = -5
fig, ax = matplotlib.pyplot.subplots(figsize=(5.0, 2.4))
seaborn.lineplot(x=df.hours, y=(df.efficiency if args.value == "efficiency" else df.num_gpus), hue=df.policy, ax=ax)
ax.lines[0].set_linestyle("-")
ax.lines[1].set_linestyle("--")
ax.lines[2].set_linestyle(":")
ax.set_xlim(0, 24)
ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax.set_xlabel("Time (hours)")
if args.value == "allocation":
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 32, 64])
    ax.set_ylabel("Alloc. GPUs")
else:
    ax.set_ylim(0, 100)
    ax.set_ylabel("Avg Stat Eff")
ax.yaxis.set_label_coords(-0.105, 0.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(ax.lines, labels[1:], fontsize=10, ncol=2, loc=(0.0, 1.1))

fig.tight_layout()
fig.savefig("physical_stat_eff.pdf" if args.value == "efficiency" else "physical_allocation.pdf")
matplotlib.pyplot.show()
