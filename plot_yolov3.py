import argparse
import dateutil.parser
import glob
import json
import numpy as np
import os
import pandas
from datetime import datetime, timedelta
import seaborn
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("policy")
parser.add_argument("--value", choices=["jobs", "gpus", "bsz", "eff"], required=True)
args = parser.parse_args()

def parse_yolov3(policy, value, name):
    records = []
    min_ts = None
    with open(os.path.join(os.path.dirname(__file__), policy, policy + ".log")) as f:
        for line in f:
            rec = json.loads(line)
            if min_ts is None and rec["submitted_jobs"]:
                min_ts = float(rec["timestamp"])
            active = ("Running", "Starting", "Stopping", "Pending")
            if value == "jobs" and min_ts is not None:
                records.append({
                    "value": sum(1 for job in rec["submitted_jobs"] if job.get("phase") in active),
                    "time": timedelta(seconds=float(rec["timestamp"]) - min_ts),
                })
                continue
            for job in rec["submitted_jobs"]:
                if job["name"] == name and job.get("phase") in active:
                    break
            else:
                continue
            if "batchSize" in job.get("train", {}):
                records.append({
                    "time": timedelta(seconds=float(rec["timestamp"]) - min_ts),
                })
                if value == "bsz":
                    records[-1]["value"] = job["train"]["batchSize"]
                elif value == "gpus":
                    records[-1]["value"] = len(job["allocation"])
                else:
                    sqr = job["train"]["gradParams"]["norm"]
                    var = job["train"]["gradParams"]["var"]
                    scale = job["train"]["batchSize"] / job["train"]["initBatchSize"]
                    gain = (var + sqr) / (var / scale + sqr)
                    records[-1]["value"] = gain / scale * 100
    return pandas.DataFrame.from_records(records)

df_list = []
for name in ["yolov3-41", "yolov3-155"]:
    df = parse_yolov3(args.policy, args.value, name)
    df = df.set_index("time").resample("15min").mean().reset_index()
    df["name"] = name
    df_list.append(df)
df = pandas.concat(df_list)

df["hours"] = df.time.dt.total_seconds() / 3600

print(df)

seaborn.set_style("whitegrid")
seaborn.set_style("whitegrid", {"font.family": "serif"})
fig, ax = matplotlib.pyplot.subplots(figsize=(3, 1.63 if args.value == "bsz" else 1.5))
seaborn.lineplot(x=df.hours, y=df.value, hue=df.name, ax=ax, legend=False)
for line in ax.lines:
    line.set_linestyle("-")
    line.set_color(ax.lines[0].get_color())
#ax.lines[1].set_linestyle("--")
#ax.lines[2].set_linestyle(":")
ax.set_xlim(2, 12)
ax.set_ylim(0, None)
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_xlabel("Time (hours)")
ax.set_ylabel({"bsz": "Batch Size", "gpus": "Num GPUs", "eff": "Stat Eff.", "jobs": "Active Jobs"}[args.value])
if args.value == "bsz":
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_yticks([0, 300, 600])
elif args.value == "gpus":
    ax.set_yticks([0, 10, 20])
elif args.value == "eff":
    ax.set_yticks([0, 50, 100])
ax.yaxis.set_label_coords(-0.16, 0.5)
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(ax.lines, labels[1:], fontsize=10, ncol=3, loc=(-0.12, 1.25))

fig.tight_layout()
fig.savefig("physical_yolov3_{}.pdf".format(args.value))
matplotlib.pyplot.show()
