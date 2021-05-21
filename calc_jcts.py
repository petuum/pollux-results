import argparse
import datetime
import dateutil.parser
import json
import os
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("policy")
args = parser.parse_args()

records = []
with open(os.path.join(os.path.dirname(__file__), args.policy, args.policy + ".log")) as f:
    records = [json.loads(line) for line in f]

def get_jcts(records):
    jcts = {}
    for job in records[-1]["submitted_jobs"]:
        start = dateutil.parser.parse(job["creationTimestamp"])
        stop = dateutil.parser.parse(job["completionTimestamp"])
        jcts[job["name"]] = (stop - start).total_seconds()
        if job["name"].startswith("imagenet-") and job["phase"] == "Failed":
            old1 = [j for j in records[-120]["submitted_jobs"] if j["name"] == job["name"]][0]
            old2 = [j for j in records[-60]["submitted_jobs"] if j["name"] == job["name"]][0]
            assert old1["allocation"] == old2["allocation"]
            rate = (old2["train"]["epoch"] - old1["train"]["epoch"]) / 3600
            jcts[job["name"]] += (90 - job["train"]["epoch"]) / rate
    return jcts

jcts = get_jcts(records)
print(jcts)
print("Average", sum(jcts.values()) / len(jcts))
imagenet = {k: v for k, v in jcts.items() if k.startswith("imagenet-")}
print("99ptile", sum(imagenet.values()) / len(imagenet))

starts = []
stops = []
for job in records[-1]["submitted_jobs"]:
    start = dateutil.parser.parse(job["creationTimestamp"]).timestamp()
    stop = start + jcts[job["name"]]
    starts.append(start)
    stops.append(stop)

print("Makespan", max(stops) - min(starts))
