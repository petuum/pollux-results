# Pollux Testbed Experiments Results

This repository contains the raw experiment logs and analysis scripts for
reproducing the results presented in Section 5.2 of the OSDI 2021 paper
"Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning".
To get started, we recommend using a virtualenv or conda environment:
```
$ conda create -n pollux python=3.8
$ conda activate pollux
$ python3 -m pip install seaborn dateutil
```

Decompress all results files:
```
$ unzip '*.zip'
```

The results can then be reproduced using the commands below.

## Reproducing Table 2

- **Pollux (p = -1):** `python calc_jcts.py pollux-p-1-a`
- **Optimus+Oracle+TunedJobs:** `python calc_jcts.py optimus-tunedjobs`
- **Tiresias+TunedJobs:** `python calc_jcts.py tiresias-tunedjobs`
- **Optimus+Oracle:** `python calc_jcts.py optimus-realistic`
- **Tiresias:** `python calc_jcts.py tiresias-realistic`
- **Pollux (p = +1):** `python calc_jcts.py pollux-p1`
- **Pollux (p = -10):** `python calc_jcts.py pollux-p-10`

## Reproducing Figure 5

- **Figure 5a:** `python plot_cluster.py --value allocation pollux-p-1-a optimus-tunedjobs tiresias-tunedjobs`
- **Figure 5b:** `python plot_cluster.py --value efficiency pollux-p-1-a optimus-tunedjobs tiresias-tunedjobs`

## Reproducing Figure 6

- **Left Row 1:** `python plot_imagenet.py --value jobs pollux-p-1-b`
- **Left Row 2:** `python plot_imagenet.py --value gpus pollux-p-1-b`
- **Left Row 3:** `python plot_imagenet.py --value bsz pollux-p-1-b`
- **Left Row 4:** `python plot_imagenet.py --value eff pollux-p-1-b`
- **Right Row 1:** `python plot_yolov3.py --value jobs pollux-p-1-a`
- **Right Row 2:** `python plot_yolov3.py --value gpus pollux-p-1-a`
- **Right Row 3:** `python plot_yolov3.py --value bsz pollux-p-1-a`
- **Right Row 4:** `python plot_yolov3.py --value eff pollux-p-1-a`
