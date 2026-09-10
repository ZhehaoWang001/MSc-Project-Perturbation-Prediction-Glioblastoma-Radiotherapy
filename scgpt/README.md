# scGPT training

The selected implementation is `Tutorial_Perturbation_Prediction_v12_revised.py`,
with `submit_myriad.sh` and a `requirements.txt` pointing to the
[shared environment record](../environments/README.md). 

Stage both files in your Myriad working directory. The job expects the virtual
environment at `~/Scratch/scgpt_tutorial_folder/scgpt_venv`, private data under the
working directory, and upstream model files under `./save/scGPT_human`. Adjust
those paths for another machine. Preserve required module settings from the job.

The retained `dev_perturb_adamson-Jun28-15-05` checkpoint is an earlier saved run;
the current selected training source is not claimed to be the exact historical
revision that produced it. `adamson` is the historical cache label in the custom
workflow, not a replacement for the glioblastoma study dataset.
