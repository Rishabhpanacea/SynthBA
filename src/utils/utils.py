import os
import numpy as np
import nibabel as nib
from src.configuration.config import *
import subprocess

ants_bin = "/home/jupyter-nafisha/Brain_age/ants_2_6_3/ants-2.6.3/bin"
env = os.environ.copy()
env["PATH"] = ants_bin + ":" + env["PATH"]

def get_inputs(inputs_path: str) -> list:
    def get_inputs_from_dir(inputs_path: str) -> list:
        is_nii = lambda f: f.endswith('.nii') or f.endswith('.nii.gz')
        files = os.listdir(inputs_path)
        input_paths = [ os.path.join(inputs_path, f) for f in files if is_nii(f) ]
        return input_paths

    def get_inputs_from_csv(inputs_path: str) -> list:
        with open(inputs_path, 'r') as f:
            return [ p.strip() for p in f.readlines() ]
        
    return get_inputs_from_dir(inputs_path) if os.path.isdir(inputs_path) \
        else get_inputs_from_csv(inputs_path)


def preprocess(input_path, model_path, strip_dir, reg_dir, template_path):

    os.makedirs(strip_dir, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)

    fname = os.path.basename(input_path)
    strip_out = os.path.join(strip_dir, fname)

    cmd = [
        "/home/jupyter-nafisha/.local/bin/nipreps-synthstrip",
        "-i", input_path,
        "-o", strip_out,
        "--model", model_path
    ]

    if not os.path.exists(strip_out):
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError("SynthStrip failed")

    base_name = fname.replace(".nii.gz", "").replace(".nii", "")
    output_prefix = os.path.join(reg_dir, base_name + "_")

    reg_cmd = [
        f"{ants_bin}/antsRegistrationSyNQuick.sh",
        "-d", "3",
        "-f", template_path,
        "-m", strip_out,
        "-o", output_prefix,
        "-n", str(os.cpu_count()),
        "-t", "a"
    ]

    warped_path = output_prefix + "Warped.nii.gz"

    if not os.path.exists(warped_path):
        subprocess.run(reg_cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return warped_path
