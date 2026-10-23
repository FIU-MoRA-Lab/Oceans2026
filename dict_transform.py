import os
import pickle
from dataclasses import asdict

import advection_diffusion_simulator.advection_diffusion_simulator as ads
from utils import read_pkl

folder_path = "swe_data"

for filename in os.listdir(folder_path):
    if filename.endswith(".pkl"):
        filepath = os.path.join(folder_path, filename)
        raw_data = read_pkl(filepath)
        if isinstance(raw_data, dict):
            print(f"{filename} is already a dictionary. Skipping.")
            continue
        
        data = raw_data.to_dict()
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

