import os

import pandas as pd
import seaborn as sns
import torch
from general_utils import (
    ModelAndTokenizer,
)
from patchscopes_utils import set_hs_patch_hooks_glm
from tqdm import tqdm

from data.load_data import load_data
from interpretability.patchscopes.code.patchscopes_utils import (
    evaluate_patch_glm_accuracy,
)

torch.set_grad_enabled(False)
sns.set_theme(
    context="notebook",
    rc={
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "legend.fontsize": 16.0,
    },
)
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style="whitegrid")
tqdm.pandas()

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "/home/students/kolber/seminars/kolber/.cache"

# Load model

glm_model_name = "plenz/GLM-t5-large"
generation_model_name = "google-t5/t5-large"


def prepare_model_and_tokenizer(model_name, torch_dtype=None):
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=torch_dtype,
        device="cpu",
    )
    mt.set_hs_patch_hooks = set_hs_patch_hooks_glm
    mt.model.eval()
    return mt


glm_mt = prepare_model_and_tokenizer(glm_model_name)
generation_mt = prepare_model_and_tokenizer(generation_model_name)

triplet_func_to_idx = {"subject": 0, "relation": 1, "object": 2}

records = []

for radius in range(1, 6):
    for mask_triplet_element in triplet_func_to_idx.keys():
        split = "test"
        data = load_data(split, radius, mask_triplet_element)
        for datapoint in data:
            accuracy = evaluate_patch_glm_accuracy(
                glm_mt=glm_mt,
                generation_mt=generation_mt,
                source_graph=datapoint["graph"],
                target_text=datapoint["target_prompt"],
                how="global",
                position_source=datapoint["source_position"],
                position_target=datapoint["target_position"],
                target_label=datapoint["label"],
            )
            records.append(
                {
                    "radius": radius,
                    "mask_triplet_element": mask_triplet_element,
                    "accuracy": accuracy,
                }
            )


results = pd.DataFrame.from_records(records)
results.to_excel("results.xlsx")
