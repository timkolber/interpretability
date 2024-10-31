import os
import sys

sys.path.append("/home/students/kolber/Investigating-GLM-hidden-states/GraphLanguageModels/")

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
from GraphLanguageModels.models.graph_T5.classifier import GraphT5Classifier
from transformers import AutoModel

print(torch.cuda.is_available())

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

glm_model_name = "plenz/GLM-t5-small"
generation_model_name = "google-t5/t5-small"

def load_finetuned_model(model_path):
    model = AutoModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=None,
        cache_dir="/home/students/kolber/seminars/kolber/.cache/",
        trust_remote_code=True,
        revision="main",
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def prepare_model_and_tokenizer(model_name, torch_dtype=None, fine_tuned_path=None):
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model=load_finetuned_model(fine_tuned_path) if fine_tuned_path else None,
    )
    mt.set_hs_patch_hooks = set_hs_patch_hooks_glm
    mt.model.eval()
    return mt

glm_mt = prepare_model_and_tokenizer(glm_model_name, fine_tuned_path="/home/students/kolber/Investigating-GLM-hidden-states/checkpoints/best_epoch_t5encoder")
generation_mt = prepare_model_and_tokenizer(generation_model_name)

triplet_func_to_idx = {"subject": 0, "relation": 1, "object": 2}

records = []
print(glm_mt.device)
print(generation_mt.device)
print(glm_mt.model.device)
print(generation_mt.model.device)

for radius in tqdm(range(1, 6)):
    for mask_triplet_element in triplet_func_to_idx.keys():
        split = "test"
        data = load_data(split, radius, mask_triplet_element)
        for datapoint in tqdm(data):
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
results.to_excel("results_ft.xlsx")
