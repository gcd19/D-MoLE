import json
import os

import pandas as pd


def get_enable_layers(scores: pd.DataFrame, budget_portion: float):
    # Split scores into LLM and ViT scores
    llm_scores = (
        scores[scores["layer"].str.contains("language_model")]
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )
    vit_scores = (
        scores[scores["layer"].str.contains("vision_model")]
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )

    # Calculate adaptive budget portions for LLM and ViT
    total_score = llm_scores["score"].sum() + vit_scores["score"].sum()
    llm_budget_portion = llm_scores["score"].sum() / total_score
    vit_budget_portion = vit_scores["score"].sum() / total_score

    # Calculate the number of layers to enable based on budget portion for LLM and ViT
    total_budget = int(len(scores) * budget_portion + 0.5)

    # Set minimum portion for ViT layers to at least 1/4 of the total budget
    vit_min_budget = max(
        int(total_budget * 0.25), int(total_budget * vit_budget_portion + 0.5)
    )
    llm_enable_layers_count = total_budget - vit_min_budget

    # Select top layers from LLM and ViT
    enable_layers = []

    # LLM layers selection
    if llm_enable_layers_count > 0:
        llm_threshold = llm_scores.iloc[llm_enable_layers_count - 1]["score"]
        llm_enable_layers = list(
            llm_scores[llm_scores["score"] > llm_threshold]["layer"]
        )

        if len(llm_scores[llm_scores["score"] == llm_threshold]) > 1:
            remaining_layers = llm_scores[llm_scores["score"] == llm_threshold][
                "layer"
            ].sample(
                n=llm_enable_layers_count - len(llm_enable_layers), random_state=42
            )
            llm_enable_layers.extend(remaining_layers.tolist())
        else:
            llm_enable_layers = list(
                llm_scores[llm_scores["score"] >= llm_threshold]["layer"]
            )

    # ViT layers selection
    if vit_min_budget > 0:
        vit_threshold = vit_scores.iloc[vit_min_budget - 1]["score"]
        vit_enable_layers = list(
            vit_scores[vit_scores["score"] > vit_threshold]["layer"]
        )

        if len(vit_scores[vit_scores["score"] == vit_threshold]) > 1:
            remaining_layers = vit_scores[vit_scores["score"] == vit_threshold][
                "layer"
            ].sample(n=vit_min_budget - len(vit_enable_layers), random_state=42)
            vit_enable_layers.extend(remaining_layers.tolist())
        else:
            vit_enable_layers = list(
                vit_scores[vit_scores["score"] >= vit_threshold]["layer"]
            )

    # Combine LLM and ViT enabled layers
    enable_layers.extend(llm_enable_layers)
    enable_layers.extend(vit_enable_layers)

    # Format layer names if "base_model" exists in the names
    if enable_layers and "base_model" in enable_layers[0]:
        enable_layers = [".".join(layer.split(".")[2:]) for layer in enable_layers]
    else:
        enable_layers = [".".join(layer.split(".")[0:]) for layer in enable_layers]

    # Ensure that the number of enabled layers matches the expected count
    assert (
        len(enable_layers) == total_budget
    ), "The number of enabled layers is not correct."

    return enable_layers


zc_scores_dir = "results/zc_scores"

tasks = [
    "vizwiz_caption",
    "skvg",
    "textcaps",
    "iconqa",
    "ocrvqa",
    "flickr30k",
    "vizwiz",
    "kvqa",
    "pmcvqa",
]
num_tasks = len(tasks)

dataframes = {}

cur_arch = {}

for i in range(1, num_tasks + 1):
    taskname = tasks[i - 1]  # Map the task names dynamically
    file_name = os.path.join(zc_scores_dir, f"{i}_InternVL2-2B_{taskname}_score.csv")
    if os.path.exists(file_name):
        dataframes[taskname] = pd.read_csv(file_name)

    for layer in dataframes[taskname]["layer"]:
        if layer not in cur_arch:
            cur_arch[layer] = []

    enable_layers = get_enable_layers(dataframes[taskname], budget_portion=0.5)
    for layer in enable_layers:
        if layer not in cur_arch:
            cur_arch[layer] = [i]
        else:
            cur_arch[layer].append(i)

    # save the current architecture and sort keys
    os.makedirs("dmole_arch", exist_ok=True)
    with open(f"dmole_arch/{i}_InternVL2-2B_{taskname}_arch.json", "w") as f:
        json.dump(cur_arch, f, indent=4, sort_keys=True)

print("D-MoLE architecture saved.")