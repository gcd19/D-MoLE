import json
import os

from internvl.scorer.score_policy import load_score_frame, select_enable_layers


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

cur_arch = {}
require_authoritative = os.environ.get("DMOLE_REQUIRE_AUTHORITATIVE_SCORE", "").strip() == "1"

for i in range(1, num_tasks + 1):
    taskname = tasks[i - 1]
    file_name = os.path.join(zc_scores_dir, f"{i}_InternVL2-2B_{taskname}_score.csv")
    if not os.path.exists(file_name):
        raise FileNotFoundError(file_name)

    score_frame = load_score_frame(
        file_name,
        require_authoritative=require_authoritative,
    )

    for layer in score_frame["layer"]:
        cur_arch.setdefault(layer, [])

    enable_layers = select_enable_layers(
        score_frame,
        budget_portion=0.5,
        require_authoritative=require_authoritative,
    )
    for layer in enable_layers:
        cur_arch.setdefault(layer, []).append(i)

    os.makedirs("dmole_arch", exist_ok=True)
    with open(f"dmole_arch/{i}_InternVL2-2B_{taskname}_arch.json", "w") as f:
        json.dump(cur_arch, f, indent=4, sort_keys=True)

print("D-MoLE architecture saved.")
