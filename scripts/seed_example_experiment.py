"""Run initial example experiment to seed results and generate configs/example_experiment.json."""

import json
from pathlib import Path

from terrainlens.data_loader import load_from_path
from terrainlens.experiment import Experiment

# 1. Run Canyon experiment with Ground Truth Reference
canyon_path = Path("data/sample_canyon.png")
ref_path = Path("data/sample_reference_mask.png")

loaded_canyon = load_from_path(canyon_path)
ref_mask = load_from_path(ref_path).data

exp = Experiment(name="Canyon Baseline & Degradation Benchmark", description="Seed benchmark on sample canyon with reference mask evaluation")
cfg = exp.run(
    loaded_data=loaded_canyon,
    feature_method="canny",
    feature_params={"low_threshold": 50, "high_threshold": 150, "aperture_size": 3},
    preprocessing_params={"smooth": True, "kernel_size": 5, "sigma": 1.0},
    reference_mask=ref_mask,
)

# 2. Save into configs/example_experiment.json
configs_dir = Path("configs")
configs_dir.mkdir(parents=True, exist_ok=True)
with open(configs_dir / "example_experiment.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

print(f"Generated example experiment: {exp.experiment_id} saved to configs/example_experiment.json")
