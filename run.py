import json
import time

from data_utils.download_data import download_data
from training_utils.create_models import create_models
from eval_utils.evaluate_model import eval_model
from eval_utils.zero_shot import run_zero_shot

download_data()

model_configs = json.load(open("model_configs_rq1_bug.json"))
model_configs.extend(json.load(open("model_configs_rq1_feature.json")))
model_configs.extend(json.load(open("model_configs_rq2_bug.json")))
model_configs.extend(json.load(open("model_configs_rq2_feature.json")))
model_configs.extend(json.load(open("model_configs_rq3_bug.json")))
model_configs.extend(json.load(open("model_configs_rq3_feature.json")))

for model_config in model_configs:
    print(f"Now running {model_config}")
    if model_config["needs_training"]:
        t0 = time.process_time()
        create_models(model_config)
        print(f"TIME ELAPSED {time.process_time() - t0} FOR {model_config}")
    Tick this model off the configs list once it is done
    json.dump(model_configs[i+1:], open("model_configs.json", "w"))

for model_config in model_configs:
    print(f"Now evaluating {model_config}")
    eval_model(model_config)
    
run_zero_shot()
