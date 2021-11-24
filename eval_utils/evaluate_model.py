import os
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
import torch
import json
from copy import copy

from data_utils.create_features import get_features
from training_utils.deep_training import get_aprf1, get_cutoff_metrics, compute_metrics
from data_utils.dataset_downloaders.label_mappings import bug_nobug_dataset_mappings, feature_nofeature_dataset_mappings

mappings_dict = {
    "bug": bug_nobug_dataset_mappings,
    "feature": feature_nofeature_dataset_mappings,
}

def eval_model(model_config):
    
    split_type = model_config["split_type"]
    get_metadata = model_config["get_metadata"]
    model_name = model_config["model_name"]
    rq_num = model_config["rq"]
    label_type = model_config["label_type"]
    fold_num = model_config["fold_num"]
    training_datasets = model_config["dataset_names"]
    is_needs_training = model_config["needs_training"]
    dataset_name = "__".join(sorted(training_datasets))
    output_dir = model_config["output_dir"] # "/mnt/Research/peter-research/training_data/user_feedback_classification_models",
    has_metadata_str = "with_metadata" if get_metadata else "text_only"
    
    # Get saved model path
    run_id = [rq_num, label_type, split_type, has_metadata_str, fold_num, dataset_name]
    final_save_model_dir = os.path.join(".", "saved_models")
    
    # If the model did not need training (I.e. it had already been trained in RQ1), then we simply load the equivalent model from RQ1
    if is_needs_training:
        model_state_path = os.path.join(*[final_save_model_dir, "__".join(run_id) + "__pytorch_model.bin"])
    else:
        rq1_run_id = ["rq1"] + run_id[1:]
        model_state_path = os.path.join(*[final_save_model_dir, "__".join(rq1_run_id) + "__pytorch_model.bin"])
    
    possible_datasets = [dataset_name for dataset_name, label_name in mappings_dict[label_type].items()]
    
    vocab_size = None
    
    # For RQ2, we test on every dataset EXCEPT the dataset we trained on
    if rq_num == "rq2":
        test_dataset_names = [dataset for dataset in possible_datasets if dataset not in training_datasets]
        
    # For RQ1 and RQ3, we test on the same datasets we trained on
    elif rq_num == "rq1" or rq_num == "rq3":
        test_dataset_names = [dataset for dataset in possible_datasets if dataset in training_datasets]
        
        if rq_num == "rq3":
            _, _, _, vocab_size = get_features(model_config)
    
    deep_model = get_deep_model(model_config, model_state_path, vocab_size)
    
    test_model_config = copy(model_config)
    
    results = {}
    
    for test_dataset_name in test_dataset_names:
        
        test_model_config["dataset_names"] = [test_dataset_name]
        
        _, _, test_dataset, vocab_size = get_features(test_model_config)

        results[test_dataset_name] = deep_model.predict(test_dataset).metrics
        
    results_path_id = [".", "results"] + run_id

    results_path = os.path.join(*results_path_id)

    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, "results.json"), "w") as f:
        json.dump(results, f)

def get_deep_model(model_config, model_state_path, vocab_size=None):
    
    model_name = model_config["model_name"]
    
    training_args = TrainingArguments(
        output_dir=os.path.join(".", "temp_eval", "models"), # output directory
        per_device_eval_batch_size=64,   # batch size for evaluation
        logging_dir=os.path.join(".", "temp_eval", "logs"), # directory for storing logs
        fp16=False,
    )
    
    if vocab_size is None:
        config = AutoConfig.from_pretrained(model_name)#, cache_dir="/mnt/Research/peter-research/peter_devine_nlp_models")
    else:
        config = AutoConfig.from_pretrained(model_name, vocab_size = vocab_size)
        
    config.num_labels = 2
    config.problem_type = "single_label_classification"
    model = AutoModelForSequenceClassification.from_config(config)

    model.load_state_dict(torch.load(model_state_path))

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        compute_metrics=compute_metrics,
    )

    return trainer