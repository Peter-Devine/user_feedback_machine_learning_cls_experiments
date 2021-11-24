import os
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from torch.nn import CrossEntropyLoss
import torch
from sklearn.metrics import brier_score_loss, accuracy_score, precision_recall_fscore_support, ndcg_score, roc_auc_score
import numpy as np

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/mnt/Research/peter-research/peter_devine_nlp_models"
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "/mnt/Research/peter-research/peter_devine_nlp_models"

def get_aprf1(labels, pred_label):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_label, average='binary', zero_division=0)
    acc = accuracy_score(labels, pred_label)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_cutoff_metrics(labels, preds, cutoff):
    pred_label = preds > cutoff
    metrics = get_aprf1(labels, pred_label)
    return {f"{k}_{cutoff}": v for k, v in metrics.items()}

def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    
    predictions = np.argmax(logits, axis=-1)
    results = get_aprf1(labels, predictions)
    
    sig = torch.nn.Sigmoid()
    perc_preds = sig(torch.Tensor(logits)).numpy()[:, 1]
    print(perc_preds)
    try:
        results[f"roc_auc"] = roc_auc_score(labels, perc_preds)
    except Exception as e:
        results[f"roc_auc"] = 0.5
    
    return results

def train_deep_model(model_config, train_dataset, eval_dataset, vocab_size):
    
    split_type = model_config["split_type"]
    get_metadata = model_config["get_metadata"]
    model_name = model_config["model_name"]
    rq_num = model_config["rq"]
    label_type = model_config["label_type"]
    fold_num = model_config["fold_num"]
    dataset_name = "__".join(sorted(model_config["dataset_names"]))
    output_dir = model_config["output_dir"] # "/mnt/Research/peter-research/training_data/user_feedback_classification_models",
    has_metadata_str = "with_metadata" if get_metadata else "text_only"
    
    run_id = [rq_num, label_type, split_type, has_metadata_str, fold_num, dataset_name]
    save_model_folder_list = [output_dir, "models"] + run_id
    save_logs_folder_list = [output_dir, "logs"] + run_id
    saved_model_current_folder = os.path.join(*save_model_folder_list)
    saved_logs_current_folder = os.path.join(*save_logs_folder_list)
    
    training_args = TrainingArguments(
        output_dir=saved_model_current_folder, # output directory
        num_train_epochs=60,              # total number of training epochs
        max_steps=500,
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=512,   # batch size for evaluation
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.00001,               # strength of weight decay
        logging_dir=saved_logs_current_folder, # directory for storing logs
        metric_for_best_model="f1",
        logging_steps=10,
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=False,
    )
    print("Loading config...")
    config = AutoConfig.from_pretrained(model_name, vocab_size = vocab_size)#, cache_dir="/mnt/Research/peter-research/peter_devine_nlp_models")
    print("Config loaded!")
    print("Loading model...")
    config.num_labels = 2
    config.problem_type = "single_label_classification"
    model = AutoModelForSequenceClassification.from_config(config)
    
    print("Model loaded!")

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=eval_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )
    print("Model training initiated...")

    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    
    final_save_model_dir = os.path.join(".", "saved_models")
    os.makedirs(final_save_model_dir, exist_ok=True)
    
    saved_model_current_path = os.path.join(saved_model_current_folder, "pytorch_model.bin")
    saved_model_target_path = os.path.join(*[final_save_model_dir, "__".join(run_id) + "__pytorch_model.bin"])
    
    os.system(f"cp {saved_model_current_path} {saved_model_target_path}")
    os.system(f"rm -r {saved_model_current_folder}")
     
    del trainer
