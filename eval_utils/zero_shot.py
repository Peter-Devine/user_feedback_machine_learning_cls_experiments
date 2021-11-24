from transformers import pipeline
import os
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import json
    
def get_f1_score(classifier, df, candidate_labels, cutoffs):

    text_to_pred = df.text.tolist()

    BATCH_SIZE = 64

    scores = []

    for i in range(0, len(text_to_pred), BATCH_SIZE):
        print(i)
        text_batch = text_to_pred[i:(i+BATCH_SIZE)]
        preds = classifier(text_batch, candidate_labels, multi_label=True)
        scores.extend([np.mean(x[0]['scores']) for x in preds])
    
    results_cutoffs = {}
    
    for cutoff in cutoffs:
        results_cutoffs[cutoff] = f1_score(df.label.tolist(), np.array(scores) > cutoff)
    
    return results_cutoffs

def run_bug_and_feature_zs(classifier):
    
    for label in ["bug", "feature"]:
        data_dir = os.path.join("data", label, "mixed")

        if label == "bug":
            candidate_labels = ['bug report']
        elif label == "feature":
            candidate_labels = ['feature request']

        zs_results = {}

        for fold_num in os.listdir(data_dir):
            fold_dir = os.path.join(data_dir, fold_num)

            for dataset in os.listdir(fold_dir):

                if "all_" in dataset:
                    continue
                if "test.json" not in dataset:
                    continue

                print(dataset)
                dataset_path = os.path.join(fold_dir, dataset)

                df = pd.read_json(dataset_path, lines=True)

                if dataset not in zs_results.keys():
                    zs_results[dataset] = []

                zs_results[dataset].append(get_f1_score(classifier, df, candidate_labels, [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]))
                
        json.dump(zs_results, open(os.path.join("results", f"zero_shot_{label}.json"), "w"))
        
def run_zero_shot():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    run_bug_and_feature_zs(classifier)