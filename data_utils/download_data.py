import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
import numpy as np

from data_utils.dataset_downloaders.label_mappings import dataset_downloaders, bug_nobug_dataset_mappings, feature_nofeature_dataset_mappings

base_save_dir = "data"
random_state = 123

k_splits = 5

def get_labels(df, select_label_name):
    # Apply a mapped label if the original label is in the original label mapping
    df["label"] = df.label.apply(lambda x: select_label_name in x)
    return df

# def split(df, test_size, group_by_app=True):
#     if group_by_app:
#         df = df[~df['app_name'].isna()]
#         splitter = GroupShuffleSplit(test_size=test_size, n_splits=5, random_state = random_state)
#         train_idx, test_idx = next(splitter.split(df.index, groups=df['app_name']))
#     else:
#         splitter = ShuffleSplit(test_size=test_size, n_splits=5, random_state = random_state)
#         train_idx, test_idx = next(splitter.split(df.index))
#     return df.iloc[train_idx], df.iloc[test_idx]

def get_splits(df, test_size, n_splits, group_by_app=True):
    df = df.reset_index(drop=True)
    if group_by_app:
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state = random_state)
        splits = splitter.split(df.index, groups=df['app_name'])
    else:
        splitter = ShuffleSplit(test_size=test_size, n_splits=n_splits, random_state = random_state)
        splits = splitter.split(df.index)
    return splits
    
def balance_train(train_df):
    label_vc = train_df["label"].value_counts()
    label_min_size = label_vc.min()
    label_max_size = label_vc.max()
    minimum_label = label_vc.index[label_vc == label_min_size][0]
    additional_train_rows = train_df[train_df["label"] == minimum_label].sample(n=label_max_size-label_min_size, replace=True)
    return train_df.append(additional_train_rows).reset_index(drop=True)
    
def save(df, dataset_name, label_type, group_by_app=True):
    
    df = df[~df['app_name'].isna()]
    train_test_splits = get_splits(df, 0.2, k_splits, group_by_app)

    for i, (train_eval_idx, test_idx) in enumerate(train_test_splits):
        
        train_eval_df, test_df = df.iloc[train_eval_idx,], df.iloc[test_idx,]
                
        train_eval_df = train_eval_df.reset_index(drop=True)
        
        train_eval_splits = get_splits(train_eval_df, 0.2, 1, group_by_app)
        
        train_idx, eval_idx = next(train_eval_splits)
        
        train_df, eval_df = train_eval_df.iloc[train_idx], train_eval_df.iloc[eval_idx]
        
        split_type = "separated" if group_by_app else "mixed"

        save_dir = os.path.join(base_save_dir, label_type, split_type, str(i))

        os.makedirs(save_dir, exist_ok = True)
        
        train_df.to_json(os.path.join(save_dir, f"{dataset_name}_train.json"), lines=True, orient='records')
        eval_df.to_json(os.path.join(save_dir, f"{dataset_name}_eval.json"), lines=True, orient='records')
        test_df.to_json(os.path.join(save_dir, f"{dataset_name}_test.json"), lines=True, orient='records')

def save_all_df(file_name_list, save_dir, split_type):
    all_dfs = []
    for file_name in file_name_list:
        file_path = os.path.join(save_dir, file_name)
        all_dfs.append(pd.read_json(file_path, lines=True, orient="records"))
    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_json(os.path.join(save_dir, f"all_{split_type}.json"), lines=True, orient="records")
    
def save_all(file_path_list, save_dir):
    train_paths = [x for x in file_path_list if x.endswith("_train.json")]
    eval_paths = [x for x in file_path_list if x.endswith("_eval.json")]
    test_paths = [x for x in file_path_list if x.endswith("_test.json")]
    
    save_all_df(train_paths, save_dir, "train")
    save_all_df(eval_paths, save_dir, "eval")
    save_all_df(test_paths, save_dir, "test")
    
def append_all_data():
    
    for i in range(k_splits):
        for label_type in ["bug", "feature"]:
            separated_dir = os.path.join(base_save_dir, label_type, "separated", str(i))
            save_all(os.listdir(separated_dir), separated_dir)
            mixed_dir = os.path.join(base_save_dir, label_type, "mixed", str(i))
            save_all(os.listdir(mixed_dir), mixed_dir)

# Make sure that any duplicate pieces of text have the same set of labels.
# If two apps have the exact same piece of text, then the metadata from the first app (ordered within original dataset) is chosen.
def de_duplicate_text(df):
    grouped_df = df.groupby("text").first()
    grouped_labels = df.groupby("text")["label"].apply(list)
    grouped_df["label"] = grouped_labels
    return grouped_df.reset_index(drop=False)
    
def download_data():
    
    os.makedirs(base_save_dir, exist_ok = True)
    raw_dir = os.path.join(base_save_dir, "raw")
    os.makedirs(raw_dir, exist_ok = True)

    dataset_list = dataset_downloaders.keys()
    
    for dataset_name, dataset_downloader in dataset_downloaders.items():
        
        print(f"Downloading {dataset_name}")
        
        raw_df = dataset_downloader().reset_index(drop=True)
        
        raw_df = raw_df[~raw_df["text"].isna()]
        
        raw_df = de_duplicate_text(raw_df)
        
        raw_df.to_json(os.path.join(raw_dir, f"{dataset_name}.json"), lines=True, orient="records")
        
        for label_type in ["bug", "feature"]:
            
            df = raw_df.copy()
            
            if label_type == "bug":
                if dataset_name not in bug_nobug_dataset_mappings.keys():
                    continue
                else:
                    label_name = bug_nobug_dataset_mappings[dataset_name]
            elif label_type == "feature":
                if dataset_name not in feature_nofeature_dataset_mappings.keys():
                    continue
                else:
                    label_name = feature_nofeature_dataset_mappings[dataset_name]

            df = get_labels(df, label_name)

            save(df, dataset_name, label_type, group_by_app=True)
            save(df, dataset_name, label_type, group_by_app=False)
    
    # Make an "all" dataset
    append_all_data()