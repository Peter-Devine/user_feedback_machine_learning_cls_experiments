import os
import torch
import pandas as pd
from transformers import AutoTokenizer

class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, is_multiclass=False):
        self.encodings = encodings
        self.labels = labels
        self.is_multiclass = is_multiclass

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        if self.labels is not None:
            if self.is_multiclass:
                item['labels'] = torch.tensor(self.labels[idx, :], dtype=torch.float)
            else:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def get_tokens(text_list, tokenizer, special_keywords=[]):
    return tokenizer(text = text_list, padding=True, truncation=True, max_length=256, return_tensors="np")

def replace_space_with_underbar(text):
    return str(text).replace(" ", "_")

def get_metadata_cols(df_cols):
    return [x for x in df_cols if x.startswith("meta_")]

def get_text_and_unique_metadata(df, get_metadata=False):
    
    text_list = df["text"]
    
    if get_metadata:
        metadata_cols = get_metadata_cols(df.columns)
        
        metadata_vars_list = []
        metadata_unique_list = []
        for metadata_col in metadata_cols:
            metadata_col_clean = replace_space_with_underbar(metadata_col)
            metadata_vars = df[metadata_col].apply(lambda x: f"[METADATA_{metadata_col_clean}_{replace_space_with_underbar(x)}]")
            
            unique_metadata_vars = metadata_vars.unique().tolist()
            metadata_unique_list.extend(unique_metadata_vars)
            
            metadata_vars_list.append(metadata_vars)
            
        for metadata_vars in metadata_vars_list:
            text_list = metadata_vars + " " + text_list

    else:
        metadata_unique_list = []
            
    return text_list.to_list(), metadata_unique_list


def get_transformer_features(train_df, eval_df, test_df, tokenizer_name, label_column, get_metadata=False):
    train_text, train_unique_metadata = get_text_and_unique_metadata(train_df, get_metadata)
    eval_text, _ = get_text_and_unique_metadata(eval_df, get_metadata)
    test_text, _ = get_text_and_unique_metadata(test_df, get_metadata)
    
#     lens = train_text.str.len()
#     print(train_text[lens >= lens.max()].iloc[0])
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, additional_special_tokens=train_unique_metadata)
    train_tokens = get_tokens(train_text, tokenizer)
    eval_tokens = get_tokens(eval_text, tokenizer)
    test_tokens = get_tokens(test_text, tokenizer)
    
    vocab_size = tokenizer.vocab_size + len(train_unique_metadata)
    
    print(vocab_size)
    
    train_labels = train_df[label_column]
    eval_labels = eval_df[label_column]
    test_labels = test_df[label_column]
    
    train_dataset = FeedbackDataset(train_tokens, train_labels)
    eval_dataset = FeedbackDataset(eval_tokens, eval_labels)
    test_dataset = FeedbackDataset(test_tokens, test_labels)
    
    return train_dataset, eval_dataset, test_dataset, vocab_size

def get_dummies(category_series, possible_categories=None):
    if possible_categories is None:
        return pd.get_dummies(category_series, prefix="meta_"), category_series.unique()
    else:
        return pd.get_dummies(category_series.astype(pd.CategoricalDtype(categories=possible_categories)), prefix="meta_")

def join_dfs(original_df, df_to_join):
    if original_df is None:
        return df_to_join
    else:
        return original_df.join(df_to_join)

def is_col_numeric(col_series):
    try:
        col_series.astype(float)
        return True
    except Exception:
        return False
    
def get_dummy_df(train_df, eval_df, test_df, metadata_cols):
    
    train_dummies_df = None
    eval_dummies_df = None
    test_dummies_df = None
    
    for metadata_col in metadata_cols:
        
        if is_col_numeric(train_df[metadata_col]) and is_col_numeric(eval_df[metadata_col]) and is_col_numeric(test_df[metadata_col]):
            # Is column a float that we can make into 1 feature?
            train_dummies = pd.DataFrame(train_df[metadata_col].astype(float), columns=[metadata_col])
            eval_dummies = pd.DataFrame(eval_df[metadata_col].astype(float), columns=[metadata_col])
            test_dummies = pd.DataFrame(test_df[metadata_col].astype(float), columns=[metadata_col])
        else:
            train_dummies, dummy_categories = get_dummies(train_df[metadata_col])
            eval_dummies = get_dummies(eval_df[metadata_col], dummy_categories)
            test_dummies = get_dummies(test_df[metadata_col], dummy_categories)
        
        train_dummies_df = join_dfs(train_dummies_df, train_dummies)
        eval_dummies_df = join_dfs(eval_dummies_df, eval_dummies)
        test_dummies_df = join_dfs(test_dummies_df, test_dummies)
        
    return train_dummies_df, eval_dummies_df, test_dummies_df

def get_data_splits(dataset_names, label_type, split_type, fold_num):
    
    combined_train_df = None
    combined_eval_df = None
    combined_test_df = None
    for dataset_name in dataset_names:
        train_df = pd.read_json(os.path.join("data", label_type, split_type, fold_num, f"{dataset_name}_train.json"), lines=True, orient='records')
        eval_df = pd.read_json(os.path.join("data", label_type, split_type, fold_num, f"{dataset_name}_eval.json"), lines=True, orient='records')
        test_df = pd.read_json(os.path.join("data", label_type, split_type, fold_num, f"{dataset_name}_test.json"), lines=True, orient='records')
        
        combined_train_df = train_df if combined_train_df is None else combined_train_df.append(train_df).reset_index(drop=True)
        combined_eval_df = eval_df if combined_eval_df is None else combined_eval_df.append(eval_df).reset_index(drop=True)
        combined_test_df = test_df if combined_test_df is None else combined_test_df.append(test_df).reset_index(drop=True)
        
    return combined_train_df, combined_eval_df, combined_test_df

def get_features(features_dict):
    
    dataset_names = features_dict["dataset_names"]
    split_type = features_dict["split_type"]
    label_type = features_dict["label_type"]
    fold_num = features_dict["fold_num"]
    
    train_df, eval_df, test_df = get_data_splits(dataset_names, label_type, split_type, fold_num)
    
    get_metadata = features_dict["get_metadata"]
    label_column = features_dict["label_column"]
    
    tokenizer_name = features_dict["model_name"]
    train_dataset, eval_dataset, test_dataset, vocab_size = get_transformer_features(train_df, eval_df, test_df, tokenizer_name, label_column, get_metadata=get_metadata)
    return train_dataset, eval_dataset, test_dataset, vocab_size