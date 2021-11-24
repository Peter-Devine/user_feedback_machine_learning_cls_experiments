import requests
import zipfile
import time
import os
import io
import sqlite3
import shutil

import pandas as pd

def download_tizard():

    task_data_path = os.path.join(".", "temp_data", "tizard_2019")
    os.makedirs(task_data_path, exist_ok = True)

    r = requests.get("https://zenodo.org/record/3340156/files/RE_Submission_17-master.zip?download=1")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=task_data_path)

    db = sqlite3.connect(os.path.join(task_data_path, "RE_Submission_17-master", "VLC_labelled_sentences_4RE.sqlite"))

    column_names = [x[1] for x in db.execute("""PRAGMA table_info('labelled_sentences');""")]

    row_data = [row for row in db.execute("SELECT * from labelled_sentences")]

    db.close()

    df = pd.DataFrame(row_data, columns=column_names)

    bad_label_list = [
        "and Remove Cookies Warning\xa0! ",
        "and Remove Cookies  ",
        "remove",
        "continued from previous"
    ]

    # Remove the four rows with mistake labels
    df = df.loc[~df.label.apply(lambda x: x in bad_label_list)]

    def forum_label_transformer(raw_label):
        lower_label = raw_label.lower()

        if "application guidance" in lower_label:
            return "application guidance"
        if "non-informative" in lower_label:
            return "non-informative"
        if "apparent bug" in lower_label:
            return "apparent bug"
        if "question on application" in lower_label:
            return "question on application"
        if "feature request" in lower_label:
            return "feature request"
        if "help seek" in lower_label:
            return "help seeking"
        if "user setup" in lower_label:
            return "user setup"
        if "usage" in lower_label:
            return "application usage"
        if "is-background" in lower_label:
            return "question on background"
        if "attempted solution" in lower_label:
            return "attempted solution"
        if "requesting" in lower_label:
            return "requesting more information"
        if "dispraise" in lower_label:
            return "dispraise for application"
        if "praise application" in lower_label:
            return "praise for application"
        if "acknowledgement" in lower_label:
            return "acknowledgement of problem resolution"
        if "problem resolution" in lower_label:
            return "acknowledgement of problem resolution"
        if "agreeing with the problem" in lower_label:
            return "agreeing with the problem"
        if "limitation confirmation" in lower_label:
            return "limitation confirmation"
        if "bug confirmation" in lower_label:
            return "malfunction confirmation"
        if "agreeing with the request" in lower_label:
            return "agreeing with the feature request"
        return "other"

    # Map all labels to their lower-case proper label. Mark all labels outside of the literature label set as "other".
    df["original_label"] = df.label
    df.label = df.label.apply(forum_label_transformer)

    df = df.rename(columns={'sentence': 'text'})
    df["app_name"] = df["topic_forum"].str.lower()
    df["sublabel"] = df["post_position"].str.lower().str.strip()
    
    def is_num(x):
        try:
            float(x)
            return True
        except Exception:
            return False
        
    def get_num(x):
        if is_num(x):
            return float(x)
        else:
            return 0
    
    df["post_position"] = df["post_position"].str.lower().apply(lambda x: 'original post' if x.lower() == 'initial post' else x)
    df["post_position"] = df["post_position"].apply(lambda x: x.lower().replace(" ", ""))
    
    df["meta_post_position_num"] = df["post_position"].apply(lambda x: x.split("-")[0].strip())
    df["meta_post_position_num"] = df["meta_post_position_num"].apply(get_num)
    
    # If post position cannot be parsed as float, then it has "op" in the str, is "title", or "original post".
    # Thus, if the post position is not numeric, the author is the original poster
    df["meta_is_op"] = ~df["post_position"].apply(is_num)
    
    df["meta_topic_forum"] = df["topic_forum"].str.lower()
    df["meta_user_level"] = df["user_level"].str.lower().str.replace(" ", "").str.replace("-", "")

    shutil.rmtree(task_data_path)

    return df