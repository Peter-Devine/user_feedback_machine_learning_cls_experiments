import requests
import zipfile
import json
import os
import io
import shutil

import pandas as pd

def download_maalej():
    task_data_path = os.path.join(".", "temp_data", "maalej_2016_temp")
    os.makedirs(task_data_path, exist_ok = True)

    # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
    # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
    r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path = task_data_path)


    json_path = os.path.join(task_data_path, "REJ_data", "all.json")
    
    df = pd.read_json(json_path)
    df["text"] = df.apply(lambda x: x["comment"] if x["title"] is None else x["title"] + ". " + x["comment"], axis=1)
    df["app_name"] = df["appId"]
    df["meta_rating"] = df["rating"]
    
    shutil.rmtree(task_data_path)

    return df
