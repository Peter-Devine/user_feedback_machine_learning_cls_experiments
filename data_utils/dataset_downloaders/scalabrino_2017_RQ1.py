import pandas as pd
from scipy.io import arff
import os
import requests
import zipfile
import shutil

def download_scalabrino_rq1():
    
    task_data_path = os.path.join(".", "temp_data", "scalabrino_2017_rq1_temp")
    os.makedirs(task_data_path, exist_ok = True)

#     r = requests.get("https://dibt.unimol.it/reports/clap/downloads/rq1-reviews.zip")
#     z = zipfile.ZipFile(io.BytesIO(r.content))
    z = zipfile.ZipFile("data_utils/dataset_downloaders/rq1-reviews.zip", "r")
    z.extractall(path = task_data_path)

    arff_path = os.path.join(task_data_path, "rq1-3000.arff")
    
    arff_data = arff.loadarff(arff_path)
    
    shutil.rmtree(task_data_path)

    df = pd.read_csv("data_utils/dataset_downloaders/rq1-raw-data.csv")

    df = df.rename(columns = {"review": "text", "category": "label", "app": "app_name", "rating": "meta_rating"})

    df["meta_app_category"] = pd.DataFrame(arff_data[0])["AppCategory"].apply(lambda x: x.decode("utf-8"))
    
    return df
