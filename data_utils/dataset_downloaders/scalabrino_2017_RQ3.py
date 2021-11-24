import pandas as pd

def download_scalabrino_rq3():

    df = pd.read_csv("data_utils/dataset_downloaders/rq3-manually-classified-implemented-reviews.csv")

    df = df.rename(columns = {"body": "text", "category": "label", "App-name": "app_name", "rating": "meta_rating"})
    
    return df
