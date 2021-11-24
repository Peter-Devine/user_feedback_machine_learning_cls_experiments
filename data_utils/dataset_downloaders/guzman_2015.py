import pandas as pd

def download_guzman():
    df = pd.read_csv("https://ase.in.tum.de/lehrstuhl_1/images/publications/Emitza_Guzman_Ortega/truthset.tsv",
                     sep="\t", names=["column0", "label", "column2", "app", "meta_rating", "text"])

    int_to_str_label_map = {
        5: "Praise",
        3: "Feature shortcoming",
        1: "Bug report",
        2: "Feature strength",
        7: "Usage scenario",
        4: "User request",
        6: "Complaint",
        8: "Noise"
    }

    df["label"] = df.label.apply(lambda x: int_to_str_label_map[x])

    int_to_app_name_map = {
        6: "Picsart",
        8: "Whatsapp",
        7: "Pininterest",
        1: "Angrybirds",
        3: "Evernote",
        5: "Tripadvisor",
        2: "Dropbox"
    }

    df["app_name"] = df.app.apply(lambda x: int_to_app_name_map[x])

    return df
