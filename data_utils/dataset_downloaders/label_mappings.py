from data_utils.dataset_downloaders.chen_2014 import download_chen
from data_utils.dataset_downloaders.ciurumelea_2017 import download_ciurumelea
from data_utils.dataset_downloaders.guzman_2015 import download_guzman
from data_utils.dataset_downloaders.maalej_2016 import download_maalej
from data_utils.dataset_downloaders.scalabrino_2017_RQ1 import download_scalabrino_rq1
from data_utils.dataset_downloaders.scalabrino_2017_RQ3 import download_scalabrino_rq3
from data_utils.dataset_downloaders.tizard_2019 import download_tizard
from data_utils.dataset_downloaders.williams_2017 import download_williams

dataset_downloaders = {
    "ciurumelea_2017": download_ciurumelea,
    "guzman_2015": download_guzman,
    "maalej_2016": download_maalej,
    "scalabrino_2017_RQ1": download_scalabrino_rq1,
    "scalabrino_2017_RQ3": download_scalabrino_rq3,
    "tizard_2019": download_tizard,
    "williams_2017": download_williams,
}

bug_nobug_dataset_mappings = {
    "ciurumelea_2017": "ERROR",
    "guzman_2015": "Bug report",
    "scalabrino_2017_RQ1": "BUG",
    "scalabrino_2017_RQ3": "BUG",
    "maalej_2016": "Bug",
    "tizard_2019":"apparent bug",
    "williams_2017": "bug",
}

feature_nofeature_dataset_mappings = {
    "guzman_2015": "User request",
    "scalabrino_2017_RQ1": "FEATURE",
    "scalabrino_2017_RQ3": "FEATURE",
    "maalej_2016": "Feature",
    "tizard_2019":"feature request",
    "williams_2017": "fea",
}