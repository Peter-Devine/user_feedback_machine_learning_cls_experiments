from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from scipy.stats import logser
import joblib
import os
import time

from functools import partial

random_state = 123

MODEL_DICT = {
    "KNeighborsClassifier": KNeighborsClassifier,
#     "SVC": partial(SVC, random_state=random_state),
#     "GaussianProcessClassifier": partial(GaussianProcessClassifier, random_state=random_state),
    "RandomForestClassifier": partial(RandomForestClassifier, random_state=random_state),
    "AdaBoostClassifier": partial(AdaBoostClassifier, random_state=random_state),
    "GaussianNB": GaussianNB,
#     "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "LogisticRegression": partial(LogisticRegression, random_state=random_state),
}

MODEL_HYPERPARAMS = {
    "KNeighborsClassifier": dict(n_neighbors = logser(0.999, loc=0)),
    
#     "SVC": dict(C = loguniform(1e-1, 1e2)),
    
#     "GaussianProcessClassifier": dict(max_iter_predict = loguniform(1e2, 5e3)),
    
    "RandomForestClassifier": dict(n_estimators=logser(0.999, loc=0), criterion=["gini", "entropy"], max_depth=logser(0.999, loc=0), min_samples_split=logser(0.999, loc=1), min_samples_leaf=logser(0.999, loc=0)),
    
    "AdaBoostClassifier": dict(n_estimators=logser(0.999, loc=0)),
    
    "GaussianNB": dict(var_smoothing=loguniform(1e-11, 1e-5)),
    
#     "QuadraticDiscriminantAnalysis": dict(reg_param=uniform(loc=0, scale=5), tol=loguniform(1e-3, 1e-1)),
    
    "LogisticRegression": dict(C=uniform(loc=0, scale=4), penalty=['l2', 'none'], solver=["saga", "lbfgs"], tol=loguniform(1e-3, 1e-1)),
}

def get_search_model(model_name, train_idx, eval_idx):
    model = MODEL_DICT[model_name]()
    
    parameters = MODEL_HYPERPARAMS[model_name]
    
    cv = [(train_idx, eval_idx)]
    
    search_model = RandomizedSearchCV(model, parameters, random_state=random_state, cv=cv)
    
    return search_model

def get_feature_str(feature_name, feature_bool):
    if feature_bool:
        return f"with_{feature_name}"
    else:
        return f"without_{feature_name}"

def train_classical_model(model_config, train_dataset, eval_dataset):
    split_type = model_config["split_type"]
    balance_type = model_config["balance_type"]
    get_metadata = model_config["get_metadata"]
    dataset_name = model_config["dataset_name"]
    output_dir = model_config["output_dir"]
    training_type = model_config["training_type"]
    model_names = model_config["model_names"]
    
    remove_stopwords = get_feature_str(model_config["remove_stopwords"], "remove_stopwords")
    stemming = get_feature_str(model_config["stemming"], "stemming")
    is_tfidf = get_feature_str(model_config["is_tfidf"], "is_tfidf")
    bitrigrams = get_feature_str(model_config["bitrigrams"], "bitrigrams")
    sentiment = get_feature_str(model_config["sentiment"], "sentiment")
    
    train_df, train_y = train_dataset
    eval_df, eval_y = train_dataset
    
    # Append train with eval as we have to pass only one dataset to RandomizedSearchCV
    combined_df = train_df.append(eval_df).reset_index(drop=True)
    combined_y = train_y.append(eval_y).reset_index(drop=True)
    
    train_idx = [x for x in range(train_df.shape[0])]
    eval_idx = [x + train_df.shape[0] for x in range(eval_df.shape[0])]
    
    for model_name in model_names:
        print(model_name)
        t0 = time.process_time()
        model = get_search_model(model_name, train_idx, eval_idx)
        model.fit(combined_df.values, combined_y.values)
        has_metadata_str = "with_metadata" if get_metadata else "text_only"
        model_dir = os.path.join(output_dir, "models", training_type, split_type, balance_type, has_metadata_str, model_name, remove_stopwords, stemming, is_tfidf, bitrigrams, sentiment)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, f"{dataset_name}_model.sklearn"), compress=9)
        print(time.process_time() - t0)