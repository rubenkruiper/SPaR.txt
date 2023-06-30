import logging
from typing import Union, List
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

import spar_api_utils as sau
from spar_predictor import SparPredictor
from allennlp.common.util import import_module_and_submodules


# set requests and urllib3 logging to Warnings only todo; not sure if this helps if implemented here only
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# Set up a SPaR.txt predictor
import_module_and_submodules("spar_lib")
Path.cwd().joinpath("SPaR.txt", "trained_models", "debugger_train")
default_experiment_path = Path.cwd().joinpath("experiments", "docker_conf.json")
default_output_path = Path.cwd().joinpath("trained_models", "debugger_train", "model.tar.gz")
spar_predictor = SparPredictor(default_output_path, default_experiment_path)
predictor = spar_predictor.predictor

# Set up the API
SPaR_api = FastAPI()
term_extractor = sau.TermExtractor(max_num_cpu_threads=1)


class ToPredict(BaseModel):
    texts: Union[List[str], str]


class NumPredictors(BaseModel):
    num_cpu_threads: int


@SPaR_api.post("/set_number_of_predictors/")
def set_number_of_predictors(num_concurrent: NumPredictors):
    # Note, this currently re-initializes all predictors
    SPaR_api.term_extractor = sau.TermExtractor(max_num_cpu_threads=num_concurrent.num_cpu_threads)


@SPaR_api.post("/predict_objects/")
def predict_objects(to_be_predicted: ToPredict):
    """
    Predict the object in a given string or list of strings.
    """
    if not to_be_predicted.texts:
        # No input
        return {
            "texts": [], "sentences": [], "predictions": []
        }

    texts = to_be_predicted.texts
    # either string or list of strings
    if type(texts) == str:
        texts = [texts]

    text_and_predictions = term_extractor.process_texts(texts)
    sentences, predictions = zip(*text_and_predictions)
    return {
        "texts": texts,
        "sentences": sentences,
        "predictions": predictions
    }
