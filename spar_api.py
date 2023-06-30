import logging
from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel

import spar_api_utils as sau

# set requests and urllib3 logging to Warnings only todo; not sure if this helps if implemented here only
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Set up the API  and SPaR.txt predictor
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

    sentences, predictions = SPaR_api.term_extractor.process_texts(texts)
    return {
        "texts": texts,
        "sentences": sentences,
        "predictions": predictions
    }
