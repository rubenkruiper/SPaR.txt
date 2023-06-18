import logging
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

## set up SPaR predictor
from spar_predictor import SparPredictor
from allennlp.common.util import import_module_and_submodules
import spar_serving_utils as su


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


## Set up the API
SPaR_api = FastAPI()


class ToPredict(BaseModel):
    sentence: str

# @SPaR_api.get("/tokenize/{input_str}")
# def tokenize(input_str: Optional[str] = Query(default=Required, max_length=1000)):
#     return predictor._dataset_reader.tokenizer.tokenize(input_str)


@SPaR_api.post("/predict_objects/")
def predict_objects(to_be_predicted: ToPredict):
    """
    Predict the object in a given string (expecting a single sentence usually).
    """
    if to_be_predicted.sentence:
        sentence = to_be_predicted.sentence
        # SPaR doesn't handle all uppercase sentences well, which the OCR system sometimes outputs
        if sentence.isupper():
            sentence = sentence.lower()

        # prepare instance and run model on single instance
        docid = ''  # ToDo - add doc_id during pre_processing?
        token_list = spar_predictor.predictor._dataset_reader.tokenizer.tokenize(sentence)

        # truncating the input to SPaR.txt to maximum 512 tokens
        token_length = len(token_list)
        if token_length > 512:
            token_list = token_list[:511] + [token_list[-1]]
            token_length = 512

        instance = spar_predictor.predictor._dataset_reader.text_to_instance(
            docid, sentence, token_list, spar_predictor.predictor._dataset_reader._token_indexer
        )
        res = predictor.predict_instance(instance)
        printable_result, spans_token_length = su.parse_spar_output(res, ['obj'])
        return {
            "prediction": printable_result,
            "num_input_tokens": token_length,
            "num_output_tokens": spans_token_length
        }
    # If the input is None, or too long, return an empty list of objects
    return {
            "prediction": {'obj': []},
            "num_input_tokens": 0,
            "num_output_tokens": 0
        }

