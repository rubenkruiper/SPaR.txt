import sys
import json
import torch
import concurrent.futures
from typing import List, Union
from pathlib import Path
from textblob import TextBlob
from threading import current_thread
import spar_serving_utils as su

from allennlp.commands import main
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor as AllenNLPPredictor
import_module_and_submodules("spar_lib")


class SparPredictor:
    """
    """
    def __init__(self,
                 serialization_dir: Path = Path.cwd().joinpath("trained_models", "debugger_train"),
                 config_fp: Path = Path.cwd().joinpath("experiments", "docker_conf.json")):

        if not serialization_dir.joinpath("model.tar.gz").exists():
            # If the model doesn't exist, train a model and save it to the specified directory.
            print("No trained model found, creating one at {}.".format(serialization_dir),
                  "\nIf a GPU is available, this will take several minutes. "
                  "If no GPU is available, this will take 20+ minutes.")

            if not config_fp.exists():
                print(f"Make sure a configuration file exists at the location you specified: {config_fp}")

            # Assemble the command into sys.argv
            sys.argv = [
                "allennlp",  # command name, not used by main
                "train", str(config_fp),
                "-s", str(serialization_dir),
                "--include-package", "spar_lib"
            ]

            # Simple overrides to train on CPU if no GPU available, with a possibly smaller batch_size
            if not torch.cuda.is_available():
                overrides = json.dumps({"trainer": {"cuda_device": -1}})  # ,
                # "data_loader": {"batch_sampler": {"batch_size": 16}}})
                sys.argv += ["-o", overrides]

            main()

        spartxt_archive = load_archive(serialization_dir.joinpath("model.tar.gz"))  # ,overrides=model_overrides
        self.predictor = AllenNLPPredictor.from_archive(spartxt_archive, predictor_name="span_tagger")

    def parse_output(self, prediction, span_types=['obj', 'act', 'func', 'dis']):
        """
        SPaR.txt outputs are formatted following the default AllenNLP json structure. This function grabs
        the spans from the output in text format.
        """
        return su.parse_spar_output(prediction, span_types)


class SparInstance:
    def __init__(self):
        """
        These should automatically run on your Nvidia GPU if available
        """
        Path.cwd().joinpath("SPaR.txt", "trained_models", "debugger_train")
        default_experiment_path = Path.cwd().joinpath("experiments", "docker_conf.json")
        default_output_path = Path.cwd().joinpath("trained_models", "debugger_train", "model.tar.gz")
        self.sp = SparPredictor(default_output_path, default_experiment_path)

    def prepare_instances(self, sentences: List[str]):
        instances = []
        for idx, sentence in enumerate(sentences):
            # SPaR doesn't handle all uppercase sentences well, which the OCR system sometimes outputs
            sentence = sentence.lower() if sentence.isupper() else sentence

            # prepare instance and run model on single instance
            docid = str(idx)  # this is just a placeholder really
            token_list = self.sp.predictor._dataset_reader.tokenizer.tokenize(sentence)

            # truncating the input to SPaR.txt to maximum 512 tokens
            token_length = len(token_list)
            if token_length > 512:
                token_list = token_list[:511] + [token_list[-1]]

            instances.append(self.sp.predictor._dataset_reader.text_to_instance(
                docid, sentence, token_list, self.sp.predictor._dataset_reader._token_indexer
            ))
        return instances

    def call(self, input_str: Union[List[str], str]):
        input_str = [input_str] if type(input_str) == str else input_str
        if not input_str:
            # If the input is None, or too long, return an list with and empty dict
            return [{'obj': [], 'dis': [], 'func': [], 'act': []}]
        else:
            instances = self.prepare_instances(input_str)
            results = self.sp.predictor.predict_batch_instance(instances)

            predictions_per_sentence_list = []
            for res in results:
                printable_result, _ = su.parse_spar_output(res, ['obj', 'dis', 'func', 'act'])
                predictions_per_sentence_list.append(printable_result)
            return predictions_per_sentence_list


class TermExtractor:
    def __init__(self, split_length=300, max_num_cpu_threads=4):
        """
        Initialise `max_num_cpu_threads` separate SPaR.txt predictors
        """
        self.split_length = split_length  # in number of tokens
        self.max_num_cpu_threads = max_num_cpu_threads
        self.PREDICTORS = []
        for i in range(max_num_cpu_threads + 1):
            self.PREDICTORS.append(SparInstance())

    def process_sentence_batch(self, batch_of_sentences: List[str]):
        """
        """
        if not batch_of_sentences:
            return []

        predictor_to_use = int(current_thread().name.rsplit('_', 1)[1])
        spartxt = self.PREDICTORS[predictor_to_use]
        prediction_per_sentence_list = spartxt.call(batch_of_sentences)
        return prediction_per_sentence_list

    def split_into_sentences(self, to_be_split: Union[str, List[str]]) -> List[str]:
        """
        """
        if type(to_be_split) == str:
            if ';' in to_be_split:
                # some of the WikiData definitions contain multiple definitions separated by ';'
                to_be_split = to_be_split.split(';')
            else:
                to_be_split = [to_be_split]

        sentences = []
        for text in to_be_split:
            for part in text.split('\n'):
                # split into sentences using PunktSentTokenizer (TextBlob implements NLTK's version under the hood)
                sentences += [str(s) for s in TextBlob(part.strip()).sentences if len(str(s)) > 10]
        return sentences

    def process_text(self, text: str):
        """
        Process a text, which will be split into sentences and then for each sentence a prediction will be made of the
        spans that occur.

        :return sentences: List of sentences that were found in the text.
        :return flat_predictions: Flattened list of predictions
        """
        sentences = self.split_into_sentences(text)
        prediction_dict_per_sentence_list = self.process_sentence_batch(sentences)
        return sentences, prediction_dict_per_sentence_list

    def process_texts(self, texts: List[str]):
        """
        """
        sentences = []
        predictions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_num_cpu_threads) as executor:
            futures = [executor.submit(self.process_text, texts[idx]) for idx in range(len(texts))]
            sent_pred_tuples = [sent_pred_tuple for f in futures for sent_pred_tuple in f.result()]
            sents, preds = sent_pred_tuples
            sentences += sents
            predictions += preds
        return sentences, predictions
