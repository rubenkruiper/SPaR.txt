import concurrent.futures

from typing import List, Union
from textblob import TextBlob
from threading import current_thread

import spar_serving_utils as su
from spar_predictor import SparPredictor


class SparInstance:
    def __init__(self):
        """
        These should automatically run on your Nvidia GPU if available
        """
        self.sp = SparPredictor()

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
            results = self.sp.predict_batch_instance(instances)

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
        instances = spartxt.prepare_instances(batch_of_sentences)
        prediction_per_sentence_list = spartxt.call(instances)
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
        text_and_predictions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_num_cpu_threads) as executor:
            futures = [executor.submit(self.process_text, texts[idx]) for idx in range(len(texts))]
            text_and_predictions += [sent_pred_tuple for f in futures for sent_pred_tuple in f.result()]
        return text_and_predictions
