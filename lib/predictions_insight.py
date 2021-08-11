import json
from typing import List, Dict, Any
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from lib.readers.reader_utils.my_read_utils import *

class PredictionInsight():
    """ Grab objects from an predictions output_file """

    def __init__(self,
                 bert_model_name: str = "SpanBERT/spanbert-base-cased"):
        self.predictions_input = ''

        if bert_model_name is not None:
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name


    def run(self, predictions_fp):

        # read predictions from output file
        with open(predictions_fp, 'r') as f:
            predictions_list = [json.loads(jsonline) for jsonline in f.readlines()]

        all_mwes = {"objects": [],
                    "actions": []}
        for prediction in predictions_list:
            mask = prediction['mask']
            tag_list = prediction['tags']
            token_list = prediction['words']
            sentence = prediction["sentence"]
            doc_id = prediction["doc_id"]

            predicted_tags = [t for m, t in zip(mask, tag_list) if m]

            all_mwes["objects"] += self.get_mwes(token_list, predicted_tags, 'obj')

    def get_mwes(self, word_list, tag_list, type):

        mwes = []
        current_mwe = []
        previous_mwe = []
        for idx, t in tag_list:
            if t[:-3] != type:
                mwes.append(current_mwe)
                continue
            # ToDo
            #  - keep track of previous_mwe until BH type
            #  - if a BH of type was found, it means the end of a discontiguous part
            #       - at the end of current_mwe update previous_mwe (keep track with a boolean?)
            #       - at the end of discontiguous parts  add both current_mwe to previous_mwe
            #  - if

            if t[:2] == 'BD':
                # Append token.text to previous_mwe
                previous_mwe.append(word_list[idx])


            elif t[:2] == 'ID':
                print('do somtihg')



if __name__ == "__main__":

    my_pred_obj = PredictionInsight()
    my_pred_obj.run('predictions/debug_output.json')


