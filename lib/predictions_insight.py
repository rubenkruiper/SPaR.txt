import json
from collections import Counter
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


    def collect_mwes(self, predictions_fp):

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

            # not sure if this is necessary; doesn't seem to be
            predicted_tags = [t for m, t in zip(mask, tag_list) if m]

            all_mwes["objects"] += self.get_mwes(token_list, predicted_tags, 'obj')
            all_mwes["actions"] += self.get_mwes(token_list, predicted_tags, 'act')

        return all_mwes

    def get_mwes(self, word_list, tag_list, type):
        """
        Collects the MWEs found by a trained tagger, handles discontiguous spans.
        """

        mwes = []
        current_mwe = []
        previous_head = []
        just_removed = []

        for idx, t in enumerate(tag_list):
            if t[-3:] != type:
                # We're only interested in collecting a specific type of MWE for now, e.g., objects / actions
                if current_mwe != []:
                    # store mwe
                    mwes.append(current_mwe)
                    # ToDo - I do not always want to store current_mwe into previous head
                    if just_removed == []:
                        # als we net een BD hebben gehad, dan moet previous_head niet veranderen...
                        previous_head = current_mwe
                current_mwe = []
                continue

            if t[:2] == 'BH':
                just_removed = []

                if current_mwe == []:
                    # Start collecting a new mwe
                    current_mwe.append(word_list[idx])
                else:
                    # Store old mwe and start collecting a new mwe
                    mwes.append(current_mwe)
                    previous_head = current_mwe
                    current_mwe = [word_list[idx]]
            elif t[:2] == 'IH':
                # Continue collecting the same object
                current_mwe.append(word_list[idx])
            elif t[:2] == 'BD':
                # Remove singleton previous_head from mwes once
                if just_removed != previous_head:
                    mwes.reverse()
                    mwes.remove(previous_head)
                    mwes.reverse()
                    just_removed = previous_head

                # Append token.text to previous_mwe
                current_mwe = previous_head + [word_list[idx]]
            elif t[:2] == 'ID':
                current_mwe.append(word_list[idx])

        return mwes

    def mwe_list_to_string(self, mwe_list):
        text = ''
        for w in mwe_list:
            if w.startswith('##'):
                text += w[2:]
            else:
                text += " " + w
        return text

    def print_mwe_stats(self, all_mwes):

        for k in all_mwes:
            print("MWE type: {}".format(k))
            counter = Counter()
            for mwe in all_mwes[k]:
                counter[self.mwe_list_to_string(mwe)] += 1

            print("Top 20 counts: \n{}".format(counter.most_common(20)))






if __name__ == "__main__":

    my_pred_obj = PredictionInsight()
    mwe_dict = my_pred_obj.collect_mwes('predictions/debug_output.json')
    my_pred_obj.print_mwe_stats(mwe_dict)


