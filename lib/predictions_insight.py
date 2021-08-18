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


    def count_spans(self, predictions_fp):
        """
        Count and print the number of different span types and lengths.
        """

        # read predictions from output file
        with open(predictions_fp, 'r') as f:
            predictions_list = [json.loads(jsonline) for jsonline in f.readlines()]

        discontiguous_obj_count = 0
        discontiguous_act_count = 0
        all_spans = {"objects": [],
                     "actions": [],
                     "functional": [],
                     "discourse": []}

        for prediction in predictions_list:
            mask = prediction['mask']
            tag_list = prediction['tags']
            token_list = prediction['words']
            sentence = prediction["sentence"]
            doc_id = prediction["doc_id"]

            # not sure if this is necessary; doesn't seem to be
            predicted_tags = [t for m, t in zip(mask, tag_list) if m]
            mwes, discontiguous_count = self.get_mwes(token_list, predicted_tags, 'obj')
            discontiguous_obj_count += discontiguous_count
            all_spans["objects"] += mwes

            mwes, discontiguous_count = self.get_mwes(token_list, predicted_tags, 'act')
            discontiguous_act_count += discontiguous_count
            all_spans["actions"] += mwes

        print("Found {} discontiguous spans of type 'object'".format(discontiguous_obj_count))
        print("Found {} discontiguous spans of type 'action'".format(discontiguous_act_count))

        return all_spans

    def get_mwes(self, word_list, tag_list, mwe_type):
        """
        Collects the MWEs found by a trained tagger, handles discontiguous spans.
        """

        mwes = []
        current_mwe = []
        previous_head = []
        just_removed = []
        number_of_discontiguous_mwes = 0

        for idx, t in enumerate(tag_list):
            t_head, t_type = t.split('-')
            if t_type != mwe_type:
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

            if t_head == 'BH':
                just_removed = []

                if current_mwe == []:
                    # Start collecting a new mwe
                    current_mwe.append(word_list[idx])
                else:
                    # Store old mwe and start collecting a new mwe
                    mwes.append(current_mwe)
                    previous_head = current_mwe
                    current_mwe = [word_list[idx]]
            elif t_head == 'IH':
                # Continue collecting the same object
                current_mwe.append(word_list[idx])
            elif t_head == 'BD':
                number_of_discontiguous_mwes += 1
                # Remove singleton previous_head from mwes once
                if just_removed != previous_head and previous_head in mwes:
                    mwes.reverse()
                    mwes.remove(previous_head)
                    mwes.reverse()
                    just_removed = previous_head

                # Append token.text to previous_mwe
                current_mwe = previous_head + [word_list[idx]]
            elif t_head == 'ID':
                current_mwe.append(word_list[idx])

        return mwes, number_of_discontiguous_mwes

    def mwe_list_to_string(self, mwe_list):
        text = ''
        for w in mwe_list:
            if w.startswith('##'):
                text += w[2:]
            else:
                text += " " + w

        text = text[1:]
        if text.startswith('the ') or text.startswith('The '):
            text = text[4:]
        elif text.startswith('a ') or text.startswith('A '):
            text = text[2:]
        elif text.startswith('an ') or text.startswith('An '):
            text = text[3:]
        return text

    def count_mwes(self, all_mwes):
        counter_list = []
        for k in all_mwes:
            counter = Counter()
            for mwe in all_mwes[k]:
                counter[self.mwe_list_to_string(mwe)] += 1
            counter_list.append(counter)
        return counter_list


if __name__ == "__main__":

    my_pred_obj = PredictionInsight()
    mwe_dict = my_pred_obj.count_spans('predictions/all_sentence_predictions.json')
    # mwe_dict = my_pred_obj.collect_mwes('predictions/debug_output.json')
    mwe_counter_lists = my_pred_obj.count_mwes(mwe_dict)
    for counter in mwe_counter_lists:
        print("Top 20 counts: \n{}".format(counter.most_common(20)))
    print("Need to determine what I want to do/see with the counts")


