import json, glob
from typing import List, Dict, Any
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from lib.readers.reader_utils.my_read_utils import *
from sklearn.metrics import f1_score, classification_report


class SimpleEvaluator():
    """ Ugly temporary solution to evaluate a model; compares the predicted output against the gold input """

    def __init__(self,
                 predictions_fp,
                 gold_fp,
                 bert_model_name: str = "SpanBERT/spanbert-base-cased"):
        self.predictions_input = predictions_fp
        self.gold_input = gold_fp

        if bert_model_name is not None:
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name

    def read_gold(self):
        text_files = sorted(glob.glob(self.gold_input + "*.txt"))
        ann_files = sorted(glob.glob(self.gold_input + "*.ann"))

        gold_annotations = []
        for text_file, ann_file in zip(text_files, ann_files):
            doc_name = text_file.rsplit('/', 1)[1].rsplit('.', 1)[0]

            with open(text_file, "r") as tf:
                original_sentence = tf.read()

            token_list = self.tokenizer.tokenize(original_sentence)
            tag_list = get_annotations_from_ann_file(ann_file, original_sentence, token_list)
            gold_annotations.append({"sent_id": doc_name, "sentence": original_sentence,
                                     "token_list": [t.text for t in token_list], "tag_list": tag_list})
        return gold_annotations


    def evaluate(self):
        # vocab = Vocabulary.from_files("data/vocab_discontiguous_tags/")
        # measure = SimpleF1Measure()

        # read gold data from 'predictions_input'
        gold_instances = self.read_gold()

        # read predictions from output file
        with open(self.predictions_input, 'r') as f:
            predictions_list = [json.loads(jsonline) for jsonline in f.readlines()]

        # sklearn test
        y_true = []
        y_pred = []

        for prediction in predictions_list:
            mask = prediction['mask']
            tag_list = prediction['tags']
            token_list = prediction['words']

            for gold in gold_instances:
                # should be same order for predict, at least only 1 match
                if gold['token_list'] == token_list:
                    # compare tag_lists
                    predicted_tags = [t for m, t in zip(mask, tag_list) if m]
                    # measure(predicted_tags, gold["tag_list"])
                    y_true += gold["tag_list"][1:-1]
                    y_pred += predicted_tags[1:-1]

        # p, r, f = measure.get_metric()
        # print("Overall precision: {:.4f}, recall: {:.4f}, F1: {:.4f}".format(p, r, f))

        # check using sklearn
        # print("sklearn F1: {}".format(f1_score(y_true, y_pred, average='micro')))
        print(classification_report(y_true, y_pred, digits=4))


# class SimpleF1Measure():
#     def __init__(self) -> None:
#         self.span_type_acc = F1Scorer(F1Scorer.span_type_accuracy)
#
#     def __call__(self,  # type: ignore
#                  predicted_labels: List[str],
#                  gold_labels: List[str]):
#         """
#         """
#
#         self.span_type_acc.batch_gold_labels = len([x for x in gold_labels if x != "PD-pad"])
#         self.span_type_acc.update(gold_labels, predicted_labels)
#
#     def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
#         span_f1 = self.span_type_acc.get_prf()
#         if reset:
#             self.reset()
#         return span_f1
#
#     def reset(self):
#         self.span_type_acc = F1Scorer(F1Scorer.span_type_accuracy)
#
# #### F1 scorer
# class F1Scorer:
#     def __init__(self, metric):
#         ### initialise the object with 0 scores
#         self.precision_numerator = 0
#         self.precision_denominator = 0
#         self.recall_numerator = 0
#         self.recall_denominator = 0
#         self.metric = metric
#         self.batch_gold_labels = 0
#
#     def update(self, gold_batch, predicted_batch):
#         p_num, p_den, r_num, r_den = self.metric(self, gold_batch, predicted_batch)
#
#         self.precision_numerator += p_num
#         self.precision_denominator += p_den
#         self.recall_numerator += r_num
#         self.recall_denominator += r_den
#
#     def get_f1(self):
#         precision = 0 if self.precision_denominator == 0 else \
#             self.precision_numerator / float(self.precision_denominator)
#         recall = 0 if self.recall_denominator == 0 else \
#             self.recall_numerator / float(self.recall_denominator)
#         return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
#
#     def get_recall(self):
#         if self.recall_numerator == 0:
#             return 0
#         else:
#             return self.recall_numerator / float(self.recall_denominator)
#
#     def get_precision(self):
#         if self.precision_numerator == 0:
#             return 0
#         else:
#             return self.precision_numerator / float(self.precision_denominator)
#
#     def get_prf(self):
#         return self.get_precision(), self.get_recall(), self.get_f1()
#
#     def span_type_accuracy(self, gold_labels, pred_labels):
#         """
#         Accuracy based on correctly predicting correct values of a mask, including only partial mask matches.
#         ◮ Precision: (# labels correctly assigned ~ p_num) TP / (# labels assigned ~ p_den) TP+FP
#         ◮ Recall: (# labels correctly assigned ~ r_num) TP / (total # of labels ~ r_den) TP+FN
#         """
#         p_num, p_den, r_num = 0, 0, 0
#         # every gold_span that should be 1 (TP + FN)
#         r_den = self.batch_gold_labels
#
#         for li, label_value in enumerate(pred_labels[1:-1]):
#             # ignore [CLS] and [SEP]
#             # if label_value != "PD-pad":  # ignore padding
#             p_den += 1              # all predicted mask labels, both true and false (TP+FP)
#             if label_value == gold_labels[li + 1]:
#                 p_num += 1          # TP
#                 r_num += 1          # TP
#
#         return p_num, p_den, r_num, r_den


if __name__ == "__main__":
    evaluator = SimpleEvaluator("../predictions/test_predictions.json",
                                "../data/test/",
                                "bert-base-cased")
    evaluator.evaluate()

