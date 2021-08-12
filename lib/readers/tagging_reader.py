import glob, json
from lib.readers.reader_utils.my_read_utils import *

from typing import Dict, List, Optional, Sequence, Iterable, Any
import logging
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Token, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("tag_reader")
class TagReader(DatasetReader):
    """
    Reads a pair of brat text and annotation files, with the following structure:

    ```
    brat: sentence string\n
    annotations:
    ```
    Objects (BH-obj, IH-obj, BD-obj, ID-obj)
    Actions (BH-act, IH-act, BD-act, ID-act)
    Functional and Discourse (BH-func, IH-func, BH-dis, IH-dis)

    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_instances: Optional[int] = None,
                 bert_model_name: str = None,
                 **kwargs) -> None:
        """
        https://github.com/allenai/allennlp-models/tree/845fe4cc5896d5492ed16b4299da5ff69ddb99ed/allennlp_models/structured_prediction
        """
        super().__init__(**kwargs)
        # make sure max instances is a positive int
        if max_instances is not None and max_instances < 0:
            raise ValueError("If specified, max_instances should be a positive int")

        self.max_instances = max_instances

        # set tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif bert_model_name is not None:
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self._tokenizer = None
            self.lowercase_input = False

        # set token_indexer
        if token_indexers is not None:
            self._token_indexer = token_indexers
        elif bert_model_name is not None:
            from allennlp.data.token_indexers import PretrainedTransformerIndexer
            self._token_indexer = {"tokens": PretrainedTransformerIndexer(bert_model_name)}
        else:
            self._token_indexer = {"tokens": TokenIndexer()}

    @overrides
    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:

        text_files = sorted(glob.glob(file_path + "*.txt"))
        ann_files = sorted(glob.glob(file_path + "*.ann"))

        if ann_files != []:
            for text_file, ann_file in zip(text_files, ann_files):
                doc_name = text_file.rsplit('/', 1)[1].rsplit('.', 1)[0]

                with open(text_file, "r") as tf:
                    original_sentence = tf.read()

                token_list = self.tokenizer.tokenize(original_sentence)
                tag_list = get_annotations_from_ann_file(ann_file, original_sentence, token_list)
                yield self.text_to_instance(doc_name, original_sentence, token_list, self._token_indexer, tag_list)
        else:
            # .txt files as input without annotations
            for text_file in text_files:
                doc_name = text_file.rsplit('/', 1)[1].rsplit('.', 1)[0]
                with open(text_file, "r") as tf:
                    original_sentence = tf.read()
                token_list = self.bert_tokenizer.tokenize(original_sentence)
                yield self.text_to_instance(doc_name, original_sentence, token_list, self._token_indexer)

    def text_to_instance(  # type: ignore
        self,
        doc_name: str,
        sentence: str,
        tokens: List[Token],
        token_indexer: Dict[str, TokenIndexer],
        tag_list: List[str] = None
    ) -> Instance:
        """
        Should describe this
        """
        token_sequence = TextField(tokens, token_indexer)
        label_field = SequenceLabelField(tag_list, token_sequence, label_namespace="labels")

        # Store the full annotation information as metadata
        metadata: Dict[str, Any] = {"original_text": sentence,
                                    "words": [x.text for x in tokens],
                                    "token_len": len(tokens),
                                    "gold_labels": tag_list,
                                    "doc_id": doc_name}

        instance_fields: Dict[str, Field] = {"tokens": token_sequence,
                                             "gold_labels": label_field,
                                             "metadata": MetadataField(metadata)}

        return Instance(instance_fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self._token_indexer  # type: ignore
