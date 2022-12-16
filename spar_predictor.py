import os
import sys
import torch
import json

from allennlp.predictors.predictor import Predictor as AllenNLPPredictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules
from allennlp.commands import main


import_module_and_submodules("lib")


class SparPredictor:
    def __init__(self, default_path="/trained_models/debugger_train/model.tar.gz"):

        if not os.path.exists(default_path):
            # todo if the model doesn't exist, train the default model.
            print("No trained model found, creating one at{}.".format(default_path),
                  "\nIf a GPU is available, this will take several minutes. "
                  "If no GPU is available, this will take 20+ minutes.")

            default_config_file = "/data/spar_data/experiments/attention_tagger.json"
            serialization_dir = default_path
            # Assemble the command into sys.argv
            sys.argv = [
                "allennlp",  # command name, not used by main
                "train",
                default_config_file,
                "-s", serialization_dir,
                "--include-package", "lib"
            ]

            # Simple overrides to train on CPU if no GPU available, with a possibly smaller batch_size
            if not torch.cuda.is_available():
                overrides = json.dumps({"trainer": {"cuda_device": -1}})  # ,
                # "data_loader": {"batch_sampler": {"batch_size": 16}}})
                sys.argv += ["-o", overrides]

            main()

        default_archive = load_archive(default_path) # , overrides=model_overrides
        self.predictor = AllenNLPPredictor.from_archive(default_archive, predictor_name="span_tagger")



