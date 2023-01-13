import os
import sys
import torch
import json
from pathlib import Path

from allennlp.predictors.predictor import Predictor as AllenNLPPredictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules
from allennlp.commands import main
from spar_serving_utils import *


cwd = Path.cwd()
sys.path.insert(0, cwd.joinpath("SPaR.txt"))
import_module_and_submodules("lib")


class SparPredictor:
    """
    Because of our dumb project name that contains a '.', import as follows:
    ```
    import imp
    with open('SPaR.txt/spar_predictor.py', 'rb') as fp:
        spar_predictor = imp.load_module(
            'spar_predictor', fp, 'SPaR.txt.spar_predictor.py',
            ('.py', 'rb', imp.PY_SOURCE)
        )
    ```
    """
    def __init__(self,
                 default_serialization_dir: Path = Path.cwd().joinpath("SPaR.txt", "trained_models", "debugger_train"),
                 default_config_file: Path = Path.cwd().joinpath("SPaR.txt", "experiments", "span_predictor_tagger.json")):

        if not default_serialization_dir.joinpath("model.tar.gz").exists():
            # If the model doesn't exist, train a model and save it to the specified directory.
            print("No trained model found, creating one at {}.".format(default_serialization_dir),
                  "\nIf a GPU is available, this will take several minutes. "
                  "If no GPU is available, this will take 20+ minutes.")

            if not default_config_file.exists():
                print(f"Make sure a configuration file exists at the location you specified: {default_config_file}")

            # Assemble the command into sys.argv
            sys.argv = [
                "allennlp",  # command name, not used by main
                "train", default_config_file,
                "-s", default_serialization_dir,
                "--include-package", "lib"
            ]

            # Simple overrides to train on CPU if no GPU available, with a possibly smaller batch_size
            if not torch.cuda.is_available():
                overrides = json.dumps({"trainer": {"cuda_device": -1}})  # ,
                # "data_loader": {"batch_sampler": {"batch_size": 16}}})
                sys.argv += ["-o", overrides]

            main()

        spartxt_archive = load_archive(default_serialization_dir.joinpath("model.tar.gz"))  # ,overrides=model_overrides
        self.predictor = AllenNLPPredictor.from_archive(spartxt_archive, predictor_name="span_tagger")

    def parse_output(self, prediction, span_types=['obj', 'act', 'func', 'dis']):
        """
        SPaR.txt outputs are formatted following the default AllenNLP json structure. This function grabs
        the spans from the output in text format.
        """
        return parse_spar_output(prediction, span_types)