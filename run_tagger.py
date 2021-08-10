from argparse import ArgumentParser
import json
import shutil
import sys
import torch

from allennlp.commands import main


if __name__ == "__main__":
    argparse = ArgumentParser(description='lib semantic chunking')
    argparse.add_argument('-c', "--config_file", type=str, default="experiments/simplest_tagger.json")
    argparse.add_argument('-m', "--model_path", default="trained_models/debugger_train")
    argparse.add_argument('-i', "--input_file_path", default="")
    argparse.add_argument('-o', "--output_file_path", default="")
    argparse.add_argument("--batchsize", type=int, default=2)
    argparse.add_argument("--predict", action='store_true')
    argparse.add_argument("--evaluate", action='store_true')

    args = argparse.parse_args()
    config_file = args.config_file
    serialization_dir = args.model_path
    predictions_output = args.output_file_path

    if args.predict:
        # overrides = json.dumps({})

        sys.argv = [
            "allennlp",  # command name, not used by main
            "predict",
            serialization_dir,
            args.input_file_path,
            "--predictor", "span_tagger",
            "--include-package", "lib",
            "--use-dataset-reader",
            "--batch-size", str(args.batchsize),
            # "-o", overrides,
        ]
        # Add output-file if specified
        if predictions_output != "":
            sys.argv += ["--output-file", predictions_output]

    elif args.evaluate:
        print("not implemented yet")
        # ToDo - need to implement a pipeline: 1st) predict 2nd) evaluate predicted output
        #  issue: can't figure out how to use/set the pretrained transformer tokenizer, thus running into vocab issues
    #     overrides = json.dumps({"data_loader": {"tokenizer": {
    #                                               "type": "pretrained_transformer",
    #                                               "model_name": "SpanBERT/spanbert-base-cased",
    #                                               "add_special_tokens": True
    #                                             }, "batch_sampler": {"batch_size": args.batchsize}}})
    #
    #     sys.argv = [
    #         "allennlp",  # command name, not used by main
    #         "evaluate",
    #         serialization_dir,
    #         args.input_file_path,
    #         "--include-package", "lib",
    #         "-o", overrides
    #     ]
    #     # Add output-file for predictions if specified
    #     if predictions_output != "":
    #         sys.argv += ["--predictions-output-file", predictions_output]
    #
    else:
        # Training will fail if the serialization directory already
        # has stuff in it. If you are running the same training loop
        # over and over again for debugging purposes, it will.
        # Hence we wipe it out in advance.
        # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
        shutil.rmtree(serialization_dir, ignore_errors=True)

        # Assemble the command into sys.argv
        sys.argv = [
            "allennlp",  # command name, not used by main
            "train",
            config_file,
            "-s", serialization_dir,
            "--include-package", "lib"
        ]

        # Use overrides to train on CPU.
        if not torch.cuda.is_available():
            overrides = json.dumps({"trainer": {"cuda_device": -1},
                                    "data_loader": {"batch_sampler": {"batch_size": 8}}})
            sys.argv += ["-o", overrides]

    main()