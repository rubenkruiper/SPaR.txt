import json, os, shutil, sys, torch
from argparse import ArgumentParser
from allennlp.commands import main
from lib.evaluation_script import SimpleEvaluator


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

        sys.argv = [
            "allennlp",  # command name, not used by main
            "predict",
            serialization_dir,
            args.input_file_path,
            "--predictor", "span_tagger",
            "--include-package", "lib",
            "--use-dataset-reader",
            "--batch-size", str(args.batchsize)
        ]
        # Add output-file if specified
        if predictions_output != "":
            sys.argv += ["--output-file", predictions_output]
        # Actually run prediction
        main()

    elif args.evaluate:
        # ToDo - need to implement a pipeline: 1st) predict 2nd) evaluate predicted output
        #  issue: can't figure out how to use/set the pretrained transformer tokenizer, thus running into vocab issues
        if predictions_output == "":
            predictions_output = 'predictions/debug_output.json'

        if os.path.exists(predictions_output):
            predictions_file_exists = True
        else:
            predictions_file_exists = False

        if not predictions_file_exists:
            # First predict ~ default output is location is 'predictions/debug_output.json'
            sys.argv = [
                "allennlp",  # command name, not used by main
                "predict",
                serialization_dir,
                args.input_file_path,
                "--predictor", "span_tagger",
                "--output-file", predictions_output,
                "--include-package", "lib",
                "--use-dataset-reader",
                "--batch-size", str(args.batchsize)
            ]
            main()

        # Then simply run a script to evaluate and print metrics for the output file
        evaluator = SimpleEvaluator(predictions_output, args.input_file_path)
        evaluator.evaluate()


    else:
        # Training will fail if the serialization directory already
        # has stuff in it. To re-use the same directory for debugging
        # we clear the contents in advance.
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

        # actually run the training
        main()
