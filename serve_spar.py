"""
Interactive terminal demo for SPaR.txt.

Usage
-----
    python serve_spar.py
    python serve_spar.py --model-dir trained_models/

Enter a sentence at the prompt and see extracted spans.  Type 'quit' to exit.
"""
import argparse
import time
from pathlib import Path

from spar_api_utils import SparPredictor


def main():
    p = argparse.ArgumentParser(description="SPaR.txt interactive terminal demo")
    p.add_argument("--model-dir",  default="trained_models/",
                   help="Directory containing model.pt")
    p.add_argument("--bert-model", default="bert-base-cased")
    args = p.parse_args()

    predictor = SparPredictor(
        model_dir=Path(args.model_dir),
        bert_model=args.bert_model,
    )

    print("NOTE:\tTo stop running, simply enter 'quit' as input.")
    print("-------------------------------------------------------")

    while True:
        user_query = input("Enter text to be parsed: ")
        if user_query.strip().lower() == "quit":
            break

        start = time.time()
        results = predictor.predict_sentences([user_query])
        spans, _ = predictor.parse_output(results[0])
        print(spans)
        print(f"Parsing took {time.time() - start:.3f}s")


if __name__ == "__main__":
    main()
