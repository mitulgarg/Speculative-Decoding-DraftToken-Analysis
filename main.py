import argparse
from evaluator import SpeculativeDecoderEvaluator
from plotter import ResultPlotter

def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding analysis.")
    parser.add_argument(
        "--model", type=str, default="phi-3-mini-4k-instruct",
        help="Base model name (without -8bit/-4bit suffixes)"
    )
    args = parser.parse_args()

    num_draft_tokens_list = [0, 1, 2, 3, 4]

    evaluator = SpeculativeDecoderEvaluator(base_model=args.model)
    results = evaluator.evaluate(num_draft_tokens_list)

    plotter = ResultPlotter(num_draft_tokens_list, results)
    plotter.plot_all()

if __name__ == "__main__":
    main()
