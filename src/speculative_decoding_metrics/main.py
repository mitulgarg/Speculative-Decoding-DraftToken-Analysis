import argparse
from speculative_decoding_metrics.evaluator import SpeculativeDecoderEvaluator
from speculative_decoding_metrics.plotter import ResultPlotter

def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding analysis.")
    parser.add_argument(
        "--model", type=str, default="phi-3-mini-4k-instruct",
        help="Base model name (without -8bit/-4bit suffixes)"
    )
    parser.add_argument(
        "--prompt", type=str, default="Write a story about Einstein.",
        help="Prompt to evaluate speculative decoding on"
    )
    args = parser.parse_args()

    num_draft_tokens_list = [0, 1, 2, 3, 4, 5]

    evaluator = SpeculativeDecoderEvaluator(base_model=args.model, prompt_text=args.prompt)
    results = evaluator.evaluate(num_draft_tokens_list)

    plotter = ResultPlotter(num_draft_tokens_list, results)
    plotter.plot_all()

if __name__ == "__main__":
    main()
