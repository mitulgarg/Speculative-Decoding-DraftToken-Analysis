from evaluator import SpeculativeDecoderEvaluator
from plotter import ResultPlotter

def main():
    num_draft_tokens_list = [0, 1, 2, 3, 4]
    evaluator = SpeculativeDecoderEvaluator()
    results = evaluator.evaluate(num_draft_tokens_list)

    plotter = ResultPlotter(num_draft_tokens_list, results)
    plotter.plot_all()

if __name__ == "__main__":
    main()
