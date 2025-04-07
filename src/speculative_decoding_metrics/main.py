from speculative_decoding_metrics.evaluator import SpeculativeDecoderEvaluator
from speculative_decoding_metrics.plotter import ResultPlotter

def run_evaluation(
    base_model="phi-3-mini-4k-instruct",
    prompt="Write a story about Einstein.",
    main_quant="8bit",
    draft_quant="4bit",
    num_draft_tokens_list=None,
    max_tokens=32,
    plot=True,
):
    if num_draft_tokens_list is None:
        num_draft_tokens_list = [0, 1, 2, 3, 4, 5]

    evaluator = SpeculativeDecoderEvaluator(
        base_model=base_model,
        prompt_text=prompt,
        main_quant=main_quant,
        draft_quant=draft_quant
    )
    results = evaluator.evaluate(num_draft_tokens_list)

    if plot:
        plotter = ResultPlotter(num_draft_tokens_list, results)
        plotter.plot_all()

    return results