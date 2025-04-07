from speculative_decoding_metrics.main import run_evaluation

run_evaluation(
    base_model="phi-3-mini-4k-instruct",  #Use a model that'll run on your local
    main_quant="8bit",          #use q8 instead of "8bit" based on HuggingFace Repo name 
    draft_quant="4bit",         #use q4 instead of "4bit" based on HuggingFace Repo name
    prompt="How do LLMs work?",
    max_tokens=64,              #Tweak max tokens per output
    num_draft_tokens_list=[0, 1, 2, 3, 4]
)
