import time
import matplotlib.pyplot as plt
import mlx_lm

# Define prompt
prompt_text = "Write a story about Einstein."
messages = [{"role": "user", "content": prompt_text}]

# Settings
# Only use quantizations that exist on Hugging Face for this model
quantizations = ["4bit", "8bit", "bf16"]

num_draft_tokens_list = [0, 2, 4, 6, 8, 10, 12]
draft_model_path = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
max_gen_tokens = 256

results = {}  # Stores performance + accuracy
reference_outputs = {}  # Autoregressive outputs

# Tokenizer used across quantizations
_, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-8bit")

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

def get_tokens(output_str):
    return tokenizer.encode(output_str)

for q in quantizations:
    print(f"\n🔍 Testing quantization: {q}")
    
    if q == "bf16":
        model_path = "mlx-community/Mistral-7B-Instruct-v0.3"
    else:
        model_path = f"mlx-community/Mistral-7B-Instruct-v0.3-{q}"
    
    main_model, _ = mlx_lm.load(model_path)
    results[q] = []

    ref_output = None

    for n_draft in num_draft_tokens_list:
        print(f"  → num_draft_tokens = {n_draft}")
        
        draft_model = None
        if n_draft > 0:
            draft_model, _ = mlx_lm.load(draft_model_path)

        start = time.time()
        output_tokens = mlx_lm.generate(
            model=main_model,
            tokenizer=tokenizer,
            prompt=prompt,
            verbose=False,
            draft_model=draft_model,
            num_draft_tokens=n_draft if n_draft > 0 else None,
            max_tokens=max_gen_tokens,
        )
        print("DEBUG → output_tokens:", output_tokens)
        print("DEBUG → type:", type(output_tokens))

        generated_text = output_tokens


        end = time.time()
        
        elapsed = end - start
        tokens_per_sec = max_gen_tokens / elapsed

        # If num_draft_tokens = 0, store as reference output
        if n_draft == 0:
            reference_outputs[q] = generated_text
            accuracy = 1.0
        else:
            ref_tokens = get_tokens(reference_outputs[q])
            gen_tokens = get_tokens(generated_text)
            match_len = min(len(ref_tokens), len(gen_tokens), max_gen_tokens)
            matches = sum(1 for i in range(match_len) if ref_tokens[i] == gen_tokens[i])
            accuracy = matches / match_len if match_len > 0 else 0

        results[q].append((tokens_per_sec, accuracy))
        print(f"    ⏱️ {tokens_per_sec:.2f} tokens/sec | 🎯 Accuracy: {accuracy:.2%}")

# Plotting
plt.figure(figsize=(12, 6))
for q in quantizations:
    speeds = [s for s, _ in results[q]]
    plt.plot(num_draft_tokens_list, speeds, label=f"{q}", marker="o")

plt.xlabel("num_draft_tokens")
plt.ylabel("Tokens per second")
plt.title("Speculative Decoding Performance on M2 Ultra")
plt.legend(title="Quantization Level")
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy plot
plt.figure(figsize=(12, 6))
for q in quantizations:
    accs = [a for _, a in results[q]]
    plt.plot(num_draft_tokens_list, accs, label=f"{q}", marker="o")

plt.xlabel("num_draft_tokens")
plt.ylabel("Accuracy vs Autoregressive Output")
plt.title("Speculative Decoding Output Accuracy (vs AR Baseline)")
plt.legend(title="Quantization Level")
plt.grid(True)
plt.tight_layout()
plt.show()
