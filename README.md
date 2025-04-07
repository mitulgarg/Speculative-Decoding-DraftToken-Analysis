# Speculative-Decoding-DraftToken-Analysis
#MLX #AppleSilicon #SpeculativeDecoding 
This project analyzes the **performance and quality trade-offs** in speculative decoding using draft tokens and different quantized model configurations. It evaluates:

- ⚡ **Generation speed** (tokens/sec)  
- 🔍 **Semantic similarity (Accuracy metric)** (cosine similarity via sentence embeddings)  
- 📝 **Text quality (Accuracy metric)** (ROUGE-L score)  
 
---
### ▶️ Install the pip package
```bash
pip install speculative-decoding-metrics
```
---
### Customize the Speculative decoding analysis
###### After installing the package, create a demo script with a default prompt and model: 
demo/run_example.py
```
from speculative_decoding_metrics.main import run_evaluation

run_evaluation(
    base_model="phi-3-mini-4k-instruct",  #Use a model that'll run on your local
    main_quant="8bit",          #use q8 instead of "8bit" based on HuggingFace Repo name 
    draft_quant="4bit",         #use q4 instead of "4bit" based on HuggingFace Repo name
    prompt="How do LLMs work?",
    max_tokens=64,              #Tweak max tokens per output
    num_draft_tokens_list=[0, 1, 2, 3, 4]
)


```
## ▶️ Run the Demo
```bash
python demo/run_example.py
```

---

## 📌 What Is Speculative Decoding?

**Speculative decoding** speeds up language generation by using a smaller "draft" model to propose tokens, which are then verified by a larger "main" model.

This repo benchmarks speculative decoding using:

- **Main model**: Quantized to `8bit` → `mlx-community/<model>-8bit`
- **Draft model**: Quantized to `4bit` → `mlx-community/<model>-4bit`

---

## 📊 Visualized Metrics
Based on these results, the user can decide which speculative decoding they want to run for the best results!
This package generates plots comparing output quality and speed across draft token counts:

1. **Tokens/sec** – Speed boost with draft tokens  
2. **Cosine Similarity** – Semantic match with baseline (no draft)  
3. **ROUGE-L** – Text overlap quality score  

![Results Graph](https://raw.githubusercontent.com/mitulgarg/Speculative-Decoding-DraftToken-Analysis/main/assets/image.png)

---

## 🛠 Customization Tips

- 🔧 **Change the prompt** – Modify the `prompt` in `demo/run_example.py`  
- 🧠 **Try other models** – Swap the `base_model` string (e.g., Mistral, TinyLlama)  
- 🎛️ **Adjust draft token range** – Modify `num_draft_tokens_list` for finer control 
- 📏 **Set max output length** – Use `max_tokens` to limit generation length   

---

## Acknowledgments

- **Apple MLX** – Lightweight ML framework  
- **HuggingFace** – Transformers + SentenceTransformers 
- **Google Research** – ROUGE scoring tools  
