# Speculative-Decoding-DraftToken-Analysis

This project analyzes the performance and quality trade-offs in **speculative decoding** using draft tokens with different model configurations. It compares output speed, semantic similarity, and ROUGE-L scores across varying numbers of draft tokens.

## Run main analysis script (Runs default phi-3-mini-4k-instruct model)
```bash
python main.py
```
## You can also specify your preferred model using --model and --prompt
```bash
python main.py --model phi-3-mini-4k-instruct --prompt "What are the benefits of AI in education?"

```
## 📌 Overview

Speculative decoding is a technique to accelerate language generation by proposing draft tokens before validating them with a larger model. This repo evaluates:

- **Throughput** (tokens/sec)
- **Semantic similarity** (cosine similarity via sentence embeddings)
- **Text quality** (ROUGE-L score)

All experiments are run using:
- **Main model**: 8-bit quantized (`mlx-community/<model>-8bit`)
- **Draft model**: 4-bit quantized (`mlx-community/<model>-4bit`)

## 📊 Visualized Metrics

Three metrics are plotted against the number of draft tokens:

1. **Tokens per second** – Measures generation speed.
2. **Cosine Similarity** – Semantic similarity vs baseline (no draft).
3. **ROUGE-L** – Overlap-based quality score vs baseline.

![alt text](assets/image.png)

## ⚙️ Requirements

- Python 3.8+
- MLX + `mlx_lm`
- SentenceTransformers
- rouge_score
- Matplotlib
- NumPy

## Install dependencies using pip 

```bash 
pip install mlx_lm sentence-transformers rouge-score matplotlib numpy
```


## ⚙️ Customization

Tailor the analysis to your specific needs:

- **Prompt Modification**: Adjust the input prompt within `evaluator.py` by changing the `self.prompt_text` variable.
- **Model Selection**: Experiment with different MLX-compatible models by modifying the model names in the scripts.
- **Draft Token Range**: Alter the range of draft tokens explored in `main.py`.

## 🖼️ Example Output

The script will generate plots showcasing the trade-offs between generation speed and output quality as a function of the number of draft tokens used. These visualizations provide insights into the optimal number of draft tokens for different use cases.

## 🙏 Acknowledgments

This work leverages the following open-source projects:

- **MLX**: Developed by Apple.
- **HuggingFace Transformers & SentenceTransformers**: Provided by Hugging Face.
- **ROUGE Scoring**: Developed by Google Research.