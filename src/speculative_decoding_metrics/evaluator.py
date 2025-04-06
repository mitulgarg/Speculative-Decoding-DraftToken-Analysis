import time
import gc
import os
import mlx_lm
import mlx.core as mx
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

class SpeculativeDecoderEvaluator:
    def __init__(self, base_model="phi-3-mini-4k-instruct", max_tokens=64, prompt_text="Write a story about Einstein."):
        os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"
        self.max_tokens = max_tokens
        self.prompt_text = prompt_text

        # Model paths
        self.main_model_path = f"mlx-community/{base_model}-8bit"
        self.draft_model_path = f"mlx-community/{base_model}-4bit"

        # Tokenizer and prompt setup
        _, self.tokenizer = mlx_lm.load(self.draft_model_path)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        self.messages = [{"role": "user", "content": self.prompt_text}]
        self.prompt = self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True)

        self.main_model, _ = mlx_lm.load(self.main_model_path)


    def embed(self, text):
        return np.array(self.embedding_model.encode([text])[0], dtype=np.float32)

    def evaluate(self, draft_token_options):
        results = []
        reference_output = ""
        reference_embedding = None

        for n_draft in draft_token_options:
            print(f"\n→ num_draft_tokens = {n_draft}")

            draft_model = None
            if n_draft > 0:
                draft_model, _ = mlx_lm.load(self.draft_model_path)

            start = time.time()
            output_tokens = mlx_lm.generate(
                model=self.main_model,
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                verbose=False,
                draft_model=draft_model,
                num_draft_tokens=n_draft if n_draft > 0 else None,
                max_tokens=self.max_tokens,
            )
            end = time.time()

            generated_text = output_tokens
            elapsed = end - start
            tokens_per_sec = self.max_tokens / elapsed

            if n_draft == 0:
                reference_output = generated_text
                reference_embedding = self.embed(generated_text)
                similarity = 1.0
                rouge_l = 1.0
            else:
                gen_embedding = self.embed(generated_text)
                similarity = float(np.dot(gen_embedding, reference_embedding) /
                                (np.linalg.norm(gen_embedding) * np.linalg.norm(reference_embedding)))
                rouge = self.rouge.score(reference_output, generated_text)
                rouge_l = rouge['rougeL'].fmeasure

            results.append((tokens_per_sec, similarity, rouge_l))
            print(f" Time : {tokens_per_sec:.2f} tokens/sec | Cosine similarity : {similarity:.2%} | ROUGE-L: {rouge_l:.2%}")

            if draft_model is not None:
                del draft_model
            gc.collect()
            mx.eval()

        del self.main_model
        gc.collect()
        mx.eval()
        return results
