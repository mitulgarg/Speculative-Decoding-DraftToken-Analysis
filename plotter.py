import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class ResultPlotter:
    def __init__(self, num_draft_tokens_list, results):
        self.tokens_list = num_draft_tokens_list
        self.results = results

    def plot_all(self):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        speeds = [s for s, _, _ in self.results]
        sims = [a * 100 for _, a, _ in self.results]
        rouges = [r * 100 for _, _, r in self.results]

        # Speed
        axs[0].plot(self.tokens_list, speeds, label="8bit + 4bit draft", marker="o")
        axs[0].set_title("Speculative Decoding Performance (8bit main, 4bit draft)")
        axs[0].set_xlabel("num_draft_tokens")
        axs[0].set_ylabel("Tokens per second")
        axs[0].set_ylim(bottom=0)
        axs[0].xaxis.set_major_locator(MultipleLocator(1))
        axs[0].grid(True)
        axs[0].legend()

        # Cosine Similarity
        axs[1].plot(self.tokens_list, sims, label="Cosine Similarity", marker="o", color="orange")
        axs[1].set_title("Output Cosine Similarity (vs 8bit AR Output)")
        axs[1].set_xlabel("num_draft_tokens")
        axs[1].set_ylabel("Cosine Similarity (%)")
        axs[1].set_ylim(0, 100)
        axs[1].xaxis.set_major_locator(MultipleLocator(1))
        axs[1].yaxis.set_major_locator(MultipleLocator(10))
        axs[1].grid(True)
        axs[1].legend()

        # ROUGE-L
        axs[2].plot(self.tokens_list, rouges, label="ROUGE-L F1 Score", marker="o", color="green")
        axs[2].set_title("ROUGE-L Score (vs 8bit AR Output)")
        axs[2].set_xlabel("num_draft_tokens")
        axs[2].set_ylabel("ROUGE-L F1 Score (%)")
        axs[2].set_ylim(0, 100)
        axs[2].xaxis.set_major_locator(MultipleLocator(1))
        axs[2].yaxis.set_major_locator(MultipleLocator(10))
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()
