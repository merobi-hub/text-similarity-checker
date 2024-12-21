from sentence_transformers import SentenceTransformer
from tokenizer import split_into_sentences
from dylan_lecture import dylan_lec
from moby_dick_sparknotes import spark_notes
from numpy.typing import NDArray
import torch
from torch import tensor

# Texts (strings) for comparison. Import from root then modify these definitions.
TEXT1 = dylan_lec
TEXT2 = spark_notes

class SimilarityChecker:

	def __init__(self, text1, text2):
		self.text1 = text1
		self.text2 = text2
		self.sentences1: list = []
		self.sentences2: list = []
		self.embeddings1: NDArray = []
		self.embeddings2: NDArray = []
		self.similarities: tensor = []
		self.model = SentenceTransformer("all-MiniLM-L6-v2")

	def split_texts(self):
		self.sentences1 = split_into_sentences(self.text1)
		self.sentences2 = split_into_sentences(self.text2)

	def encode_texts(self):
		self.embeddings1 = self.model.encode(self.sentences1)
		self.embeddings2 = self.model.encode(self.sentences2)

	def produce_similarities(self):
		self.similarities = self.model.similarity(self.embeddings1, self.embeddings2)

	def compare_texts(self):
		high_scorers = []
		high_score = 0
		high_scoring_match = ()
		num_scores = 0
		agg_score = 0
		for idx_i, sentence1 in enumerate(self.sentences1):
			for idx_j, sentence2 in enumerate(self.sentences2):
				score = self.similarities[idx_i][idx_j]
				num_scores += 1
				agg_score += score
				if score > high_score:
					high_score = score
					high_scoring_match = (sentence1, sentence2)
				if score > 0.7:
					high_scorers.append((sentence1, sentence2, score))

		print("===================")
		print("===================")
		print("Closest match:")
		print(f"Dylan: {high_scoring_match[0]}") 
		print(f"Spark: {high_scoring_match[1]}")
		print(f"Score: {high_score:.4f}")

		print("===================")
		print("High scorers:")
		for item in sorted(high_scorers, key=lambda x: x[2], reverse=True):
			print("Dylan: ", item[0])
			print("Spark: ", item[1])
			print("Score: ", item[2])
			print("---------------")

		print("===================")
		print("===================")
		print(f"Similarity index: ", agg_score / num_scores)

def main():
	check = SimilarityChecker(TEXT1, TEXT2)
	check.split_texts()
	check.encode_texts()
	check.produce_similarities()
	check.compare_texts()

if __name__ == "__main__":
    main()
