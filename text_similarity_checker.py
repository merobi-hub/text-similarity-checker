from sentence_transformers import SentenceTransformer
from tokenizer import split_into_sentences
from dylan_lecture import dylan_lec
from moby_dick_sparknotes import spark_notes

model = SentenceTransformer("all-MiniLM-L6-v2")

# Pass your own strings here (don't forget to add imports first)
sentences1 = split_into_sentences(dylan_lec)
sentences2 = split_into_sentences(spark_notes)

embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

similarities = model.similarity(embeddings1, embeddings2)

high_scorers = []
high_score = 0
high_scoring_match = ()
# Output the pairs with their score
for idx_i, sentence1 in enumerate(sentences1):
	# print(sentence1)
	for idx_j, sentence2 in enumerate(sentences2):
		score = similarities[idx_i][idx_j]
		# print(f" - {sentence2: <30}: {score:.4f}")
		if score > high_score:
			high_score = score
			high_scoring_match = (sentence1, sentence2)
		if score > 0.7:
			high_scorers.append((sentence1, sentence2, score))

# Output the pair with the highest score
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
