## Text similarity checker using SBERT and NLTK

This tool uses the [SBERT](https://sbert.net/index.html) SentenceTransformer and [NLTK](https://www.nltk.org/) Punkt Sentence Tokenizer to compare two texts for similarity.

It outputs a sorted list of high-scoring sentence pairs with scores for each, the most similar pair, and the similarity index (average similarity).

### To use

Install the required dependencies:

```
pip install -r requirements.txt
```

Run the `text_similarity_checker` script from the command line:

```
python3 text_similarity_checker.py
```

> [!NOTE] 
> By default, the script compares Bob Dylan's [rather infamous Nobel lecture](https://slate.com/culture/2017/06/did-bob-dylan-take-from-sparknotes-for-his-nobel-lecture.html) to its alleged source. Add your own texts for comparison as Python strings to the project's root directory. You will then need to modify `text_similarity_checker` slightly to use them. See the script for details.

### Status

In development.
