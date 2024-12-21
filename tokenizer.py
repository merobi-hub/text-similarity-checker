import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def split_into_sentences(text):
    """Splits a text into sentences using NLTK's sentence tokenizer."""
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)
