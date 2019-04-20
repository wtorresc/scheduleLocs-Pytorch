import random
import spacy

spacy_en = spacy.load('en')

def _spacy_tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]



