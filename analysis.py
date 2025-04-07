import spacy
from spacy.language import Language
from spacy.tokens import Span
from .config import PARANORMAL_TERMS
from .dataset_loader import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nlp = spacy.load("en_core_web_lg")


@Language.component("paranormal_entities")
def paranormal_entities(doc):
    new_ents = []
    text_lower = doc.text.lower()

    for label, phrases in PARANORMAL_TERMS.items():
        for phrase in phrases:
            if phrase in text_lower:
                start = text_lower.find(phrase)
                end = start + len(phrase)
                span = doc.char_span(start, end, label=label)
                if span:
                    new_ents.append(span)

    doc.ents = new_ents
    return doc


class ParanormalAnalyzer:
    def __init__(self):
        self.nlp = nlp
        self.nlp.add_pipe("paranormal_entities", after="ner")
        self.dataset = load_dataset()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self._prepare_dataset_vectors()

    def _prepare_dataset_vectors(self):
        self.dataset_texts = [item['text'] for item in self.dataset]
        self.vectorizer.fit(self.dataset_texts)

    def analyze(self, text):
        doc = self.nlp(text)

        entities = [{
            "text": ent.text,
            "type": ent.label_,
            "description": f"Recognized as {ent.label_}"
        } for ent in doc.ents]

        similarity_scores = self._find_similar_reports(text)

        return {
            "entities": entities,
            "similar_reports": similarity_scores[:3],
            "is_paranormal": len(entities) > 0
        }

    def _find_similar_reports(self, text):
        text_vec = self.vectorizer.transform([text])
        dataset_vecs = self.vectorizer.transform(self.dataset_texts)

        similarities = cosine_similarity(text_vec, dataset_vecs)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        return [{
            "text": self.dataset[i]['text'],
            "similarity": float(similarities[i]),
            "label": self.dataset[i]['label']
        } for i in sorted_indices if similarities[i] > 0.3]