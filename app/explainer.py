from typing import List, Dict
import numpy as np
from lime.lime_text import LimeTextExplainer
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences

CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

class FastLIMEExplainer:
    def __init__(self, model, tokenizer, max_len: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    def _predict_proba(self, texts: List[str]):
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")
        return self.model.predict(padded, verbose=0)

    def explain(self, text: str, num_features: int = 5, num_samples: int = 150) -> List[Dict]:
        probs = self._predict_proba([text])[0]
        top_idx = int(np.argmax(probs))
        
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self._predict_proba,
            labels=[top_idx],
            num_features=num_features,
            num_samples=num_samples
        )
        token_weights = exp.as_list(label=top_idx)
        return [{"word": word, "weight": round(float(weight), 4)} for word, weight in token_weights]
