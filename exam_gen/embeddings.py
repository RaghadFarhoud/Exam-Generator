"""
embeddings.py — ترقية الاسترجاع إلى تشابه دلالي (اختياري لكنه موصى به
عند العمل بكامل محاضرات المادة).

الاستخدام:
    from exam_gen.embeddings import VoyageEmbedder   # أو SentenceTransformerEmbedder
    pipe = ExamPipeline(corpus, llm, embed_fn=VoyageEmbedder())

كلاهما يطبّق الواجهة: embed_fn(text) -> List[float]
Retriever سيستخدمها تلقائياً إن مُرّرت (مع تخزين مؤقت داخلي).
"""
from __future__ import annotations
from typing import List


class VoyageEmbedder:
    """يتطلب: pip install voyageai + مفتاح VOYAGE_API_KEY (أو أي مزوّد API)."""
    def __init__(self, model: str = "voyage-3"):
        import voyageai
        self.client = voyageai.Client()
        self.model = model
        self._cache: dict = {}

    def __call__(self, text: str) -> List[float]:
        if text not in self._cache:
            r = self.client.embed([text], model=self.model)
            self._cache[text] = r.embeddings[0]
        return self._cache[text]


class SentenceTransformerEmbedder:
    """
    بديل محلي مجاني بلا API — مناسب لمشروع تخرج (قابل لإعادة الإنتاج).
    يتطلب: pip install sentence-transformers
    نموذج متعدد اللغات يدعم العربية/الإنجليزية المختلطة.
    """
    def __init__(self, model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self._cache: dict = {}

    def __call__(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = self.model.encode(text).tolist()
        return self._cache[text]
