"""Paso 8 – Búsqueda de procesos publicados para una pyme (recuperación de información).

Puntaje híbrido por proceso:
    score = w_emb · coseno(embeddings) + w_tfidf · coseno(TF-IDF) + w_fam · P(familia del proceso | consulta)
- embeddings estáticos multilingües (model2vec, sin GPU) capturan sinónimos ("aires" ~ "climatización")
- TF-IDF premia coincidencias literales de términos técnicos
- el clasificador aporta la familia UNSPSC sugerida para la consulta
Los pesos se ajustan con las ~30 consultas de pyme juzgadas a mano (precisión@5).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MODELO_EMB = "minishlab/potion-multilingual-128M"


def _encoder():
    """Codificador completo de model2vec. Se usa para indexar todos los procesos (scripts/07)."""
    from model2vec import StaticModel
    return StaticModel.from_pretrained(MODELO_EMB)


class EncoderLigero:
    """El mismo codificador, para consultas sueltas y con poca memoria (lo usa la app).

    Un modelo de model2vec sin pesos por token es: tokenizar, promediar las filas de la matriz de embeddings
    y normalizar. Aquí la matriz (512 MB) se lee del disco con memmap, así que solo se cargan las filas de
    los tokens de la consulta. Da los mismos vectores que _encoder().encode (lo comprueba tests/test_basico.py
    cuando el modelo está descargado).
    """

    def __init__(self):
        import json
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(hf_hub_download(MODELO_EMB, "tokenizer.json"))
        self.unk = self.tok.token_to_id("[UNK]") if self.tok.token_to_id("[UNK]") is not None else self.tok.token_to_id("<unk>")
        ruta = hf_hub_download(MODELO_EMB, "model.safetensors")
        with open(ruta, "rb") as f:                       # formato safetensors: 8 bytes con el largo de la cabecera JSON
            n = int.from_bytes(f.read(8), "little")
            meta = json.loads(f.read(n))["embeddings"]
        assert meta["dtype"] == "F32"
        self.E = np.memmap(ruta, dtype="<f4", mode="r", offset=8 + n + meta["data_offsets"][0], shape=tuple(meta["shape"]))

    def encode(self, textos: list[str]) -> np.ndarray:
        out = np.zeros((len(textos), self.E.shape[1]), dtype="float32")
        for i, e in enumerate(self.tok.encode_batch_fast(list(textos), add_special_tokens=False)):
            ids = [t for t in e.ids if t != self.unk][:512]
            if ids:
                out[i] = np.asarray(self.E[ids]).mean(axis=0)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-32)


def _unit(E):
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)


class BuscadorProcesos:
    def __init__(self, procesos: pd.DataFrame, textos_preparados: list[str] | None, textos_crudos: list[str] | None,
                 embeddings: np.ndarray | None = None, tfidf=None, ligero: bool = True):
        """tfidf = (vectorizador, matriz) ya calculados (scripts/07 los guarda); si no, se ajustan aquí."""
        self.p = procesos.reset_index(drop=True)
        if tfidf is not None:
            self.tfidf, self.Xt = tfidf
        else:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
            self.Xt = self.tfidf.fit_transform(textos_preparados)
        self.enc = EncoderLigero() if ligero else _encoder()
        self.Xe = _unit(embeddings if embeddings is not None else _encoder().encode(list(textos_crudos)))
        self.fam = self.p["fam"].fillna("").values

    @classmethod
    def desde_disco(cls, carpeta):
        """Carga el índice que deja scripts/07: procesos_idx.parquet, procesos_tfidf.joblib y procesos_emb.npy."""
        import joblib
        from pathlib import Path
        carpeta = Path(carpeta)
        return cls(pd.read_parquet(carpeta / "procesos_idx.parquet"), None, None,
                   embeddings=np.load(carpeta / "procesos_emb.npy"), tfidf=joblib.load(carpeta / "procesos_tfidf.joblib"))

    def buscar(self, consulta_cruda: str, consulta_preparada: str, probs_familia: dict[str, float],
               k: int = 10, w_emb: float = 0.5, w_tfidf: float = 0.3, w_fam: float = 0.2,
               mascara: np.ndarray | None = None) -> pd.DataFrame:
        """mascara: booleano por proceso; si se pasa, solo se devuelven procesos con True (filtros de la app)."""
        s_emb = (self.Xe @ _unit(self.enc.encode([consulta_cruda])).T).ravel()
        s_tf = linear_kernel(self.tfidf.transform([consulta_preparada]), self.Xt).ravel()
        s_fam = np.array([probs_familia.get(f, 0.0) for f in self.fam])
        score = w_emb * s_emb + w_tfidf * s_tf + w_fam * s_fam
        if mascara is not None:
            score = np.where(mascara, score, -np.inf)
        idx = np.argsort(-score)[:k * 3]
        idx = idx[np.isfinite(score[idx])]
        res = self.p.iloc[idx].assign(score=score[idx], sim_semantica=s_emb[idx],
                                      sim_lexica=s_tf[idx], p_familia=s_fam[idx])
        # varias entidades publican procesos con el mismo texto: se muestra uno por texto
        return res.drop_duplicates("texto").head(k)
