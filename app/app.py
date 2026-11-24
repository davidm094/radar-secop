"""Radar SECOP – demo en Streamlit.

Ejecutar desde la raíz del repo:  streamlit run app/app.py
Requiere el modelo (scripts/03) y el índice de procesos (scripts/07); con datos congelados, antes scripts/00.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from radar.config import DATA_PROC, REPORTS, UMBRAL_DISCREPANCIA
from radar.data import familias_unspsc, segmentos_unspsc
from radar.explain import local, Vecinos
from radar.model import cargar
from radar.search import BuscadorProcesos
from radar.text import preparar, limpiar

st.set_page_config(page_title="Radar SECOP", layout="wide")
UMBRAL_APP = 0.70   # umbral calibrado (ver reports/explicabilidad.json)
UMBRAL_DUDA = 0.35  # por debajo, la familia es dudosa y se muestra también el segmento (2 dígitos)


@st.cache_resource(show_spinner="Cargando modelo y catálogo…")
def recursos():
    modelo = cargar("familia_v1")
    nombres = familias_unspsc()
    c = pd.read_parquet(DATA_PROC / "contratos_prep.parquet",
                        columns=["id_contrato", "objeto_del_contrato", "nombre_entidad", "fam", "txt_lem"])
    gold = ROOT / "data" / "gold" / "gold_ids.csv"     # el gold set nunca se muestra como ejemplo
    if gold.exists():
        c = c[~c.id_contrato.isin(set(pd.read_csv(gold).id_contrato))]
    c = c.sample(min(60_000, len(c)), random_state=0)
    vecinos = Vecinos(modelo, c.txt_lem.tolist(), c[["objeto_del_contrato", "nombre_entidad", "fam"]])
    return modelo, nombres, vecinos, segmentos_unspsc()


@st.cache_data(show_spinner=False)
def alertas():
    """Contratos del conjunto de test donde el modelo contradice el código declarado con confianza ≥ 0,70."""
    return pd.read_csv(REPORTS / "discrepancias.csv", dtype={"fam": str, "pred": str})


def segmento_mas_probable(P: np.ndarray):
    """Suma las probabilidades de las familias de cada segmento (primeros 2 dígitos)."""
    s = pd.Series(P, index=pd.Index(MODELO.classes_).str[:2]).groupby(level=0).sum()
    return s.idxmax(), float(s.max())


def aviso_segmento(P: np.ndarray):
    if P.max() < UMBRAL_DUDA:
        seg, ps = segmento_mas_probable(P)
        st.info(f"El modelo no está seguro de la familia (la mejor tiene {P.max():.0%}). "
                f"A nivel más general, lo más probable es el segmento **{seg} · {SEGMENTOS.get(seg, '¿?')}** ({ps:.0%}). "
                "Pruebe a describir con más detalle qué vende o qué servicio presta.")


@st.cache_resource(show_spinner="Cargando procesos publicados…")
def buscador():
    p = pd.read_parquet(DATA_PROC / "procesos_idx.parquet")
    emb = np.load(DATA_PROC / "procesos_emb.npy")
    return BuscadorProcesos(p, p.txt_lem.tolist(), p.texto.tolist(), embeddings=emb)


def nombre(fam: str) -> str:
    return f"{fam} · {NOMBRES.get(fam, '¿?')}"


def grafico_explicacion(exp: pd.DataFrame):
    if exp.empty:
        st.info("El texto no tiene términos que el modelo conozca.")
        return
    d = exp.sort_values("contribucion", ascending=False)
    st.bar_chart(d, x="termino", y="contribucion", horizontal=True, height=40 + 28 * len(d), sort="-contribucion",
                 x_label="", y_label="contribución al puntaje de la familia")
    st.caption("Barras positivas empujan HACIA la familia sugerida; negativas, en contra. "
               "Es la contribución exacta de cada término en el modelo lineal.")


MODELO, NOMBRES, VECINOS, SEGMENTOS = recursos()

st.title("Radar SECOP")
st.write("Encuentre en qué categoría del Estado encaja su negocio y qué procesos de SECOP II le sirven.")
tab_pyme, tab_audit, tab_alertas, tab_info = st.tabs(
    ["🔎 Para pymes", "🧾 Revisar un contrato", "🚩 Alertas para veedurías", "ℹ️ Cómo funciona"])

# ---------------------------------------------------------------- Pymes
with tab_pyme:
    q = st.text_area("Describa con sus palabras qué vende o qué servicio presta",
                     "Hacemos mantenimiento preventivo y correctivo de aires acondicionados para oficinas")
    try:
        B = buscador()
        f1, f2 = st.columns(2)
        deptos = f1.multiselect("Departamento de la entidad (opcional)", sorted(B.p.departamento_entidad.dropna().unique()),
                                placeholder="Todos los departamentos")
        solo_abiertos = f2.checkbox("Solo procesos en estado Abierto", value=False,
                                    help="Por defecto se incluyen los procesos Publicados y los Abiertos.")
    except FileNotFoundError:
        B, deptos, solo_abiertos = None, [], False
    if st.button("Buscar", type="primary") and q.strip():
        qp = preparar(q)
        P = MODELO.predict_proba([qp])[0]
        orden = np.argsort(-P)[:3]
        st.subheader("Categorías sugeridas (familia UNSPSC)")
        cols = st.columns(3)
        for c, i in zip(cols, orden):
            c.metric(nombre(MODELO.classes_[i]), f"{P[i]:.0%}")
        aviso_segmento(P)
        with st.expander("¿Por qué esta categoría?", expanded=True):
            grafico_explicacion(local(MODELO, qp, MODELO.classes_[orden[0]], k=8))
            st.write("**Contratos reales parecidos:**")
            v = VECINOS.buscar(qp, k=3)
            v["familia"] = v.fam.map(nombre)
            st.dataframe(v[["objeto_del_contrato", "nombre_entidad", "familia", "similitud"]],
                         hide_index=True, width="stretch")
        st.subheader("Procesos publicados que le pueden servir")
        if B is None:
            st.warning("Falta el índice de procesos: corra scripts/07_indexar_procesos.py")
        else:
            probs = {MODELO.classes_[i]: float(P[i]) for i in orden}
            mascara = np.ones(len(B.p), dtype=bool)
            if deptos:
                mascara &= B.p.departamento_entidad.isin(deptos).values
            if solo_abiertos:
                mascara &= (B.p.estado_del_procedimiento == "Abierto").values
            r = B.buscar(q, qp, probs, k=10, mascara=mascara)
            if r.empty:
                st.info("No hay procesos con esos filtros. Quite alguno y vuelva a buscar.")
            else:
                r["familia"] = r.fam.map(nombre) + np.where(r.familia_inferida, " (inferida)", "")
                r["precio_base"] = r.precio_base.map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                # la API entrega el enlace como {"url": "..."}; se deja solo la dirección para que sea clicable
                r["urlproceso"] = r.urlproceso.map(lambda u: u.get("url", "") if isinstance(u, dict) else (u or ""))
                st.dataframe(r[["texto", "entidad", "departamento_entidad", "precio_base", "familia",
                                "estado_del_procedimiento", "score", "urlproceso"]],
                             hide_index=True, width="stretch",
                             column_config={"urlproceso": st.column_config.LinkColumn("SECOP", display_text="abrir"),
                                            "score": st.column_config.ProgressColumn("ajuste", min_value=0, max_value=1)})
                st.caption("Ajuste = 0,5·similitud semántica + 0,3·similitud de términos + 0,2·probabilidad de la familia. "
                           "Los procesos son una foto del día de descarga; verifique el estado en SECOP.")

# ---------------------------------------------------------------- Auditoría
with tab_audit:
    st.write("Pegue el objeto de un contrato y su código UNSPSC declarado. "
             "Si el modelo sugiere otra familia con confianza alta, se marca **para revisar** "
             "(una discrepancia no es un error ni una irregularidad: es un dato a verificar).")
    obj = st.text_area("Objeto del contrato",
                       "Prestación de servicios profesionales como abogado para el acompañamiento jurídico "
                       "de la personería municipal")
    cod = st.text_input("Código declarado (8 dígitos)", "80111600")
    if st.button("Revisar") and obj.strip():
        qp = preparar(obj)
        P = MODELO.predict_proba([qp])[0]
        i = P.argmax()
        pred, conf, decl = MODELO.classes_[i], float(P[i]), cod.strip()[:4]
        c1, c2 = st.columns(2)
        c1.metric("Familia declarada", nombre(decl))
        c2.metric("Familia sugerida por el modelo", nombre(pred), f"confianza {conf:.0%}", delta_color="off", delta_arrow="off")
        if pred != decl and conf >= UMBRAL_APP:
            st.error("Posible discrepancia: la descripción apunta a otra familia. Para revisar.")
        elif pred != decl:
            st.warning("El modelo sugiere otra familia, pero con confianza baja: no se marca.")
        else:
            st.success("La familia declarada coincide con la sugerida.")
        aviso_segmento(P)
        st.write("**Términos que explican la sugerencia**")
        grafico_explicacion(local(MODELO, qp, pred, k=8))
        st.write("**Texto que ve el modelo** (sin fórmulas de contratación, lematizado):")
        st.code(qp or "(vacío)")
        st.caption("Texto antes de lematizar: " + limpiar(obj))

# ---------------------------------------------------------------- Alertas
with tab_alertas:
    st.write("Contratos donde el objeto apunta a una familia distinta de la declarada, con confianza de "
             f"{UMBRAL_APP:.0%} o más. Son contratos **para revisar**: una discrepancia no es un error ni una "
             "irregularidad, y muchas vienen de la costumbre de declarar todo en 8011.")
    try:
        A = alertas()
    except FileNotFoundError:
        A = None
        st.info("Corra scripts/06_discrepancias.py")
    if A is not None:
        st.caption(f"{len(A):,} alertas sobre 36.844 contratos que el modelo no vio al entrenar (4,5 %). ".replace(",", ".")
                   + "En una revisión manual de 100 de ellas, elegidas al azar, la familia sugerida era la correcta en 89.")
        a1, a2, a3 = st.columns([2, 2, 1])
        por_entidad = A.nombre_entidad.value_counts()
        ent = a1.selectbox("Entidad", ["Todas"] + [f"{e} ({n})" for e, n in por_entidad.items()])
        sug = a2.selectbox("Familia sugerida", ["Todas"] + [nombre(f) for f in A.pred.value_counts().index])
        cmin = a3.slider("Confianza mínima", 0.70, 0.90, 0.70, 0.05)
        V = A[A.conf >= cmin]
        if ent != "Todas":
            V = V[V.nombre_entidad == ent.rsplit(" (", 1)[0]]
        if sug != "Todas":
            V = V[V.pred == sug.split(" · ")[0]]
        m1, m2, m3 = st.columns(3)
        m1.metric("Alertas con estos filtros", f"{len(V):,}".replace(",", "."))
        m2.metric("Entidades", V.nombre_entidad.nunique())
        m3.metric("Declaradas en 8011", f"{(V.fam == '8011').mean():.0%}" if len(V) else "–")
        T = V.assign(declarada=V.fam.map(nombre), sugerida=V.pred.map(nombre))
        st.dataframe(T[["objeto_del_contrato", "nombre_entidad", "declarada", "sugerida", "conf"]],
                     hide_index=True, width="stretch", height=320,
                     column_config={"conf": st.column_config.ProgressColumn("confianza", min_value=0, max_value=1),
                                    "objeto_del_contrato": st.column_config.TextColumn("objeto", width="large")})
        st.download_button("Descargar esta lista (CSV)", T.to_csv(index=False).encode("utf-8-sig"),
                           "alertas_radar_secop.csv", "text/csv")
        g1, g2 = st.columns(2)
        with g1:
            st.write("**Entidades con más alertas**")
            st.bar_chart(V.nombre_entidad.str.slice(0, 45).value_counts().head(10), horizontal=True, height=320)
        with g2:
            st.write("**De qué familia a cuál**")
            pares = (T.declarada.str.slice(0, 4) + " → " + T.sugerida.str.slice(0, 32)).value_counts().head(10)
            st.bar_chart(pares, horizontal=True, height=320)
        if len(V):
            st.write("**¿Por qué se marcó un contrato?**")
            op = st.selectbox("Contrato", V.index, format_func=lambda i: f"{V.conf[i]:.0%} · {V.objeto_del_contrato[i][:110]}")
            fila = V.loc[op]
            st.write(f"Declarada: **{nombre(fila.fam)}** · Sugerida: **{nombre(fila.pred)}**")
            st.write(fila.objeto_del_contrato)
            grafico_explicacion(local(MODELO, preparar(fila.objeto_del_contrato), fila.pred, k=8))

# ---------------------------------------------------------------- Info
with tab_info:
    st.subheader("Métricas del modelo (acuerdo con el código declarado en SECOP)")
    try:
        st.dataframe(pd.read_csv(REPORTS / "metricas_familia.csv").round(3), hide_index=True)
    except FileNotFoundError:
        st.info("Corra scripts/03_entrenar_evaluar.py")
    st.markdown(
        "- Las etiquetas de SECOP tienen sesgo: la familia 8011 (recursos humanos) se usa como cajón de sastre. "
        "Por eso estas métricas miden **acuerdo con SECOP**, no exactitud.\n"
        "- Contra un gold set de 300 contratos, casi todos los declarados en 8011 corresponden a otra familia "
        "(95 % a 99 % según la referencia). La exactitud del modelo queda entre 0,23 (etiquetas humanas, poco "
        "confiables) y 0,48 (consenso de dos anotaciones de Claude); el detalle está en `documento_final.pdf`.\n"
        "- Datos: SECOP II (datos.gov.co, CC BY-SA 4.0). Catálogo: UNSPSC v14 en español, Colombia Compra Eficiente.")
