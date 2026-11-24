"""Radar SECOP – demo en Streamlit.

Ejecutar desde la raíz del repo:  streamlit run app/app.py
Requiere el modelo (scripts/15) y el índice de procesos (scripts/07); con datos congelados, antes scripts/00.
Lo más fácil: python scripts/preparar_demo.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from radar.config import DATA_PROC, REPORTS, MODELO_APP
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
    modelo = cargar(MODELO_APP)
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
st.write("Para **pymes**: encuentre en qué categoría del Estado encaja su negocio y qué procesos de SECOP II le sirven. "
         "Para **veedurías**: revise si el código declarado de un contrato coincide con lo que dice su objeto.")
tab_pyme, tab_audit, tab_alertas, tab_info = st.tabs(
    ["🔎 Para pymes", "🧾 Revisar un contrato", "🚩 Alertas para veedurías", "ℹ️ Cómo funciona"])

EJEMPLOS_PYME = {
    "Aires acondicionados": "Hacemos mantenimiento preventivo y correctivo de aires acondicionados para oficinas y hospitales",
    "Software a la medida": "Somos una empresa de desarrollo de software a la medida y soporte de aplicaciones web",
    "Refrigerios y eventos": "Preparamos refrigerios, almuerzos y servicio de catering para eventos y capacitaciones",
    "Aseo y cafetería": "Prestamos servicio de aseo, limpieza y cafetería para edificios de oficinas",
}
EJEMPLOS_CONTRATO = {
    "Abogado en 8011": ("Prestación de servicios profesionales como abogado para el acompañamiento jurídico "
                        "de la personería municipal", "80111600"),
    "Contador en 8011": ("Prestar servicios profesionales como contador público para apoyar la elaboración de los "
                         "estados financieros y las conciliaciones contables de la entidad", "80111600"),
    "Código que coincide": ("Suministro de combustible gasolina corriente y ACPM para los vehículos del municipio", "15101500"),
}


def pesos(x) -> str:
    return "" if pd.isna(x) else "$" + f"{x:,.0f}".replace(",", ".")


def enlace(u) -> str:
    """La API entrega el enlace como {"url": "..."}; se deja solo la dirección para que sea clicable."""
    return u.get("url", "") if isinstance(u, dict) else (u or "")


# ---------------------------------------------------------------- Pymes
with tab_pyme:
    st.caption("Ejemplos para probar:")
    for col, (rotulo, texto) in zip(st.columns(len(EJEMPLOS_PYME)), EJEMPLOS_PYME.items()):
        if col.button(rotulo, key=f"ej_{rotulo}", width="stretch"):
            st.session_state["q_pyme"] = texto
            st.session_state["consulta_pyme"] = texto      # un ejemplo busca de una vez
    q = st.text_area("Describa con sus palabras qué vende o qué servicio presta", key="q_pyme",
                     placeholder="Por ejemplo: vendemos e instalamos cámaras de seguridad y alarmas")
    try:
        B = buscador()
        f1, f2 = st.columns(2)
        deptos = f1.multiselect("Departamento de la entidad (opcional)",
                                sorted(d for d in B.p.departamento_entidad.dropna().unique() if d != "No Definido"),
                                placeholder="Todos los departamentos")
        solo_abiertos = f2.checkbox("Solo procesos en estado Abierto", value=False,
                                    help="Por defecto se incluyen los procesos Publicados y los Abiertos.")
    except FileNotFoundError:
        B, deptos, solo_abiertos = None, [], False
    if st.button("Buscar", type="primary"):
        st.session_state["consulta_pyme"] = q
    q = st.session_state.get("consulta_pyme", "")       # la consulta se conserva al cambiar filtros o de pestaña
    if q.strip() and len(preparar(q).split()) < 2:
        st.warning("La descripción es muy corta. Escriba al menos qué vende o qué servicio presta y para quién.")
    elif q.strip():
        qp = preparar(q)
        P = MODELO.predict_proba([qp])[0]
        orden = np.argsort(-P)[:3]
        st.subheader("Categorías sugeridas (familia UNSPSC)")
        st.caption(f"Para: “{q}”")
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
                         hide_index=True, width="stretch",
                         column_config={"objeto_del_contrato": st.column_config.TextColumn("objeto del contrato", width="large"),
                                        "nombre_entidad": "entidad",
                                        "similitud": st.column_config.ProgressColumn("parecido", min_value=0, max_value=1)})
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
                r["precio_base"] = r.precio_base.map(pesos)
                r["urlproceso"] = r.urlproceso.map(enlace)
                r["departamento_entidad"] = r.departamento_entidad.replace("No Definido", "")
                st.dataframe(r[["urlproceso", "texto", "entidad", "departamento_entidad", "precio_base",
                                "estado_del_procedimiento", "familia", "score"]],
                             hide_index=True, width="stretch", height=390,
                             column_config={"urlproceso": st.column_config.LinkColumn("SECOP", display_text="abrir", width="small"),
                                            "texto": st.column_config.TextColumn("proceso", width="large"),
                                            "departamento_entidad": "departamento", "precio_base": "precio base",
                                            "estado_del_procedimiento": "estado",
                                            "score": st.column_config.ProgressColumn("ajuste", min_value=0, max_value=1)})
                st.caption(f"{int(mascara.sum()):,} procesos en la búsqueda. ".replace(",", ".")
                           + "Ajuste = 0,5·similitud semántica + 0,3·similitud de términos + 0,2·probabilidad de la familia. "
                           "“(inferida)” = el proceso venía sin código y la familia la estimó el modelo. "
                           "Los procesos son una foto de octubre de 2026; verifique el estado en SECOP.")

# ---------------------------------------------------------------- Auditoría
with tab_audit:
    st.write("Pegue el objeto de un contrato y el código UNSPSC declarado. "
             "Si no coinciden con confianza alta, el contrato queda **para revisar** (no es una acusación).")
    st.caption("Ejemplos para probar:")
    for col, (rotulo, (texto, cod)) in zip(st.columns(len(EJEMPLOS_CONTRATO)), EJEMPLOS_CONTRATO.items()):
        if col.button(rotulo, key=f"ejc_{rotulo}", width="stretch"):
            st.session_state["obj_contrato"], st.session_state["cod_contrato"] = texto, cod
            st.session_state["revision"] = (texto, cod)
    obj = st.text_area("Objeto del contrato", key="obj_contrato",
                       placeholder="Pegue aquí el objeto tal como aparece en SECOP")
    cod = st.text_input("Código UNSPSC declarado (8 dígitos, con o sin 'V1.')", key="cod_contrato", placeholder="80111600")
    if st.button("Revisar", type="primary"):
        st.session_state["revision"] = (obj, cod)
    obj, cod = st.session_state.get("revision", ("", ""))
    digitos = "".join(ch for ch in cod if ch.isdigit())
    if obj.strip() and len(digitos) < 4:
        st.warning("Escriba el código UNSPSC declarado (al menos los 4 primeros dígitos).")
    elif obj.strip():
        fam_decl = digitos[:4]
        op = preparar(obj)
        P = MODELO.predict_proba([op])[0]
        i = int(np.argmax(P))
        pred, conf = MODELO.classes_[i], float(P[i])
        c1, c2 = st.columns(2)
        c1.metric("Familia declarada", nombre(fam_decl))
        c2.metric("Familia sugerida por el modelo", nombre(pred), f"confianza {conf:.0%}", delta_color="off", delta_arrow="off")
        if fam_decl not in NOMBRES:
            st.warning(f"La familia {fam_decl} no existe en el catálogo UNSPSC v14. Revise el código.")
        elif pred != fam_decl and conf >= UMBRAL_APP:
            st.warning("Posible discrepancia: el objeto del contrato apunta a otra familia. Conviene revisarlo.")
        elif pred != fam_decl:
            st.info("El modelo sugiere otra familia, pero con poca confianza. No se marca.")
        else:
            st.success("La familia declarada coincide con la sugerida.")
        if fam_decl in NOMBRES and fam_decl not in MODELO.classes_:
            st.caption(f"La familia {fam_decl} tiene muy pocos contratos y el modelo no la conoce; no puede sugerirla.")
        aviso_segmento(P)
        st.write("**Términos que explican la sugerencia:**")
        grafico_explicacion(local(MODELO, op, pred, k=8))
        with st.expander("Texto que ve el modelo (sin fórmulas de contratación, lematizado)"):
            st.code(op)

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
        tasa = f"{len(A) / 36844 * 100:.1f}".replace(".", ",")
        st.caption(f"{len(A):,}".replace(",", ".") + f" alertas sobre 36.844 contratos que el modelo no vio al entrenar ({tasa} %). "
                   + "Una persona revisó 100 alertas del modelo anterior: 94 siguen aquí con la misma familia sugerida, y en 83 "
                   "de esas 94 la sugerencia era la correcta. Cerca de la mitad de las alertas son nuevas y nadie las ha revisado.")
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
        st.dataframe(T[["objeto_del_contrato", "nombre_entidad", "declarada", "sugerida", "conf", "id_contrato"]],
                     hide_index=True, width="stretch", height=320,
                     column_config={"conf": st.column_config.ProgressColumn("confianza", min_value=0, max_value=1),
                                    "objeto_del_contrato": st.column_config.TextColumn("objeto del contrato", width="large"),
                                    "nombre_entidad": "entidad",
                                    "id_contrato": st.column_config.TextColumn("id en SECOP", help="Con este id se busca el contrato en SECOP II")})
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
            st.write(f"Entidad: {fila.nombre_entidad} · id en SECOP: `{fila.id_contrato}`")
            st.write(f"Declarada: **{nombre(fila.fam)}** · Sugerida: **{nombre(fila.pred)}** ({fila.conf:.0%})")
            st.write(fila.objeto_del_contrato)
            grafico_explicacion(local(MODELO, preparar(fila.objeto_del_contrato), fila.pred, k=8))

# ---------------------------------------------------------------- Info
with tab_info:
    st.subheader("Qué modelo usa la app y qué tan bien clasifica")
    try:
        dz = json.loads((REPORTS / "destilacion_8011.json").read_text(encoding="utf-8"))["variantes"]
        filas = []
        for k, etiqueta in (("D", "Modelo de la app (8011 corregido por Claude)"), ("B", "Mismo modelo con las etiquetas de SECOP")):
            v = dz[k]
            filas.append({"modelo": etiqueta,
                          "exactitud vs. consenso de Claude (gold)": v["gold"]["C_consenso_claude"]["accuracy"],
                          "top-3 vs. consenso de Claude (gold)": v["gold"]["C_consenso_claude"]["top3"],
                          "exactitud vs. etiqueta humana (gold)": v["gold"]["H_humano"]["accuracy"],
                          "acuerdo con SECOP (test balanceado)": v["test_balanceado"]["accuracy"],
                          "predice 8011 (test real)": v["test_natural"]["predice_8011"]})
        st.dataframe(pd.DataFrame(filas), hide_index=True)
    except FileNotFoundError:
        st.info("Corra scripts/15_destilacion_8011.py")
    st.markdown(
        "- El modelo es lineal (TF-IDF y regresión logística) y clasifica entre 175 familias. Claude **no** predice "
        "en la app.\n"
        "- Las etiquetas de SECOP tienen sesgo: la familia 8011 (recursos humanos) se usa como cajón de sastre. "
        "Contra un gold set de 300 contratos, casi todos los declarados en 8011 corresponden a otra familia "
        "(95 % a 99 % según la referencia).\n"
        "- Por eso, en el entrenamiento, los 3.000 contratos declarados en 8011 los re-etiquetó Claude Sonnet 5.5 "
        "sin ver el código declarado. El resto conserva la etiqueta de SECOP. El modelo resultante casi nunca "
        "sugiere 8011.\n"
        "- La exactitud queda entre 0,27 (etiquetas humanas, poco confiables) y 0,55 (consenso de dos anotaciones "
        "de Claude, que no es un árbitro independiente). El detalle está en `documento_final.pdf`.\n"
        "- Datos: SECOP II (datos.gov.co, CC BY-SA 4.0). Catálogo: UNSPSC v14 en español, Colombia Compra Eficiente.")
