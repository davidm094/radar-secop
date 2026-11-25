"""Radar SECOP – demo en Streamlit.

Ejecutar desde la raíz del repo:  streamlit run app/app.py
Si faltan el modelo o el índice de procesos, los descarga ya construidos (src/radar/demo.py), así que corre
directo desde un clon del repo. Para generarlos en vez de descargarlos: python scripts/preparar_demo.py
"""
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from radar.config import DATA_PROC, REPORTS, MODELO_APP
from radar.data import familias_unspsc, segmentos_unspsc
from radar.demo import asegurar_artefactos
from radar.explain import local, Vecinos
from radar.model import cargar
from radar.search import BuscadorProcesos
from radar.text import preparar

st.set_page_config(page_title="Radar SECOP", page_icon="🧭", layout="wide")
UMBRAL_APP = 0.70   # umbral calibrado (ver reports/explicabilidad.json)
UMBRAL_DUDA = 0.35  # por debajo, la familia es dudosa y se muestra también el segmento (2 dígitos)
N_TEST = 36_844     # contratos de test sobre los que se calculan las alertas
ACENTO, GRIS = "#1F4FD8", "#8793A8"

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
:root {--tinta:#0B1220; --gris:#566074; --linea:#D5DBE5; --fondo:#F2F4F8; --azul:#1F4FD8; --mono:'IBM Plex Mono', ui-monospace, monospace;}
.block-container {max-width: 1200px; padding-top: 1.6rem; padding-bottom: 3rem;}
.rs, .rs * {font-family:'IBM Plex Sans', system-ui, sans-serif;}
.rs-cab {background:var(--tinta); color:#fff; padding:1.7rem 1.9rem 0 1.9rem; border-bottom:4px solid var(--azul);}
.rs-eyebrow {font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#8FB0FF; margin:0;}
.rs-h1 {font-size:2.5rem; font-weight:700; color:#fff; margin:.35rem 0 .45rem 0; line-height:1.05; letter-spacing:-.02em;}
.rs-lead {font-size:1.02rem; color:#C9D2E3; margin:0 0 1.4rem 0; max-width:780px; line-height:1.5;}
.rs-datos {display:grid; grid-template-columns:repeat(4,1fr); border-top:1px solid #2A3550;}
.rs-dato {padding:.8rem 1rem .9rem 0; border-right:1px solid #2A3550; margin-right:1rem;}
.rs-dato:last-child {border-right:0;}
.rs-dato b {display:block; font-family:var(--mono); font-size:1.35rem; font-weight:600; color:#fff; line-height:1.15;}
.rs-dato span {font-size:.8rem; color:#9AA7C0;}
.rs-pasos {display:grid; grid-template-columns:repeat(3,1fr); gap:1.4rem; margin:.5rem 0 1.2rem 0;}
.rs-paso {border-top:2px solid var(--tinta); padding-top:.55rem;}
.rs-num {font-family:var(--mono); font-size:.8rem; font-weight:600; color:var(--azul); letter-spacing:.06em;}
.rs-paso p {margin:.15rem 0 0 0; font-size:.9rem; color:var(--gris); line-height:1.4;}
.rs-paso b {display:block; color:var(--tinta); font-size:1rem; font-weight:600; margin-bottom:.1rem;}
.rs-sec {font-size:1.2rem; font-weight:700; color:var(--tinta); margin:1.7rem 0 .15rem 0; padding-top:.7rem; border-top:1px solid var(--linea); letter-spacing:-.01em;}
.rs-sec i {font-style:normal; font-family:var(--mono); color:var(--azul); font-size:.95rem; margin-right:.55rem;}
.rs-sub {font-size:.9rem; color:var(--gris); margin:0 0 .8rem 0;}
.rs-cards {display:grid; grid-template-columns:repeat(3,1fr); gap:0; border:1px solid var(--linea);}
.rs-cards.dos {grid-template-columns:repeat(2,1fr);}
.rs-card {padding:1rem 1.15rem 1.1rem 1.15rem; background:#fff; border-right:1px solid var(--linea);}
.rs-card:last-child {border-right:0;}
.rs-card.top {background:var(--fondo); box-shadow:inset 0 3px 0 var(--azul);}
.rs-tag {font-family:var(--mono); font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--gris); margin:0;}
.rs-card.top .rs-tag {color:var(--azul);}
.rs-fam {font-size:1rem; font-weight:600; color:var(--tinta); margin:.4rem 0 .7rem 0; line-height:1.3; min-height:2.6em;}
.rs-cod {font-family:var(--mono); color:var(--gris); font-weight:500; font-size:.92rem;}
.rs-pct {word-spacing:-.4em; font-family:var(--mono); font-size:1.9rem; font-weight:600; color:var(--tinta); line-height:1;}
.rs-pct small {word-spacing:normal; font-family:'IBM Plex Sans', sans-serif; font-size:.78rem; font-weight:400; color:var(--gris); margin-left:.45rem;}
.rs-bar {height:4px; background:#DDE2EB; margin-top:.6rem;}
.rs-bar div {height:100%; background:var(--azul);}
.rs-card:not(.top) .rs-bar div {background:#8793A8;}
.rs-kpis {display:grid; grid-template-columns:repeat(4,1fr); border-top:2px solid var(--tinta); border-bottom:1px solid var(--linea); margin:.3rem 0 .8rem 0;}
.rs-kpi {padding:.8rem 1rem .9rem 0; border-right:1px solid var(--linea); margin-right:1rem;}
.rs-kpi:last-child {border-right:0;}
.rs-kpi .v {word-spacing:-.4em; font-family:var(--mono); font-size:1.6rem; font-weight:600; color:var(--tinta); line-height:1.15;}
.rs-kpi .l {font-size:.82rem; color:var(--gris); margin-top:.2rem; line-height:1.35;}
.rs-veredicto {padding:.9rem 1.1rem; margin:.4rem 0 1rem 0; border-left:5px solid; background:var(--fondo);}
.rs-veredicto h4 {margin:0 0 .15rem 0; font-size:1.1rem; font-weight:700; letter-spacing:-.01em;}
.rs-veredicto p {margin:0; font-size:.93rem; color:#2B3445;}
.rs-veredicto.revisar {border-color:#D97706; background:#FFF6E0;} .rs-veredicto.revisar h4 {color:#92400E;}
.rs-veredicto.ok {border-color:#0F7B5F; background:#E8F5F0;} .rs-veredicto.ok h4 {color:#0B5A46;}
.rs-veredicto.neutro {border-color:#8793A8;} .rs-veredicto.neutro h4 {color:var(--tinta);}
.rs-nota {border-left:5px solid var(--tinta); background:var(--fondo); padding:.75rem 1.1rem; font-size:.92rem; color:#2B3445; margin:.3rem 0 1.1rem 0;}
.rs-item {border-bottom:1px solid var(--linea); padding:.55rem 0;}
.rs-item p {margin:0 0 .2rem 0; font-size:.9rem; color:var(--tinta); line-height:1.4;}
.rs-item span {font-size:.8rem; color:var(--gris);}
.rs-vacio {border:1px dashed #B8C1D1; padding:1.5rem; text-align:center; color:var(--gris); margin-top:1rem; font-size:.95rem;}
.rs-pie {border-top:2px solid var(--tinta); margin-top:2.6rem; padding-top:.8rem; font-size:.8rem; color:var(--gris);}
.rs-pie a {color:var(--azul);}
.stTabs [data-baseweb="tab-list"] {gap:1.6rem;}
.stTabs [data-baseweb="tab"] p {font-weight:600; font-size:.95rem;}
@media (max-width: 760px) {.rs-pasos, .rs-cards, .rs-cards.dos {grid-template-columns:1fr;} .rs-kpis, .rs-datos {grid-template-columns:repeat(2,1fr);}
  .rs-card {border-right:0; border-bottom:1px solid var(--linea);} .rs-h1 {font-size:1.9rem;} .rs-cab {padding:1.2rem 1.1rem 0 1.1rem;}}
</style>
""")


# ---------------------------------------------------------------- Recursos
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


@st.cache_resource(show_spinner="Cargando procesos publicados…")
def buscador():
    return BuscadorProcesos.desde_disco(DATA_PROC)


@st.cache_data(show_spinner=False)
def alertas():
    """Contratos del conjunto de test donde el modelo contradice el código declarado con confianza ≥ 0,70."""
    return pd.read_csv(REPORTS / "discrepancias.csv", dtype={"fam": str, "pred": str})


# ---------------------------------------------------------------- Utilidades de presentación
def miles(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def pct(x, dec: int = 0) -> str:
    return f"{x * 100:.{dec}f}".replace(".", ",") + " %"


def pesos(x) -> str:
    return "" if pd.isna(x) else "$" + f"{x:,.0f}".replace(",", ".")


def enlace(u) -> str:
    """La API entrega el enlace como {"url": "..."}; se deja solo la dirección para que sea clicable."""
    return u.get("url", "") if isinstance(u, dict) else (u or "")


def nombre(fam: str) -> str:
    return f"{fam} · {NOMBRES.get(fam, '¿?')}"


def pasos(*items):
    st.html('<div class="rs rs-pasos">' + "".join(
        f'<div class="rs-paso"><span class="rs-num">PASO {i:02d}</span><p><b>{t}</b>{d}</p></div>' for i, (t, d) in enumerate(items, 1)) + "</div>")


def seccion(titulo: str, sub: str = ""):
    num, _, resto = titulo.partition(" · ") if titulo[:1].isdigit() else ("", "", titulo)
    st.html(f'<div class="rs"><p class="rs-sec">{f"<i>{int(num):02d}</i>" if num else ""}{resto}</p>'
            + (f'<p class="rs-sub">{sub}</p>' if sub else "") + "</div>")


def tarjeta_familia(etiqueta: str, fam: str, valor: str, nota: str, barra: float | None, top: bool = False) -> str:
    b = f'<div class="rs-bar"><div style="width:{max(2, round(barra * 100))}%"></div></div>' if barra is not None else ""
    return (f'<div class="rs-card{" top" if top else ""}"><p class="rs-tag">{etiqueta}</p>'
            f'<p class="rs-fam"><span class="rs-cod">{escape(fam)}</span> · {escape(NOMBRES.get(fam, "No está en el catálogo"))}</p>'
            f'<div class="rs-pct">{valor}<small>{nota}</small></div>{b}</div>')


def grafico_barras(serie: pd.Series, titulo_x: str, color: str = GRIS):
    d = serie.rename_axis("nombre").reset_index(name="n")
    st.altair_chart(
        alt.Chart(d).mark_bar(color=color, cornerRadiusEnd=3).encode(
            x=alt.X("n:Q", title=titulo_x, axis=alt.Axis(grid=False, tickMinStep=1)),
            y=alt.Y("nombre:N", sort="-x", title=None, axis=alt.Axis(labelLimit=330), scale=alt.Scale(paddingInner=0.35)),
            tooltip=[alt.Tooltip("nombre:N", title=""), alt.Tooltip("n:Q", title=titulo_x)],
        ).properties(height=alt.Step(28)).configure_view(stroke=None), width="stretch")


def grafico_explicacion(exp: pd.DataFrame):
    """Barras con la contribución exacta de cada término al puntaje de la familia sugerida."""
    if exp.empty:
        st.info("El texto no tiene palabras que el modelo conozca.")
        return
    d = exp.assign(sentido=np.where(exp.contribucion >= 0, "A favor", "En contra"))
    st.altair_chart(
        alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("contribucion:Q", title="Cuánto empuja cada palabra hacia la categoría sugerida", axis=alt.Axis(grid=False)),
            y=alt.Y("termino:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12), scale=alt.Scale(paddingInner=0.35)),
            color=alt.Color("sentido:N", scale=alt.Scale(domain=["A favor", "En contra"], range=[ACENTO, GRIS]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("termino:N", title="Palabra"), alt.Tooltip("contribucion:Q", title="Contribución", format="+.2f")],
        ).properties(height=alt.Step(28)).configure_view(stroke=None), width="stretch")
    st.caption("Las palabras ya están en su forma base y sin tildes, que es como las lee el modelo. "
               "Cada barra es la contribución exacta de esa palabra, no una aproximación.")


def segmento_mas_probable(P: np.ndarray):
    """Suma las probabilidades de las familias de cada segmento (primeros 2 dígitos)."""
    s = pd.Series(P, index=pd.Index(MODELO.classes_).str[:2]).groupby(level=0).sum()
    return s.idxmax(), float(s.max())


def aviso_segmento(P: np.ndarray):
    if P.max() < UMBRAL_DUDA:
        seg, ps = segmento_mas_probable(P)
        st.info(f"El modelo no está seguro de la categoría (la mejor tiene {pct(P.max())}). "
                f"A nivel más general, lo más probable es el segmento **{seg} · {SEGMENTOS.get(seg, '¿?')}** ({pct(ps)}). "
                "Pruebe a describir con más detalle qué vende o qué servicio presta.", icon=":material/help:")


with st.spinner("Preparando la demo. La primera vez descarga el modelo y el índice (uno o dos minutos)…"):
    asegurar_artefactos(lambda mensaje: None)   # copia los datos congelados y descarga lo que falte
MODELO, NOMBRES, VECINOS, SEGMENTOS = recursos()
try:
    A = alertas()
except FileNotFoundError:
    A = None

# ---------------------------------------------------------------- Encabezado
st.html(f"""
<div class="rs rs-cab">
<p class="rs-eyebrow">Contratación pública de Colombia / SECOP II</p>
<h1 class="rs-h1">Radar SECOP</h1>
<p class="rs-lead">Lee el texto de un contrato o la descripción de un negocio y dice a qué categoría del Estado
corresponde, con las palabras que lo explican. Sirve para encontrar procesos donde vender y para revisar si un
contrato está bien clasificado.</p>
<div class="rs-datos">
  <div class="rs-dato"><b>148.473</b><span>procesos publicados</span></div>
  <div class="rs-dato"><b>175</b><span>categorías (familias UNSPSC)</span></div>
  <div class="rs-dato"><b>{miles(len(A)) if A is not None else "–"}</b><span>contratos para revisar</span></div>
  <div class="rs-dato"><b>OCT 2026</b><span>fecha de los datos</span></div>
</div></div>""")

tab_pyme, tab_audit, tab_alertas, tab_info = st.tabs(
    [":material/search: Para pymes", ":material/fact_check: Revisar un contrato",
     ":material/flag: Alertas para veedurías", ":material/info: Cómo funciona"])

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

# ---------------------------------------------------------------- Pymes
with tab_pyme:
    st.write("**Para empresas que quieren venderle al Estado.** No necesita conocer los códigos de contratación: "
             "describa su negocio y la herramienta hace el resto.")
    pasos(("Describa su negocio", "Con sus palabras: qué vende o qué servicio presta, y para quién."),
          ("Vea en qué categoría encaja", "Las tres categorías más probables y las palabras que lo explican."),
          ("Abra los procesos en SECOP", "Diez procesos publicados que se parecen a lo que usted ofrece."))
    with st.container(border=True):
        st.caption("¿Quiere verlo funcionar primero? Elija un ejemplo:")
        for col, (rotulo, texto) in zip(st.columns(len(EJEMPLOS_PYME)), EJEMPLOS_PYME.items()):
            if col.button(rotulo, key=f"ej_{rotulo}", width="stretch"):
                st.session_state["q_pyme"] = texto
                st.session_state["consulta_pyme"] = texto      # un ejemplo busca de una vez
        q = st.text_area("Describa qué vende o qué servicio presta", key="q_pyme", height=90,
                         placeholder="Por ejemplo: vendemos e instalamos cámaras de seguridad y alarmas para conjuntos residenciales")
        try:
            B = buscador()
            f1, f2, f3 = st.columns([3, 2, 1.3], vertical_alignment="bottom")
            deptos = f1.multiselect("Departamento de la entidad (opcional)",
                                    sorted(d for d in B.p.departamento_entidad.dropna().unique() if d != "No Definido"),
                                    placeholder="Todos los departamentos")
            solo_abiertos = f2.checkbox("Solo procesos en estado Abierto", value=False,
                                        help="Si no lo marca, se incluyen los procesos Publicados y los Abiertos.")
            buscar = f3.button("Buscar procesos", type="primary", width="stretch", icon=":material/search:")
        except FileNotFoundError:
            B, deptos, solo_abiertos = None, [], False
            buscar = st.button("Buscar procesos", type="primary", icon=":material/search:")
    if buscar:
        st.session_state["consulta_pyme"] = q
    q = st.session_state.get("consulta_pyme", "")       # la consulta se conserva al cambiar filtros o de pestaña
    if not q.strip():
        st.html('<div class="rs rs-vacio">Escriba la descripción de su negocio o elija un ejemplo para ver los resultados.</div>')
    elif len(preparar(q).split()) < 2:
        st.warning("La descripción es muy corta. Escriba al menos qué vende o qué servicio presta y para quién.",
                   icon=":material/edit:")
    else:
        qp = preparar(q)
        P = MODELO.predict_proba([qp])[0]
        orden = np.argsort(-P)[:3]
        seccion("1 · Su negocio encaja en estas categorías", f"Para: “{escape(q)}”")
        rotulos = ["Más probable", "Segunda opción", "Tercera opción"]
        st.html('<div class="rs rs-cards">' + "".join(
            tarjeta_familia(rotulos[k], MODELO.classes_[i], pct(P[i]), "de probabilidad", float(P[i]), top=(k == 0))
            for k, i in enumerate(orden)) + "</div>")
        aviso_segmento(P)
        with st.expander("¿Por qué la categoría más probable?", expanded=True, icon=":material/lightbulb:"):
            e1, e2 = st.columns([1, 1], gap="large")
            with e1:
                st.write("**Palabras de su descripción que más pesaron**")
                grafico_explicacion(local(MODELO, qp, MODELO.classes_[orden[0]], k=8))
            with e2:
                st.write("**Contratos reales parecidos al suyo**")
                v = VECINOS.buscar(qp, k=3)
                st.html("".join(
                    f'<div class="rs rs-item"><p>{escape(o[:230])}{"…" if len(o) > 230 else ""}</p>'
                    f'<span>{escape(e)} · declarado en {escape(nombre(f))}</span></div>'
                    for o, e, f in zip(v.objeto_del_contrato, v.nombre_entidad, v.fam)))
                st.caption("Son contratos ya firmados, con la categoría que declaró la entidad.")
        if B is None:
            seccion("2 · Procesos publicados que le pueden servir")
            st.warning("Falta el índice de procesos: corra scripts/07_indexar_procesos.py")
        else:
            probs = {MODELO.classes_[i]: float(P[i]) for i in orden}
            mascara = np.ones(len(B.p), dtype=bool)
            if deptos:
                mascara &= B.p.departamento_entidad.isin(deptos).values
            if solo_abiertos:
                mascara &= (B.p.estado_del_procedimiento == "Abierto").values
            r = B.buscar(q, qp, probs, k=10, mascara=mascara)
            seccion("2 · Procesos publicados que le pueden servir",
                    f"Los {len(r)} que mejor encajan entre {miles(mascara.sum())} procesos. "
                    "Use el enlace de la primera columna para abrir cada uno en SECOP II.")
            if r.empty:
                st.info("No hay procesos con esos filtros. Quite alguno y vuelva a buscar.")
            else:
                r["familia"] = r.fam.map(nombre) + np.where(r.familia_inferida, " (estimada)", "")
                r["precio_base"] = r.precio_base.map(pesos)
                r["urlproceso"] = r.urlproceso.map(enlace)
                r["departamento_entidad"] = r.departamento_entidad.replace("No Definido", "")
                st.dataframe(r[["urlproceso", "texto", "entidad", "departamento_entidad", "precio_base",
                                "estado_del_procedimiento", "familia", "score"]],
                             hide_index=True, width="stretch", height=390,
                             column_config={"urlproceso": st.column_config.LinkColumn("SECOP", display_text="Abrir ↗", width="small"),
                                            "texto": st.column_config.TextColumn("Proceso", width="large"),
                                            "entidad": "Entidad", "departamento_entidad": "Departamento",
                                            "precio_base": "Precio base", "estado_del_procedimiento": "Estado",
                                            "familia": "Categoría",
                                            "score": st.column_config.ProgressColumn(
                                                "Ajuste", min_value=0, max_value=1, format="percent",
                                                help="Qué tanto se parece el proceso a su descripción: combina significado, "
                                                     "palabras en común y categoría.")})
                st.caption("“(estimada)” quiere decir que el proceso venía sin código y la categoría la calculó el modelo. "
                           "Los procesos son una foto de octubre de 2026: verifique en SECOP que sigan abiertos.")

# ---------------------------------------------------------------- Auditoría
with tab_audit:
    st.write("**Para veedurías y equipos de control interno.** Compruebe si el código con que se registró un contrato "
             "corresponde a lo que dice su objeto.")
    pasos(("Pegue el objeto del contrato", "Tal como aparece en SECOP II."),
          ("Escriba el código declarado", "El código UNSPSC de 8 dígitos con que se registró."),
          ("Lea el resultado", "Si el texto apunta a otra categoría con confianza alta, queda para revisar."))
    with st.container(border=True):
        st.caption("¿Quiere verlo funcionar primero? Elija un ejemplo:")
        for col, (rotulo, (texto, cod)) in zip(st.columns(len(EJEMPLOS_CONTRATO)), EJEMPLOS_CONTRATO.items()):
            if col.button(rotulo, key=f"ejc_{rotulo}", width="stretch"):
                st.session_state["obj_contrato"], st.session_state["cod_contrato"] = texto, cod
                st.session_state["revision"] = (texto, cod)
        obj = st.text_area("Objeto del contrato", key="obj_contrato", height=90,
                           placeholder="Pegue aquí el objeto tal como aparece en SECOP")
        c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
        cod = c1.text_input("Código UNSPSC declarado", key="cod_contrato", placeholder="80111600",
                            help="8 dígitos, con o sin el prefijo V1. Los 4 primeros son la familia.")
        if c2.button("Revisar contrato", type="primary", width="stretch", icon=":material/fact_check:"):
            st.session_state["revision"] = (obj, cod)
    obj, cod = st.session_state.get("revision", ("", ""))
    digitos = "".join(ch for ch in cod.replace("V1.", "").replace("v1.", "") if ch.isdigit())
    if not obj.strip():
        st.html('<div class="rs rs-vacio">Pegue un contrato o elija un ejemplo para ver la revisión.</div>')
    elif len(digitos) < 4:
        st.warning("Escriba el código UNSPSC declarado (al menos los 4 primeros dígitos).", icon=":material/edit:")
    else:
        fam_decl = digitos[:4]
        op = preparar(obj)
        P = MODELO.predict_proba([op])[0]
        i = int(np.argmax(P))
        pred, conf = MODELO.classes_[i], float(P[i])
        seccion("Resultado de la revisión")
        if fam_decl not in NOMBRES:
            clase, titulo, texto = "neutro", "El código no existe en el catálogo", \
                f"La familia {escape(fam_decl)} no está en el catálogo UNSPSC v14. Revise que el código esté bien escrito."
        elif pred != fam_decl and conf >= UMBRAL_APP:
            clase, titulo, texto = "revisar", "Para revisar", \
                ("El objeto del contrato apunta a una categoría distinta de la declarada, con confianza alta. "
                 "No es una acusación ni prueba un error: es una señal para que una persona lo mire.")
        elif pred != fam_decl:
            clase, titulo, texto = "neutro", "Sin señal suficiente", \
                (f"El modelo sugiere otra categoría, pero con poca confianza ({pct(conf)}, por debajo de {pct(UMBRAL_APP)}). "
                 "No se marca para revisión.")
        else:
            clase, titulo, texto = "ok", "El código coincide", "La categoría declarada es la misma que sugiere el texto del contrato."
        st.html(f'<div class="rs rs-veredicto {clase}"><h4>{titulo}</h4><p>{texto}</p></div>')
        st.html('<div class="rs rs-cards dos">'
                + tarjeta_familia("Categoría declarada", fam_decl, escape(digitos[:8]), "código registrado", None)
                + tarjeta_familia("Categoría que sugiere el texto", pred, pct(conf), "de confianza", conf, top=True) + "</div>")
        if fam_decl in NOMBRES and fam_decl not in MODELO.classes_:
            st.caption(f"La familia {fam_decl} tiene muy pocos contratos y el modelo no la conoce; no puede sugerirla.")
        aviso_segmento(P)
        seccion("Por qué el modelo sugiere esa categoría", "Términos que explican la sugerencia.")
        grafico_explicacion(local(MODELO, op, pred, k=8))
        with st.expander("Ver el texto tal como lo lee el modelo"):
            st.caption("Se quitan las fórmulas de contratación (“prestación de servicios profesionales”, “apoyo a la gestión”) "
                       "porque dicen cómo se contrató, no qué servicio se recibe. Luego cada palabra se lleva a su forma base.")
            st.code(op, language=None)

# ---------------------------------------------------------------- Alertas
with tab_alertas:
    st.write("**Para veedurías.** Lista de contratos cuyo objeto apunta a una categoría distinta de la declarada, "
             f"con confianza de {pct(UMBRAL_APP)} o más.")
    if A is None:
        st.info("Falta la lista de alertas: corra scripts/06_discrepancias.py")
    else:
        st.html('<div class="rs rs-nota"><b>Son contratos para revisar, no irregularidades.</b> Una discrepancia no prueba un '
                'error: muchas vienen de la costumbre de registrar casi todo contrato de servicios en la familia 8011 '
                '(recursos humanos), cuando la guía oficial pide registrar el servicio que recibe la entidad.</div>')
        st.html(f"""<div class="rs rs-kpis">
<div class="rs-kpi"><div class="v">{miles(len(A))}</div><div class="l">alertas entre {miles(N_TEST)} contratos analizados ({pct(len(A) / N_TEST, 1)})</div></div>
<div class="rs-kpi"><div class="v">{miles(A.nombre_entidad.nunique())}</div><div class="l">entidades con al menos una alerta</div></div>
<div class="rs-kpi"><div class="v">{pct((A.fam == "8011").mean())}</div><div class="l">de las alertas están declaradas en 8011</div></div>
<div class="rs-kpi"><div class="v">83 de 94</div><div class="l">alertas revisadas por una persona en las que la sugerencia era correcta</div></div>
</div>""")
        st.caption("La revisión humana cubre cerca de la mitad de las alertas: 1.437 son nuevas y nadie las ha revisado todavía.")
        seccion("Explore las alertas", "Filtre por entidad, por categoría sugerida o por confianza. La lista se puede descargar.")
        with st.container(border=True):
            a1, a2, a3 = st.columns([2, 2, 1.2])
            por_entidad = A.nombre_entidad.value_counts()
            ent = a1.selectbox("Entidad", ["Todas"] + [f"{e} ({n})" for e, n in por_entidad.items()])
            sug = a2.selectbox("Categoría sugerida", ["Todas"] + [nombre(f) for f in A.pred.value_counts().index])
            cmin = a3.slider("Confianza mínima", 70, 90, 70, 5, format="%d %%") / 100
        V = A[A.conf >= cmin]
        if ent != "Todas":
            V = V[V.nombre_entidad == ent.rsplit(" (", 1)[0]]
        if sug != "Todas":
            V = V[V.pred == sug.split(" · ")[0]]
        T = V.assign(declarada=V.fam.map(nombre), sugerida=V.pred.map(nombre))
        t1, t2 = st.columns([3, 1], vertical_alignment="center")
        t1.write(f"**{miles(len(V))} alertas** en {miles(V.nombre_entidad.nunique())} entidades con estos filtros.")
        t2.download_button("Descargar lista (CSV)", T.to_csv(index=False).encode("utf-8-sig"), "alertas_radar_secop.csv",
                           "text/csv", width="stretch", icon=":material/download:")
        st.dataframe(T[["objeto_del_contrato", "nombre_entidad", "declarada", "sugerida", "conf", "id_contrato"]],
                     hide_index=True, width="stretch", height=320,
                     column_config={"conf": st.column_config.ProgressColumn("Confianza", min_value=0, max_value=1, format="percent"),
                                    "objeto_del_contrato": st.column_config.TextColumn("Objeto del contrato", width="large"),
                                    "nombre_entidad": "Entidad", "declarada": "Categoría declarada", "sugerida": "Categoría sugerida",
                                    "id_contrato": st.column_config.TextColumn("Id en SECOP", help="Con este id se busca el contrato en SECOP II")})
        if len(V):
            g1, g2 = st.columns(2, gap="large")
            with g1:
                st.write("**Entidades con más alertas**")
                grafico_barras(V.nombre_entidad.str.slice(0, 48).value_counts().head(8), "Alertas")
            with g2:
                st.write("**De qué categoría a cuál**")
                grafico_barras((T.declarada.str.slice(0, 4) + " → " + T.sugerida.str.slice(0, 40)).value_counts().head(8),
                               "Alertas", color=ACENTO)
            seccion("¿Por qué se marcó un contrato?", "Elija un contrato de la lista para ver las palabras que explican la alerta.")
            op = st.selectbox("Contrato", V.index, label_visibility="collapsed",
                              format_func=lambda k: f"{pct(V.conf[k])} · {V.objeto_del_contrato[k][:120]}")
            fila = V.loc[op]
            d1, d2 = st.columns([1, 1], gap="large")
            with d1:
                st.html('<div class="rs rs-cards dos">'
                        + tarjeta_familia("Declarada", fila.fam, "", "", None)
                        + tarjeta_familia("Sugerida", fila.pred, pct(fila.conf), "de confianza", float(fila.conf), top=True) + "</div>")
                st.caption(f"{fila.nombre_entidad} · id en SECOP: {fila.id_contrato}")
                st.write(fila.objeto_del_contrato)
            with d2:
                grafico_explicacion(local(MODELO, preparar(fila.objeto_del_contrato), fila.pred, k=8))

# ---------------------------------------------------------------- Info
with tab_info:
    seccion("Qué hace Radar SECOP")
    pasos(("Lee el texto", "Quita las fórmulas de contratación y deja las palabras que describen el bien o servicio."),
          ("Lo clasifica", "Un modelo lineal elige entre 175 familias del clasificador UNSPSC que usa el Estado colombiano."),
          ("Lo explica", "Muestra cuánto pesó cada palabra. Es la cuenta exacta que hace el modelo, no una aproximación."))
    i1, i2 = st.columns(2, gap="large")
    with i1:
        st.write("**Por qué hace falta**")
        st.markdown(
            "- Toda compra pública se publica en SECOP II con un código de categoría. Los buscadores y las alertas filtran "
            "por ese código.\n"
            "- El código suele estar mal puesto. La familia 8011 (recursos humanos) se usa como cajón de sastre: en una "
            "muestra de 300 contratos revisados, entre el 95 % y el 99 % de los declarados en 8011 corresponde a otra familia.\n"
            "- Un código mal puesto esconde el contrato de quien lo busca y de quien lo vigila.")
    with i2:
        st.write("**Qué tan confiable es**")
        st.markdown(
            "- La categoría correcta está entre las tres sugeridas en 43 % a 83 % de los casos, según con qué referencia se "
            "mida. La primera sugerencia acierta entre 27 % y 55 %.\n"
            "- El rango es amplio porque no hay una referencia confiable: dos personas que anotaron los mismos contratos "
            "coincidieron en el 22 %.\n"
            "- Por eso la herramienta muestra tres opciones, explica cada una y nunca decide sola.")
    seccion("El modelo en cifras", "Medido sobre un conjunto de 300 contratos anotados aparte y sobre contratos que el modelo no vio.")
    try:
        dz = json.loads((REPORTS / "destilacion_8011.json").read_text(encoding="utf-8"))["variantes"]
        d_, b_ = dz["D"], dz["B"]
        filas = [("Acierta la categoría, según el consenso de dos anotaciones de Claude", "gold", "C_consenso_claude", "accuracy"),
                 ("La categoría correcta está entre las 3 sugeridas, según ese consenso", "gold", "C_consenso_claude", "top3"),
                 ("Acierta la categoría, según la anotación humana", "gold", "H_humano", "accuracy"),
                 ("Coincide con el código de SECOP, en una muestra balanceada", "test_balanceado", "accuracy", None),
                 ("Veces que sugiere la familia 8011", "test_natural", "predice_8011", None)]
        val = lambda v, f: pct(v[f[1]][f[2]][f[3]] if f[3] else v[f[1]][f[2]], 1)
        st.dataframe(pd.DataFrame({"Medida": [f[0] for f in filas], "Modelo de la app": [val(d_, f) for f in filas],
                                   "Mismo modelo con las etiquetas originales de SECOP": [val(b_, f) for f in filas]}),
                     hide_index=True, width="stretch",
                     column_config={"Medida": st.column_config.TextColumn(width="large")})
    except FileNotFoundError:
        st.info("Corra scripts/15_destilacion_8011.py")
    st.markdown(
        "- **Modelo.** TF-IDF de palabras y pares de palabras, con regresión logística. Corre sin GPU.\n"
        "- **Entrenamiento.** 52.396 contratos de SECOP II. Los 3.000 declarados en 8011 los re-etiquetó Claude Sonnet 5.5 "
        "sin ver el código declarado; el resto conserva la etiqueta de SECOP. Claude **no** responde en esta app.\n"
        "- **Búsqueda.** Combina significado (embeddings), palabras en común (TF-IDF) y la categoría sugerida. "
        "En 30 consultas juzgadas a mano, unos 3 de los 5 primeros procesos le servían al negocio.\n"
        "- **Límites.** Cerca de la mitad de las alertas no tiene revisión humana, y el modelo casi nunca sugiere 8011, "
        "así que no reconoce un contrato que sí sea de personal temporal.")

st.html('<div class="rs rs-pie">Proyecto final de Procesamiento de Lenguaje Natural, Pontificia Universidad Javeriana · '
        'David Moreno y Dierick Brochero · Datos: SECOP II (datos.gov.co, CC BY-SA 4.0) y clasificador UNSPSC v14 de '
        'Colombia Compra Eficiente · <a href="https://github.com/davidm094/radar-secop" target="_blank">Código y documento en GitHub</a></div>')
