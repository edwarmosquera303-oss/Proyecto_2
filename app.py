from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from proyectos import NOMBRE, UNIVERSIDAD, TEMAS, PROYECTOS


BASE_DIR = Path(__file__).parent
CARPETA_IMAGENES = BASE_DIR / "imagenes"
RUTA_LOGO = CARPETA_IMAGENES / "logo-EM.png"
PROPORCION = 16 / 9
COLUMNAS = 2

FONDO = "#17110E"
SUPERFICIE = "#241A15"
SUPERFICIE_2 = "#302219"
COBRE = "#C78652"
BRONCE = "#E2B98A"
TEXTO = "#F4EDE6"
TEXTO_SUAVE = "#BDAFA3"
BORDE = "#493528"

st.set_page_config(
    page_title="Edwar Ferney Mosquera | Data & Machine Learning",
    page_icon=str(RUTA_LOGO) if RUTA_LOGO.is_file() else "📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    :root {{
        --fondo: {FONDO};
        --superficie: {SUPERFICIE};
        --cobre: {COBRE};
        --bronce: {BRONCE};
        --texto: {TEXTO};
        --texto-suave: {TEXTO_SUAVE};
        --borde: {BORDE};
    }}
    .stApp {{ background: var(--fondo); color: var(--texto); }}
    [data-testid="stHeader"] {{ background: rgba(23,17,14,.92); }}
    [data-testid="stToolbar"] {{ right: 1rem; }}
    .block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }}
    h1, h2, h3 {{ color: var(--texto) !important; letter-spacing: -0.035em; }}
    h1 {{ line-height: 1.08; }}
    p, label, [data-testid="stMarkdownContainer"] {{ color: var(--texto); }}
    .eyebrow {{ color: var(--bronce); text-transform: uppercase; letter-spacing: .16em;
                font-size: .76rem; font-weight: 700; margin: 0 0 .55rem 0; }}
    .hero-copy {{ color: var(--texto-suave); font-size: 1.04rem; line-height: 1.75;
                  max-width: 760px; }}
    .hero-panel {{ background: linear-gradient(135deg, #2b1e16 0%, #17110e 70%);
                   border: 1px solid var(--borde); border-radius: 22px;
                   padding: 1.5rem; margin: .5rem 0 1.5rem 0; }}
    .brand-name {{ color: var(--bronce); font-size: .88rem; letter-spacing: .15em;
                   text-transform: uppercase; font-weight: 700; margin-bottom: .4rem; }}
    .brand-subtitle {{ color: var(--texto-suave); font-size: .86rem; letter-spacing: .05em; }}
    .tema {{ color: var(--bronce); font-size: .78rem; text-transform: uppercase;
             letter-spacing: .12em; font-weight: 700; margin: .8rem 0 .35rem 0; }}
    .tag {{ display:inline-block; color: #E8C9A9; background: #34251B;
            border: 1px solid #57402E; border-radius: 999px; padding: .22rem .6rem;
            margin: 0 .3rem .35rem 0; font-size: .76rem; }}
    .sin-imagen {{ aspect-ratio: 16/9; border: 1px solid var(--borde); border-radius: 12px;
                   display:flex; align-items:center; justify-content:center; color:var(--bronce);
                   font-weight:600; background: radial-gradient(circle at 70% 30%, #493020, #211711 65%); }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(180deg, rgba(48,34,25,.75), rgba(36,26,21,.85));
        border: 1px solid var(--borde); border-radius: 16px; transition: border-color .2s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{ border-color: var(--cobre); }}
    div.stButton > button, div.stLinkButton > a {{
        border-radius: 10px; border: 1px solid var(--borde); min-height: 2.65rem;
        font-weight: 600; transition: all .18s ease;
    }}
    div.stButton > button[kind="primary"], div.stLinkButton > a[kind="primary"] {{
        background: var(--cobre); color: #1b120c; border-color: var(--cobre);
    }}
    div.stButton > button:hover, div.stLinkButton > a:hover {{
        border-color: var(--bronce); color: var(--bronce);
    }}
    div[data-testid="stMetric"] {{ background: var(--superficie); border: 1px solid var(--borde);
                                    padding: 1rem; border-radius: 14px; }}
    div[data-testid="stMetricLabel"] p {{ color: var(--texto-suave) !important; }}
    div[data-testid="stMetricValue"] {{ color: var(--bronce) !important; }}
    div[role="radiogroup"] {{ gap: .5rem; }}
    div[role="radiogroup"] label {{ background: #241a15; border: 1px solid var(--borde);
                                     border-radius: 10px; padding: .35rem .75rem; }}
    hr {{ border-color: var(--borde); }}
    .section-note {{ color: var(--texto-suave); margin-top: -.4rem; margin-bottom: 1rem; }}
    .footer {{ color: #9D8A7B; font-size: .8rem; text-align: center; padding-top: .8rem; }}
    @media (max-width: 700px) {{
        .block-container {{ padding-top: 1rem; }}
        .hero-panel {{ padding: 1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cargar_imagen(nombre):
    """Carga una imagen local y la recorta al centro para mantener tarjetas uniformes."""
    ruta = CARPETA_IMAGENES / nombre
    if not ruta.is_file():
        return None
    try:
        img = Image.open(ruta).convert("RGB")
        w, h = img.size
        if w / h > PROPORCION:
            nuevo_w = int(h * PROPORCION)
            izq = (w - nuevo_w) // 2
            img = img.crop((izq, 0, izq + nuevo_w, h))
        else:
            nuevo_h = int(w / PROPORCION)
            sup = (h - nuevo_h) // 2
            img = img.crop((0, sup, w, sup + nuevo_h))
        return img.resize((800, 450))
    except (OSError, ValueError):
        return None


def mostrar_imagen(proyecto):
    ref = proyecto.get("imagen", "")
    if isinstance(ref, str) and ref.startswith(("http://", "https://")):
        st.image(ref, use_container_width=True)
        return
    img = cargar_imagen(ref) if ref else None
    if img is not None:
        st.image(img, use_container_width=True)
    else:
        st.markdown(
            '<div class="sin-imagen">Visualización del proyecto</div>',
            unsafe_allow_html=True,
        )


def estilizar_grafica(fig, ax):
    """Aplica el tema cobre/café a las gráficas de Matplotlib."""
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.tick_params(colors=TEXTO_SUAVE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDE)
    ax.grid(True, color=BORDE, alpha=0.55, linewidth=0.7)
    ax.xaxis.label.set_color(TEXTO_SUAVE)
    ax.yaxis.label.set_color(TEXTO_SUAVE)
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor(SUPERFICIE)
        legend.get_frame().set_edgecolor(BORDE)
        for label in legend.get_texts():
            label.set_color(TEXTO)


# -----------------------------------------------------------------------------
# Encabezado / marca personal
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
    logo_col, text_col = st.columns([1, 4], vertical_alignment="center")
    with logo_col:
        if RUTA_LOGO.is_file():
            st.image(str(RUTA_LOGO), use_container_width=True)
        else:
            st.markdown("<div class='sin-imagen'>JP</div>", unsafe_allow_html=True)
    with text_col:
        st.markdown('<p class="brand-name">Edwar Mosquera · Data & Machine Learning</p>', unsafe_allow_html=True)
        st.title("Convirtiendo datos en soluciones inteligentes.")
        st.markdown(
            '<p class="hero-copy">Exploro datos, desarrollo experimentos de aprendizaje automático '
            'y construyo soluciones con una mentalidad de ingeniería de software. Este espacio reúne '
            'proyectos, visualizaciones y aplicaciones que convierten problemas en oportunidades de análisis.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="brand-subtitle">PYTHON&nbsp;&nbsp;·&nbsp;&nbsp; DATA ANALYTICS&nbsp;&nbsp;·&nbsp;&nbsp; MACHINE LEARNING</p>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"{escape(NOMBRE)} · {escape(UNIVERSIDAD)}")


# -----------------------------------------------------------------------------
# Mini-Lab interactivo: se conserva la lógica de regresión y clasificación.
# -----------------------------------------------------------------------------
st.markdown('<p class="eyebrow">Experimenta con los modelos</p>', unsafe_allow_html=True)
st.header("Mini-Lab predictivo")
st.markdown(
    '<p class="section-note">Ajusta los parámetros y observa cómo cambian las predicciones y las métricas.</p>',
    unsafe_allow_html=True,
)

tipo_modelo = st.radio(
    "Selecciona el experimento",
    ["Regresión (Ajuste de Línea)", "Clasificación (Frontera de Decisión)"],
    horizontal=True,
)

np.random.seed(42)

if tipo_modelo == "Regresión (Ajuste de Línea)":
    x = np.linspace(0, 10, 50)
    y_real = 2 * x + 1 + np.random.normal(0, 2, 50)

    c1, c2 = st.columns(2)
    with c1:
        pendiente = st.slider("Pendiente (m)", 0.0, 4.0, 1.8, step=0.1)
    with c2:
        intercepto = st.slider("Intercepto (b)", -5.0, 5.0, 1.5, step=0.2)

    y_pred = pendiente * x + intercepto
    mse = np.mean((y_real - y_pred) ** 2)
    st.metric(label="Error cuadrático medio (MSE)", value=f"{mse:.2f}")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.scatter(x, y_real, color=BRONCE, edgecolors=FONDO, linewidths=0.5, alpha=0.9, label="Datos observados")
    ax.plot(x, y_pred, color=COBRE, linewidth=2.8, label="Línea del modelo")
    ax.set_xlabel("Variable X")
    ax.set_ylabel("Variable Y")
    ax.legend(frameon=True)
    estilizar_grafica(fig, ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    cluster1 = np.random.normal(loc=3, scale=1.0, size=(25, 2))
    cluster2 = np.random.normal(loc=7, scale=1.0, size=(25, 2))
    umbral = st.slider("Umbral de decisión (eje X)", 1.0, 9.0, 5.0, step=0.1)

    aciertos_c1 = np.sum(cluster1[:, 0] < umbral)
    aciertos_c2 = np.sum(cluster2[:, 0] >= umbral)
    accuracy = (aciertos_c1 + aciertos_c2) / 50.0
    st.metric(label="Exactitud del umbral (accuracy)", value=f"{accuracy * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], color=BRONCE, edgecolors=FONDO, label="Clase A", alpha=0.9)
    ax.scatter(cluster2[:, 0], cluster2[:, 1], color=COBRE, edgecolors=FONDO, label="Clase B", alpha=0.9)
    ax.axvline(umbral, color="#E9D7C5", linestyle="--", linewidth=2, label="Frontera de decisión")
    ax.set_xlabel("Característica X")
    ax.set_ylabel("Característica Y")
    ax.legend(frameon=True)
    estilizar_grafica(fig, ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Catálogo de proyectos: conserva TEMAS, PROYECTOS y sus campos actuales.
# -----------------------------------------------------------------------------
st.divider()
st.markdown('<p class="eyebrow">Trabajo aplicado</p>', unsafe_allow_html=True)
st.header("Proyectos seleccionados")
st.markdown(
    '<p class="section-note">Explora proyectos de análisis, visualización, algoritmos y aprendizaje automático.</p>',
    unsafe_allow_html=True,
)

filtro = st.radio(
    "Filtrar por tema",
    ["Todos"] + list(TEMAS),
    horizontal=True,
    label_visibility="collapsed",
)
visibles = [p for p in PROYECTOS if filtro == "Todos" or p["tema"] == filtro]
st.caption(f"{len(visibles)} proyecto(s) en esta vista")

if not visibles:
    st.info("Aún no hay proyectos en este tema. Asígnalo en el campo 'tema' de proyectos.py.")

for inicio in range(0, len(visibles), COLUMNAS):
    columnas = st.columns(COLUMNAS, gap="large")
    for col, (i, proyecto) in zip(
        columnas,
        enumerate(visibles[inicio : inicio + COLUMNAS], start=inicio),
    ):
        with col, st.container(border=True):
            mostrar_imagen(proyecto)
            st.markdown(f'<p class="tema">{escape(str(proyecto["tema"]))}</p>', unsafe_allow_html=True)
            st.subheader(proyecto["titulo"])
            st.write(proyecto["desc"])
            tags = proyecto.get("tags", [])
            if tags:
                st.markdown(
                    "".join(f'<span class="tag">{escape(str(tag))}</span>' for tag in tags),
                    unsafe_allow_html=True,
                )

            b1, b2 = st.columns(2)
            with b1:
                if proyecto.get("repo"):
                    st.link_button("↗ Ver código", proyecto["repo"], type="primary", use_container_width=True)
                else:
                    st.button(
                        "Ver código", disabled=True, key=f"repo_{i}",
                        help="Falta el enlace del repositorio", use_container_width=True,
                    )
            with b2:
                if proyecto.get("demo"):
                    st.link_button("Abrir demo ↗", proyecto["demo"], use_container_width=True)
                else:
                    st.button(
                        "Demo no disponible", disabled=True, key=f"demo_{i}",
                        use_container_width=True,
                    )

st.divider()
st.markdown(
    f'<p class="footer">Diseñado y construido con Python + Streamlit · {escape(UNIVERSIDAD)}<br>'
    'Data Analytics&nbsp; · &nbsp;Machine Learning&nbsp; · &nbsp;Software Engineering</p>',
    unsafe_allow_html=True,
)
