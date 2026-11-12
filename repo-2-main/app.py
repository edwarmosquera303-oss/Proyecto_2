

from html import escape
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from proyectos import NOMBRE, UNIVERSIDAD, TEMAS, PROYECTOS

CARPETA_IMAGENES = Path(__file__).parent / "imagenes"
PROPORCION = 16 / 9
COLUMNAS = 2

st.set_page_config(page_title="Portafolio · IA y datos", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
    h1 {letter-spacing:-0.02em; line-height:1.05;}
    .quien {color:#0f7f86; font-weight:500; margin-bottom:0.25rem;}
    .tema {color:#0f7f86; font-weight:500; font-size:0.9rem; margin:0.6rem 0 0 0;}
    .tag {display:inline-block; background:rgba(120,140,170,0.22); border-radius:6px;
         padding:0.1rem 0.55rem; margin:0 0.35rem 0.25rem 0; font-size:0.85rem;}
    .sin-imagen {aspect-ratio:16/9; border-radius:8px; display:flex; align-items:center;
                 justify-content:center; color:#fff; font-weight:500;
                 background:linear-gradient(135deg,#0f7f86,#e8503f);}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-top:5px solid #e8503f;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------
# Imágenes: recorta al centro en 16:9 para que todas las tarjetas queden parejas
# -----------------------------------------------------------------
@st.cache_data
def cargar_imagen(nombre):
    ruta = CARPETA_IMAGENES / nombre
    if not ruta.is_file():
        return None
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


def mostrar_imagen(proyecto):
    ref = proyecto.get("imagen", "")
    if ref.startswith("http"):
        st.image(ref, use_container_width=True)
        return
    img = cargar_imagen(ref) if ref else None
    if img is not None:
        st.image(img, use_container_width=True)
    else:
        st.markdown('<div class="sin-imagen">Imagen pendiente</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------
st.markdown(f'<p class="quien">{escape(NOMBRE)} · Programación Avanzada · {escape(UNIVERSIDAD)}</p>', unsafe_allow_html=True)
st.title("Portafolio de proyectos de IA y datos")
st.write(
    "Portafolio de proyectos de inteligencia artificial, análisis de datos y "
    "aprendizaje automático. Reúne aplicaciones interactivas y proyectos orientados "
    "a la exploración de datos, la modelación predictiva, la visualización analítica "
    "y la resolución de problemas reales con técnicas de IA: preparación de datos, "
    "regresión, clasificación, series de tiempo, optimización y lógica algorítmica, "
    "aplicados a contextos concretos."
)

# -----------------------------------------------------------------
# Mini-Lab Interactivo: Clasificación y Regresión en Vivo
# -----------------------------------------------------------------
st.subheader("🧪 Mini-Lab: Modelado Predictivo en Vivo")
st.write(
    "Explora de forma interactiva cómo responden los modelos predictivos cambiando "
    "sus hiperparámetros y observando las métricas en tiempo real."
)

tipo_modelo = st.radio(
    "Selecciona el tipo de modelo:",
    ["Regresión (Ajuste de Línea)", "Clasificación (Frontera de Decisión)"],
    horizontal=True,
)

np.random.seed(42)

if tipo_modelo == "Regresión (Ajuste de Línea)":
    # Datos sintéticos para Regresión
    x = np.linspace(0, 10, 50)
    y_real = 2 * x + 1 + np.random.normal(0, 2, 50)

    c1, c2 = st.columns(2)
    with c1:
        pendiente = st.slider("Pendiente (m)", 0.0, 4.0, 1.8, step=0.1)
    with c2:
        intercepto = st.slider("Intercepto (b)", -5.0, 5.0, 1.5, step=0.2)

    # Predicción
    y_pred = pendiente * x + intercepto

    # Métricas
    mse = np.mean((y_real - y_pred) ** 2)
    st.metric(label="📉 Error Cuadrático Medio (MSE)", value=f"{mse:.2f}")

    # Gráfica
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(x, y_real, color="#0f7f86", alpha=0.7, label="Datos reales")
    ax.plot(x, y_pred, color="#e8503f", linewidth=2.5, label="Línea de regresión")
    ax.set_ylim(-2, 25)
    ax.legend(frameon=False)
    ax.axis("off")
    fig.patch.set_alpha(0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    # Datos sintéticos para Clasificación
    cluster1 = np.random.normal(loc=3, scale=1.0, size=(25, 2))
    cluster2 = np.random.normal(loc=7, scale=1.0, size=(25, 2))

    umbral = st.slider("Ajustar Umbral de Decisión (Frontera X)", 1.0, 9.0, 5.0, step=0.1)

    # Evaluación simple del umbral
    aciertos_c1 = np.sum(cluster1[:, 0] < umbral)
    aciertos_c2 = np.sum(cluster2[:, 0] >= umbral)
    accuracy = (aciertos_c1 + aciertos_c2) / 50.0

    st.metric(label="🎯 Precisión del Modelo (Accuracy)", value=f"{accuracy * 100:.1f}%")

    # Gráfica
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], color="#0f7f86", label="Clase A", alpha=0.8)
    ax.scatter(cluster2[:, 0], cluster2[:, 1], color="#e8503f", label="Clase B", alpha=0.8)
    ax.axvline(umbral, color="gray", linestyle="--", linewidth=2, label="Frontera de decisión")
    ax.legend(frameon=False)
    ax.axis("off")
    fig.patch.set_alpha(0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------------
# Proyectos en cuadrícula
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Proyectos en cuadrícula
# -----------------------------------------------------------------
filtro = st.radio("Filtrar por tema", ["Todos"] + TEMAS, horizontal=True, label_visibility="collapsed")
visibles = [p for p in PROYECTOS if filtro == "Todos" or p["tema"] == filtro]

st.header(f"Proyectos ({len(visibles)})")

if not visibles:
    st.info("Aún no hay proyectos en este tema. Asígnalo en el campo 'tema' de proyectos.py.")

for inicio in range(0, len(visibles), COLUMNAS):
    columnas = st.columns(COLUMNAS)
    for col, (i, p) in zip(columnas, enumerate(visibles[inicio:inicio + COLUMNAS], start=inicio)):
        with col, st.container(border=True):
            mostrar_imagen(p)
            st.markdown(f'<p class="tema">{escape(p["tema"])}</p>', unsafe_allow_html=True)
            st.subheader(p["titulo"])
            st.write(p["desc"])
            st.markdown("".join(f'<span class="tag">{escape(tag)}</span>' for tag in p["tags"]), unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                if p.get("repo"):
                    st.link_button("Ver código", p["repo"], type="primary", use_container_width=True)
                else:
                    st.button("Ver código", disabled=True, key=f"repo_{i}", help="Falta el link del repo", use_container_width=True)
            with b2:
                if p.get("demo"):
                    st.link_button("Abrir app", p["demo"], use_container_width=True)

st.divider()
st.caption(f"Hecho con Python y Streamlit · Programación Avanzada · {UNIVERSIDAD}")