import streamlit as st
import pandas as pd
import joblib

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
st.set_page_config(
    page_title="SafeTrip USA",
    page_icon="🔒",
    layout="wide"
)

# ======================================================
# ESTILOS (WEB SOBRIA + MAPA USA)
# ======================================================
st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(18,22,30,0.90), rgba(18,22,30,0.90)),
        url("https://upload.wikimedia.org/wikipedia/commons/a/a5/Map_of_USA_with_state_names.svg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #f2f2f2;
    font-family: "Segoe UI", sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0e1525;
}

/* Tarjetas generales */
.card {
    background-color: rgba(255,255,255,0.08);
    padding: 34px;
    border-radius: 18px;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}

/* Tarjeta índice estimado (FORZAR fondo blanco) */
.index-card {
    background-color: white !important;
    color: black !important;
    padding: 20px !important;
    border-radius: 12px !important;
    text-align: center !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-top: 20px !important;
}

/* Forzar visibilidad de labels e inputs */
label, .stSelectbox label {
    color: #f2f2f2 !important;
    font-weight: 500;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.95);
    color: #111;
    border-radius: 8px;
}

/* Botones */
.stButton > button {
    background-color: #2d6cdf;
    color: white;
    border-radius: 10px;
    padding: 8px 22px;
    border: none;
}

.stButton > button:hover {
    background-color: #1f4fb2;
}

/* Resultados */
.result {
    padding: 18px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
    text-align: center;
}

.safe { background-color: #1f7a5a; }
.medium { background-color: #c9b037; color: #1f1f1f; }
.high { background-color: #a94442; }

/* Leyenda */
.legend {
    display: flex;
    gap: 30px;
    margin-top: 20px;
    font-size: 15px;
}

.dot {
    height: 12px;
    width: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

.green { background-color: #2ecc71; }
.yellow { background-color: #f1c40f; }
.red { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# CARGA DE MODELO
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("regressionlineal_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("selected_features.txt", "r") as f:
        features = [line.strip() for line in f.readlines()]
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

if "ViolentCrimesPerPop" in feature_names:
    feature_names.remove("ViolentCrimesPerPop")

# ======================================================
# NOMBRES DESCRIPTIVOS
# ======================================================
pretty_names = {
    "PctIlleg": "Niños nacidos fuera del matrimonio",
    "racepctblack": "Población afroamericana (%)",
    "pctWPubAsst": "Asistencia pública (población blanca)",
    "FemalePctDiv": "Mujeres divorciadas (%)",
    "TotalPctDiv": "Personas divorciadas (%)",
    "MalePctDivorce": "Hombres divorciados (%)",
    "PctPopUnderPov": "Población bajo el umbral de pobreza (%)",
    "PctUnemployed": "Tasa de desempleo (%)",
    "PctHousNoPhone": "Viviendas sin teléfono (%)",
    "PctNotHSGrad": "Población sin estudios secundarios (%)",
    "PctVacantBoarded": "Viviendas abandonadas (%)",
    "PctHousLess3BR": "Viviendas con menos de 3 habitaciones (%)",
    "NumIlleg": "Número de actividades ilegales",
    "PctPersOwnOccup": "Vivienda en propiedad (%)",
    "pctWInvInc": "Ingresos por inversión",
    "PctTeen2Par": "Adolescentes con dos padres (%)",
    "PctYoungKids2Par": "Niños pequeños con dos padres (%)",
    "racePctWhite": "Población blanca (%)",
    "PctFam2Par": "Familias con dos padres (%)",
    "PctKids2Par": "Niños con dos padres (%)"
}

# ======================================================
# NAVEGACIÓN (DESPLEGABLE)
# ======================================================
st.sidebar.title("🔒 SafeTrip USA")
pagina = st.sidebar.selectbox(
    "Menú",
    ["🏠 Inicio", "🧪 Test de Peligrosidad", "📤 Subir Archivo", "🚀 Próximos Pasos"]
)

# ======================================================
# INICIO
# ======================================================
if pagina == "🏠 Inicio":
    st.title("SafeTrip USA")
    st.subheader("Evaluación predictiva del nivel de inseguridad en Estados Unidos")

    st.write("""
    SafeTrip USA es una aplicación web basada en aprendizaje automático cuyo objetivo es
    estimar el nivel de criminalidad de una zona a partir de indicadores socioeconómicos,
    demográficos y estructurales.
    """)

    st.write("""
    El sistema utiliza un modelo predictivo entrenado sobre datos reales para generar un
    **índice de riesgo normalizado entre 0 y 1**, donde valores más altos indican una mayor
    probabilidad de criminalidad violenta.
    """)

    st.write("""
    Esta herramienta está pensada como apoyo a la toma de decisiones para viajeros,
    investigadores o usuarios interesados en analizar el contexto de seguridad de
    diferentes áreas dentro de Estados Unidos.
    """)

    st.markdown("""
    <div class="card">
        <div class="legend">
            <div><span class="dot green"></span>Zona segura</div>
            <div><span class="dot yellow"></span>Riesgo medio</div>
            <div><span class="dot red"></span>Alto riesgo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# TEST
# ======================================================
elif pagina == "🧪 Test de Peligrosidad":
    st.markdown("<div class='card'><h2>Test de peligrosidad</h2></div>", unsafe_allow_html=True)

    nivel = {"Bajo": 0.0, "Medio": 0.5, "Alto": 1.0}
    opciones = ["Bajo", "Medio", "Alto"]
    valores = {}

    for f in feature_names:
        valores[f] = nivel[st.selectbox(pretty_names.get(f, f), opciones, index=1)]

    if st.button("Calcular riesgo"):
        df = pd.DataFrame([valores])[feature_names]
        pred = model.predict(scaler.transform(df))[0]

        # Índice estimado en tarjeta completamente blanca
        st.markdown(
            f"<div class='index-card'>Índice estimado: <b>{pred:.3f}</b></div>",
            unsafe_allow_html=True
        )

        # Resultado de riesgo con colores
        if pred <= 0.33:
            st.markdown("<div class='result safe'>Zona segura</div>", unsafe_allow_html=True)
        elif pred <= 0.66:
            st.markdown("<div class='result medium'>Zona con riesgo medio</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result high'>Zona de alto riesgo</div>", unsafe_allow_html=True)

# ======================================================
# SUBIR ARCHIVO
# ======================================================
elif pagina == "📤 Subir Archivo":
    st.markdown("<div class='card'><h2>Clasificación mediante archivo CSV</h2></div>", unsafe_allow_html=True)
    file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

# ======================================================
# PRÓXIMOS PASOS
# ======================================================
elif pagina == "🚀 Próximos Pasos":
    st.markdown("""
    <div class="card">
        <h2>Desarrollo futuro</h2>
        <ul>
            <li>Mapas interactivos de riesgo</li>
            <li>Modelos específicos por estado</li>
            <li>Integración con datos en tiempo real</li>
            <li>Aplicación móvil orientada a viajeros</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
