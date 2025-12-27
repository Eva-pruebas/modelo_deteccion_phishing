

# Aplicación web (Streamlit) para detectar correos phishing
# usando TF-IDF + Regresión Logística (con scikit-learn).


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Establece el título de la pestaña y el layout de la aplicación.
st.set_page_config(page_title="Detector de Phishing", layout="centered")


st.title("Detector de Correos Phishing con IA")
st.write("Carga un dataset de correos electrónicos y analiza si un nuevo correo es phishing o legítimo.")


# Paso 1: ENTRADA DE DATOS: SUBIR CSV

# El uploader pide un CSV con columnas 'email' (texto) y 'label' (0/1).
uploaded_file = st.file_uploader(
    "Cargar archivo CSV con columnas 'email' y 'label'", type=["csv"]
)

# Inicializo referencias a modelo y vectorizador.
model = None
vectorizer = None

# Si el usuario sube un archivo
if uploaded_file:
    # Carga del CSV en un DataFrame de pandas.
    df = pd.read_csv(uploaded_file)

    # Validación de los datos del csv: el fichero debe contener las columnas esperadas.
    if "email" in df.columns and "label" in df.columns:
        st.success("Dataset cargado correctamente.")
        st.write("Muestra del dataset:")
        # Visualización de las primeras filas para ver que se ha leído bien.
        st.dataframe(df.head())

       
        # Paso 2: VECTORIZACIÓN DEL TEXTO (TF-IDF)
       
        # TfidfVectorizer convertirá textos en vectores numéricos ponderados.
        # 'stop_words' en inglés, eliminará las palabras vacías (stop words) en este caso en inglés 
        # durante el preprocesamiento del texto, se podría cambiar a 'spanish' 
        # o proporcionar una lista de stopwords.
        vectorizer = TfidfVectorizer(stop_words="english")

        # Se ajusta el vocabulario y se transforma todo el conjunto (fit_transform).
        # aprende el vocabulario e IDF a partir de todos los textos de email
        X = vectorizer.fit_transform(df["email"])

        #Extrae el vector de etiquetas de cada correo: 0 si es legítimo y 1 si es phishing.
        y = df["label"]

    
        # Paso 3: ENTRENAMIENTO DE CLASIFICACIÓN DE CORREOS (REGRESIÓN LOGÍSTICA)
       
        # Se usa LogisticRegression con los parámetros por defecto. 
        model = LogisticRegression()
        model.fit(X, y)


        # Paso 4: EVALUACIÓN  DE LOS RESULTADOS
     
        # Toma la matriz de características X (la representación TF‑IDF de los correos) 
        # y aplica el modelo entrenado (Regresión Logística) para generar las predicciones 
        # y se guardan en un array.
        y_pred = model.predict(X)

        # Reporte de métricas: precision, recall, F1 por clase, macro/weighted, etc.
        report = classification_report(y, y_pred, output_dict=True)
        st.write("Métricas del modelo:")
        st.json(report)

        # Matriz de confusión (valores absolutos): dónde acierta y dónde se confunde el modelo.
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión")
        st.pyplot(fig)

        # Paso 5: ANALIZAR UN CORREO DE FORMA MANUAL
      
        st.write("---")
        st.subheader("Analizar nuevo correo")

        # Campo de texto para pegar el contenido del correo a analizar.
        new_email = st.text_area("Escribe el contenido del correo aquí:")

        # Botón que ejecuta la predicción del texto introducido.
        if st.button("Analizar"):
            # Validamos que haya contenido.
            if new_email.strip() != "":
                # Transforma con el vectorizador ya ajustado (sin reaprender vocabulario).
                new_vector = vectorizer.transform([new_email])

                # Predicción binaria (0/1) con el modelo entrenado.
                prediction = model.predict(new_vector)[0]

                # Mensaje para el usuario según el resultado.
                if prediction == 1:
                    st.error("Este correo parece ser PHISHING.")
                else:
                    st.success("Este correo parece ser LEGÍTIMO.")
            else:
                st.warning("Por favor, escribe el contenido del correo.")
    else:
        # Error de formato si el CSV porque no trae las columnas esperadas.
        st.error("El archivo debe contener las columnas 'email' y 'label'.")
