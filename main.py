import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

# 1. Configurare Streamlit
st.set_page_config(page_title="Analiză Imobiliară King County", layout="wide")
st.title("Analiza pieței imobiliare - King County, Washington")


# Încărcarea datelor
@st.cache_data
def load_data():
    df = pd.read_csv("kc_house_data.csv")
    return df


df = load_data()

# Bara laterală pentru navigare
st.sidebar.header("Meniu navigare")
sectiune = st.sidebar.radio("Alege secțiunea:", [
    "1. Explorare si curatare date",
    "2. Agregari si statistici",
    "3. Transformari (Codificare & Scalare)",
    "4. Analiza spatiala (Geopandas)",
    "5. Machine Learning (K-Means & Regresie)"
])

#1 curatare si explorare
if sectiune == "1. Explorare si curatare date":
    st.header("Explorarea si curatarea datelor")
    st.write("Primele 5 randuri din setul de date:")
    st.dataframe(df.head())

    # Tratarea valorilor lipsa (Simulare/Forțare dacă exista)
    st.subheader("Tratarea valorilor lipsa")
    df['waterfront'] = df['waterfront'].fillna(0)
    df['view'] = df['view'].fillna(0)
    st.success("Valorile lipsa din 'waterfront' si 'view' au fost tratate completandu-le cu 0.")

    # Tratarea valorilor extreme (Outliers) pt pret
    st.subheader("Tratarea valorilor extreme pentru pret")
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=df['price'], ax=ax, color='skyblue')
    st.pyplot(fig)

    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    limita_sup = Q3 + 1.5 * IQR

    df_clean = df[df['price'] <= limita_sup]
    st.success(f"S-au eliminat {len(df) - len(df_clean)} inregistrari cu preturi extreme (peste {limita_sup:,.0f}$)")
    st.write(f"Datele curatate au ramas cu {len(df_clean)} de randuri analizabile.")

#2 agregari si statistici
elif sectiune == "2. Agregari si statistici":
    st.header("Statistici, grupari si agregari (pandas)")

    st.subheader("Pretul mediu si calitatea in functie de conditia casei")
    # Condiția casei este notată de la 1 la 5
    grup_conditie = df.groupby('condition').agg({
        'price': 'mean',
        'id': 'count',
        'sqft_living': 'mean'
    }).rename(columns={'price': 'Pret Mediu ($)', 'id': 'Numar Case', 'sqft_living': 'Suprafata Medie'})

    st.dataframe(grup_conditie.style.format("{:.2f}"))

    fig, ax = plt.subplots()
    sns.barplot(x=grup_conditie.index, y=grup_conditie['Pret Mediu ($)'], palette='coolwarm', ax=ax)
    plt.title("Pretul mediu in functie de starea casei")
    st.pyplot(fig)

#3 transformarti
elif sectiune == "3. Transformari (Codificare & Scalare)":
    st.header("Pregatirea datelor")

    # codificare
    st.subheader("1. Codificarea variabilelor categorice/nominale")
    st.write("Transformam starea casei (`condition`) în coloane Dummy (One-Hot Encoding):")
    df_encoded = pd.get_dummies(df, columns=['condition'], drop_first=True)
    st.dataframe(df_encoded[['price', 'condition_2', 'condition_3', 'condition_4', 'condition_5']].head())

    # scalare
    st.subheader("2. Scalarea datelor (StandardScaler)")
    scaler = StandardScaler()
    coloane_de_scalat = ['price', 'sqft_living', 'bedrooms', 'bathrooms']
    df_scaled = df.copy()
    df_scaled[coloane_de_scalat] = scaler.fit_transform(df[coloane_de_scalat].fillna(0))

    st.write("Datele scalate (Media devine 0, Deviatia Standard 1):")
    st.dataframe(df_scaled[coloane_de_scalat].head())

#4analiza statistica
elif sectiune == "4. Analiza spatiala (geopandas)":
    st.header(" Analiza spasiala cu geopandas")
    st.write("Transformam latitudinea si longitudinea intr-un format de geometrie spatiala.")

    #esantion pentru viteza
    df_sample = df.sample(1500, random_state=42)

    geometry = [Point(xy) for xy in zip(df_sample['long'], df_sample['lat'])]
    gdf = gpd.GeoDataFrame(df_sample, geometry=geometry)

    st.write("GeoDataFrame generat:")
    st.dataframe(gdf[['price', 'bedrooms', 'geometry']].head())

    st.subheader("Distributia spatiala a caselor")
    st.map(df_sample[['lat', 'long']])

#5 machine learning
elif sectiune == "5.Machine Learning (K-Means & Regresie)":
    st.header("Machine Learning & Modelare")

    #curat datele
    df_ml = df.dropna(subset=['lat', 'long', 'price', 'sqft_living', 'bedrooms', 'bathrooms'])

    st.subheader("1. Segmentarea Zonelor (K-Means clusterizare din scikit-learn)")
    st.write("Grupam casele in 3 zone investitionale bazate pe locatie si pret.")
    X_cluster = df_ml[['lat', 'long', 'price']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_ml['Cluster'] = kmeans.fit_predict(X_cluster)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_ml.sample(2000), x='long', y='lat', hue='Cluster', palette='Set1', s=15, ax=ax)
    plt.title("Clustere geografice (Esantion)")
    st.pyplot(fig)

    st.subheader("2. Regresie Multipla OLS (statsmodels)")
    st.write("Analizam cat de mult influenteaza suprafata, numarul de dormitoare si baile pretul final.")

    X = df_ml[['sqft_living', 'bedrooms', 'bathrooms']]
    X = sm.add_constant(X)
    y = df_ml['price']

    model = sm.OLS(y, X).fit()

    st.text(model.summary())
    st.success(
        "Tabelul OLS confirma ca 'sqft_living' are o valoare P mica (P>|t|), fiind cel mai bun prezicator pentru pret.")