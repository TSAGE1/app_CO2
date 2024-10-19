import pandas as pd
import gdown

# URL de Google Drive
url = 'https://drive.google.com/uc?id=19agA1zcwSmj9E1hDxl57wAG96KJXceBQ'
output = 'data.csv'

# Télécharger le fichier
gdown.download(url, output, quiet=False)

# Chargement du DataFrame
df = pd.read_csv(output, index_col=0)

# Affichage des premières lignes du DataFrame
print("Voici les premières lignes du dataset :")
print(df.head())
# Filtrage des données pour ne garder que les véhicules de France (FR) et Allemagne (DE)
df = df.loc[(df['Country'].isin(['FR', 'DE'])) & (df['Ewltp (g/km)'] > 0)]

# Affichage des premières lignes après le filtrage
print("\nLes premières lignes après filtrage des véhicules de France et d'Allemagne :")
print(df.head())

# Identification et suppression des lignes dupliquées
print("\nNombre de lignes dupliquées avant suppression :", df.duplicated().sum())
df = df.drop_duplicates()
print("Nombre de lignes après suppression des doublons :", df.shape[0])

# Identification des colonnes avec une seule valeur unique et suppression
columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
print("\nColonnes avec une valeur unique qui seront supprimées :", columns_to_drop)
df = df.drop(columns=columns_to_drop)

# Filtrage des données pour ne garder que les véhicules de France (FR) et Allemagne (DE)
df = df.loc[(df['Country'].isin(['FR', 'DE'])) & (df['Ewltp (g/km)'] > 0)]

# Affichage des premières lignes après le filtrage
print("\nLes premières lignes après filtrage des véhicules de France et d'Allemagne :")
print(df.head())

# Identification et suppression des lignes dupliquées
print("\nNombre de lignes dupliquées avant suppression :", df.duplicated().sum())
df = df.drop_duplicates()
print("Nombre de lignes après suppression des doublons :", df.shape[0])

# Identification des colonnes avec une seule valeur unique et suppression
columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
print("\nColonnes avec une valeur unique qui seront supprimées :", columns_to_drop)
df = df.drop(columns=columns_to_drop)

# Filtrage des données pour ne garder que les véhicules de France (FR) et Allemagne (DE)
df = df.loc[(df['Country'].isin(['FR', 'DE'])) & (df['Ewltp (g/km)'] > 0)]

# Affichage des premières lignes après le filtrage
print("\nLes premières lignes après filtrage des véhicules de France et d'Allemagne :")
print(df.head())

# Identification et suppression des lignes dupliquées
print("\nNombre de lignes dupliquées avant suppression :", df.duplicated().sum())
df = df.drop_duplicates()
print("Nombre de lignes après suppression des doublons :", df.shape[0])

# Identification des colonnes avec une seule valeur unique et suppression
columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
print("\nColonnes avec une valeur unique qui seront supprimées :", columns_to_drop)
df = df.drop(columns=columns_to_drop)

# Afficher la nouvelle dimension du DataFrame après suppression des colonnes
print("\nDimensions du DataFrame après suppression des colonnes avec une seule valeur unique :", df.shape)
import streamlit as st
import pandas as pd
import gdown

# Télécharger les données si elles n'existent pas
url = 'https://drive.google.com/uc?id=19agA1zcwSmj9E1hDxl57wAG96KJXceBQ'
output = 'data.csv'
gdown.download(url, output, quiet=False)

# Charger les données
df = pd.read_csv(output, index_col=0)

# Utiliser Streamlit pour afficher le DataFrame
st.title("Analyse des données d'émissions de CO2")
st.write("Voici les premières lignes des données :")
st.dataframe(df.head())

# Filtrage des données pour les véhicules français et allemands
df = df.loc[(df['Country'].isin(['FR', 'DE'])) & (df['Ewltp (g/km)'] > 0)]

st.write("Données après filtrage pour la France et l'Allemagne :")
st.dataframe(df.head())

# Nombre de lignes dupliquées
st.write(f"Nombre de lignes dupliquées avant suppression : {df.duplicated().sum()}")

# Suppression des doublons
df = df.drop_duplicates()
st.write(f"Nombre de lignes après suppression des doublons : {df.shape[0]}")
# Calcul du pourcentage de valeurs manquantes pour chaque colonne
missing_percent = df.isnull().mean() * 100

# Afficher les colonnes avec plus de 75% de valeurs manquantes
high_missing_cols = missing_percent[missing_percent > 75].index
print(f"Colonnes avec plus de 75% de valeurs manquantes : {list(high_missing_cols)}")

# Suppression des colonnes avec plus de 75% de valeurs manquantes
df = df.drop(columns=high_missing_cols)

# Afficher la dimension du DataFrame après suppression
print(f"\nDimensions du DataFrame après suppression des colonnes à forte proportion de valeurs manquantes : {df.shape}")
from sklearn.impute import SimpleImputer

# Imputation des valeurs manquantes pour les variables numériques (par la médiane)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Imputation des valeurs manquantes pour les variables catégorielles (par la valeur la plus fréquente)
cat_cols = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Affichage des premières lignes après imputation
print("\nLes premières lignes après imputation des valeurs manquantes :")
print(df.head())

# Vérification des valeurs manquantes après imputation
print("\nValeurs manquantes après imputation :")
print(df.isnull().sum())
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('data.csv', index_col=0)

# Titre de l'application
st.title("Analyse des émissions de CO2 des véhicules")

# Affichage des premières lignes du DataFrame
st.subheader("Données brutes")
st.dataframe(df.head())

# Filtrage par pays (France et Allemagne uniquement)
countries = st.multiselect("Sélectionner les pays", options=df['Country'].unique(), default=['FR', 'DE'])
df_filtered = df[df['Country'].isin(countries)]

# Afficher les données filtrées
st.subheader("Données filtrées")
st.dataframe(df_filtered.head())

# Distribution des variables numériques
st.subheader("Distribution des variables numériques")
num_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns
selected_num_col = st.selectbox("Sélectionner une variable numérique", num_cols)

fig, ax = plt.subplots()
sns.histplot(df_filtered[selected_num_col], kde=True, ax=ax)
st.pyplot(fig)

# Distribution des variables catégorielles
st.subheader("Distribution des variables catégorielles")
cat_cols = df_filtered.select_dtypes(include=['object']).columns
selected_cat_col = st.selectbox("Sélectionner une variable catégorielle", cat_cols)

fig, ax = plt.subplots()
sns.countplot(x=selected_cat_col, data=df_filtered, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
# Sélectionner une plage de valeurs pour une variable numérique (ex. Ewltp (g/km))
st.subheader("Filtrer par plage de valeurs")
min_value, max_value = st.slider("Sélectionner la plage pour les émissions de CO2 (g/km)", 
                                 float(df_filtered['Ewltp (g/km)'].min()), 
                                 float(df_filtered['Ewltp (g/km)'].max()), 
                                 (float(df_filtered['Ewltp (g/km)'].min()), 
                                  float(df_filtered['Ewltp (g/km)'].max())))

df_filtered = df_filtered[(df_filtered['Ewltp (g/km)'] >= min_value) & (df_filtered['Ewltp (g/km)'] <= max_value)]

# Afficher les données filtrées
st.write(f"Données filtrées pour les émissions de CO2 entre {min_value} et {max_value} g/km")
st.dataframe(df_filtered.head())
# Ajout de la possibilité de choisir le type de graphique
st.subheader("Visualisation personnalisée")

# Sélectionner les variables à tracer
selected_num_col_x = st.selectbox("Sélectionner la variable numérique pour l'axe des X", num_cols)
selected_num_col_y = st.selectbox("Sélectionner la variable numérique pour l'axe des Y", num_cols)

# Sélectionner le type de graphique
plot_type = st.radio("Choisir le type de graphique", ("Nuage de points", "Boîte à moustaches", "Histogramme"))

# Créer des visualisations basées sur les choix
fig, ax = plt.subplots()
if plot_type == "Nuage de points":
    sns.scatterplot(x=selected_num_col_x, y=selected_num_col_y, data=df_filtered, ax=ax)
elif plot_type == "Boîte à moustaches":
    sns.boxplot(x=selected_num_col_x, y=selected_num_col_y, data=df_filtered, ax=ax)
elif plot_type == "Histogramme":
    sns.histplot(df_filtered[selected_num_col_x], kde=True, ax=ax)

st.pyplot(fig)
# Affichage de plusieurs graphiques dans un tableau de bord
st.subheader("Tableau de bord interactif")

col1, col2 = st.columns(2)

with col1:
    st.write(f"Distribution de {selected_num_col_x}")
    fig, ax = plt.subplots()
    sns.histplot(df_filtered[selected_num_col_x], kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.write(f"Nuage de points entre {selected_num_col_x} et {selected_num_col_y}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=selected_num_col_x, y=selected_num_col_y, data=df_filtered, ax=ax)
    st.pyplot(fig)
@st.cache_data
def load_data():
    return pd.read_csv('data.csv', index_col=0)

# Charger les données avec cache
df = load_data()
# Charger le modèle pour les prédictions
st.subheader("Prédiction des émissions de CO2")

# Charger le modèle sauvegardé
if st.button("Charger le modèle pour prédiction"):
    model = joblib.load('model_CO2.joblib')
    st.success("Modèle chargé avec succès !")

    # Saisie des valeurs pour prédiction
    poids = st.number_input("Entrer le poids du véhicule (kg)", min_value=500, max_value=3000, value=1200)
    puissance = st.number_input("Entrer la puissance du moteur (KW)", min_value=50, max_value=400, value=100)

    # Prédiction basée sur les entrées de l'utilisateur
    if st.button("Prédire les émissions de CO2"):
        prediction = model.predict([[poids, puissance]])
        st.write(f"Les émissions de CO2 prédites pour ce véhicule sont : {prediction[0]:.2f} g/km")
# Ajout de messages de confirmation et d'interaction
if st.button("Entraîner le modèle"):
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'model_CO2.joblib')
    st.success("Le modèle a été entraîné et sauvegardé avec succès !")

if st.button("Télécharger les données filtrées"):
    st.download_button(
        label="Télécharger les données",
        data=df_filtered.to_csv(index=False),
        file_name='data_CO2_filtered.csv',
        mime='text/csv'
    )
    st.success("Les données filtrées ont été téléchargées avec succès !")





