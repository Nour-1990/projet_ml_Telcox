# Streamlit app - Churn prediction (Exploration, Préparation, ML, Évaluation)
# RandomForest + SMOTE + TF-IDF (FeedbackText) + threshold slider + pairplot/heatmap

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, recall_score,
    precision_score, f1_score, accuracy_score, roc_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from wordcloud import WordCloud, STOPWORDS

# ----------------------------
# Config & Paths
# ----------------------------
st.set_page_config(page_title="Churn Prediction - App", layout="wide")

# NOTE: Ces chemins sont des exemples. Ils doivent être remplacés par les chemins
# où vos fichiers CSV sont stockés, ou ajustés pour l'environnement de déploiement.
DATA_PATHS = {
    "clients": "data/clients.csv",
    "contracts": "data/contracts.csv",
    "interactions": "data/interactions.csv",
    "usage": "data/usage.csv"
}

# ----------------------------
# 1) Chargement & Fusion
# ----------------------------
@st.cache_data(show_spinner="Chargement et fusion des données...")
def load_and_merge(paths):
    """Charge et fusionne les différents fichiers de données."""
    try:
        # Tente de lire les fichiers csv depuis les paths fournis
        df_clients = pd.read_csv(paths["clients"])
        df_contracts = pd.read_csv(paths["contracts"])
        df_interactions = pd.read_csv(paths["interactions"])
        df_usage = pd.read_csv(paths["usage"])
    except Exception as e:
        st.error(f"Erreur lors de la lecture des fichiers CSV. Assurez-vous que les fichiers sont présents dans le répertoire de travail et que les chemins sont corrects. Erreur: {e}")
        return pd.DataFrame()

    # nettoyer noms de colonnes (retirer espaces)
    for df in [df_clients, df_contracts, df_interactions, df_usage]:
        df.columns = df.columns.str.strip()

    # fusionner successivement sur 'customerID'
    df = df_clients.merge(df_contracts, on="customerID", how="left")
    df = df.merge(df_usage, on="customerID", how="left")
    df = df.merge(df_interactions, on="customerID", how="left")

    # nettoyer TotalCharges possibles ' ' -> NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

    return df

df = load_and_merge(DATA_PATHS)
if df.empty:
    st.info("Chargement des données échoué. Arrêt de l'application.")
    st.stop()

st.title("Projet Churn - Exploration & Modélisation")
st.markdown("**Pipeline** : Exploration → Préparation → Modélisation → Évaluation (Streamlit interactif)")

# ----------------------------
# 2) Exploration des données
# ----------------------------
st.header("Exploration des données")

with st.expander("Aperçu & statistiques (raw)"):
    st.subheader("Aperçu (5 premières lignes)")
    st.dataframe(df.head())
    st.write(f"Shape : {df.shape}")

    st.subheader("Colonnes et types")
    info_df = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.count(),
        "missing_%": (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values("missing_%", ascending=False)
    st.dataframe(info_df)

# ----------------------------
# 3) Pré-traitement métier
# ----------------------------
st.header("Préparation des données (règles métier)")

df_clean = df.copy()

# 3.1 Suppression clients récents (tenure < 1)
if "tenure" not in df_clean.columns:
    st.error("La colonne 'tenure' est manquante. Arrêt.")
    st.stop()
df_clean = df_clean[df_clean["tenure"] >= 1].reset_index(drop=True)



# 3.2 conversions numériques utiles
numeric_cands = ["Age", "MonthlyCharges", "TotalCharges", "AvgDataUsage_GB", "NbContacts", "NumCalls", "LastContactDays", "SatisfactionScore"]
for c in numeric_cands:
    if c in df_clean.columns:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

# 3.3 logique métier imputations initiales (avant pipeline)
# facultatives -> keep marker or median
if "Partner" in df_clean.columns:
    df_clean["Partner"] = df_clean["Partner"].replace("__missing_value__", np.nan)
if "Dependents" in df_clean.columns:
    df_clean["Dependents"] = df_clean["Dependents"].replace("__missing_value__", np.nan)
if "InternetService" in df_clean.columns:
    df_clean["InternetService"] = df_clean["InternetService"].fillna("No Internet")
if "SatisfactionScore" in df_clean.columns:
    # Convertir en string avant de remplir car c'est souvent traité comme cat/ordinal
    df_clean["SatisfactionScore"] = df_clean["SatisfactionScore"].astype(str).replace("nan", "No Response")
if "NbContacts" in df_clean.columns:
    df_clean["NbContacts"] = df_clean["NbContacts"].fillna(0)
if "FeedbackText" in df_clean.columns:
    df_clean["FeedbackText"] = df_clean["FeedbackText"].fillna("")

# encode target
if "Churn" not in df_clean.columns:
    st.error("La colonne 'Churn' est introuvable.")
    st.stop()
df_clean["Churn"] = df_clean["Churn"].replace({"Yes": 1, "No": 0})
df_clean.dropna(subset=["Churn"], inplace=True)
df_clean["Churn"] = df_clean["Churn"].astype(int)

st.success("Nettoyage initial appliqué. Les imputations finales (median/mode/etc.) seront faites dans le pipeline ML.")

# ============================================================
# EXPLORATION DE DONNÉES
# ============================================================

st.header("Exploration de Données")

# --- Analyse descriptive ---
st.subheader("Analyse descriptive des variables numériques")

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    st.dataframe(df_clean[numeric_cols].describe().T.style.background_gradient(cmap='Blues'))
else:
    st.warning("Aucune colonne numérique détectée.")
    
st.markdown(
    """
    **Analyse descriptive :**
    * Le **nombre d'observations (count)** révèle le nombre de valeurs non nulles. Les colonnes avec un `count` inférieur au total nécessiteront une imputation.
    * La **moyenne (mean)** et l'**écart-type (std)** donnent une idée de la dispersion des données.
    * Le **min** et le **max** aident à identifier les valeurs extrêmes ou les erreurs de saisie.
    * *Exemple métier :* Observer la moyenne de `tenure` (ancienneté) et la comparer à la médiane (50%) pour comprendre si la clientèle est majoritairement ancienne ou récente.
    """
)

st.markdown("---")

# --- Distributions de variables clés ---
st.subheader("Visualisation des distributions de variables clés")

# Exclure la colonne cible "Churn" de la liste à tracer
numeric_cols_no_target = [col for col in numeric_cols if col not in ("Churn", "SeniorCitizen")]

# Choix de colonnes à visualiser
cols_to_plot = st.multiselect(
    "Choisissez les variables numériques à afficher :",
    options=numeric_cols_no_target,
    default=numeric_cols_no_target[:4] if len(numeric_cols_no_target) >= 4 else numeric_cols_no_target
)

for col in cols_to_plot:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_clean, x=col, hue="Churn", kde=True, palette=["green", "red"], ax=ax)
    ax.set_title(f"Distribution de {col} selon le Churn")
    st.pyplot(fig)
    
    # --- ANALYSE DÉTAILLÉE DES GRAPHIQUES DE DISTRIBUTION (INTEGRATION) ---
    if col == "tenure":
        st.markdown(
            """
            **Analyse de `tenure` (Ancienneté) :**
            * **Observation :** La distribution des *churners* (**rouge**) est souvent fortement concentrée aux faibles valeurs de `tenure` (0-10 mois).
            * **Interprétation métier :** Les clients récents sont beaucoup plus à risque de résiliation. Ils sont en phase d'évaluation du service.
            * **Recommandation :** Ciblez les clients dans leurs 3-6 premiers mois avec des communications et des offres proactives pour renforcer leur engagement.
            """
        )
    elif col == "MonthlyCharges":
        st.markdown(
            """
            **Analyse de `MonthlyCharges` (Frais mensuels) :**
            * **Observation :** Les *churners* peuvent se concentrer aux extrêmes : soit aux très bas prix (faible engagement), soit aux très hauts prix (insatisfaction liée au coût/service).
            * **Interprétation métier :** Les prix élevés sont un facteur de risque si la valeur perçue du service ne suit pas.
            * **Recommandation :** Pour les clients à haute mensualité, assurez-vous qu'ils utilisent et apprécient les services premium associés.
            """
        )
    elif col == "NbContacts":
        st.markdown(
            """
            **Analyse de `NbContacts` (Nombre de contacts support) :**
            * **Observation :** La distribution des *churners* est souvent décalée vers les valeurs élevées (`> 2` ou `3` contacts).
            * **Interprétation métier :** Un nombre élevé de contacts signale une **frustration ou un problème non résolu**. C'est un signal d'alerte très fort.
            * **Recommandation :** Mettre en place une alerte immédiate pour tout client ayant contacté le support plus de **X fois** dans les 30 derniers jours (X étant le seuil observé).
            """
        )
    elif col == "AvgDataUsage_GB":
        st.markdown(
            """
            **Analyse de `AvgDataUsage_GB` (Usage de données) :**
            * **Observation :** Si les *churners* se concentrent sur une faible consommation, cela signifie que le manque d'usage est lié à la résiliation.
            * **Interprétation métier :** Les clients qui n'utilisent pas beaucoup le service de données ne voient pas sa valeur.
            * **Recommandation :** Encourager l'utilisation (tutoriels, incitations à l'utilisation des apps) pour les clients à faible usage de données.
            """
        )
    # --- FIN ANALYSE DÉTAILLÉE ---

st.markdown("---")

# ============================================================
# BIVARIÉ : Visualisation simple feature vs Churn
# ============================================================
st.header("Analyse bivariée : Feature vs Churn (Boxplot)")
st.markdown(
    """
    Le **Boxplot** permet de comparer la distribution des valeurs d'une variable numérique
    entre les deux classes de la variable cible (`Churn=0` vs `Churn=1`).
    * Si la médiane (ligne au centre de la boîte) de la classe **Churn=1** est significativement différente de celle de **Churn=0**, la variable est un bon prédicteur de Churn.
    """
)

numeric_cols_no_target = [col for col in numeric_cols if col not in ("Churn", "SeniorCitizen")]

selected_feature = st.selectbox(
    "Choisissez une variable numérique à comparer au Churn :",
    options=numeric_cols_no_target,
    index=0
)

if selected_feature:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Churn", y=selected_feature, data=df_clean, palette=["green","red"], ax=ax)
    sns.stripplot(x="Churn", y=selected_feature, data=df_clean, color="black", alpha=0.3, ax=ax)
    ax.set_title(f"{selected_feature} vs Churn")
    st.pyplot(fig)

# ============================================================
# MATRICE DE CORRÉLATION
# ============================================================

st.subheader("Matrice de corrélation des variables numériques")

# Corrélation sur les variables numériques
corr_df = df_clean.select_dtypes(include=[np.number]).corr()

fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5
)
ax_corr.set_title("Matrice de corrélation entre les variables numériques")
st.pyplot(fig_corr)

st.markdown(
    """
    ### Interprétation de la Matrice de Corrélation
    * **Détection de Multicolinéarité :** Les valeurs proches de `+1` ou `-1` entre deux features (hors `Churn`) signalent une forte corrélation (ex: `MonthlyCharges` et `TotalCharges`). Il faut en tenir compte pour le modèle (ex: suppression de `TotalCharges` pour éviter la redondance).
    * **Impact sur le Churn :**
        * Les features avec une corrélation positive forte avec `Churn` (proche de 1) sont des facteurs de risque (ex: `MonthlyCharges` élevé).
        * Les features avec une corrélation négative forte avec `Churn` (proche de -1) sont des facteurs protecteurs (ex: `tenure` élevé).
    """
)

# Corrélation spécifique avec la variable cible (Churn)
if "Churn" in corr_df.columns:
    churn_corr = corr_df["Churn"].sort_values(ascending=False).drop("Churn", errors="ignore")
    st.markdown("### Corrélation des variables avec la cible `Churn`")
    st.bar_chart(churn_corr)
    st.markdown(
        """
        Le graphique en barres résume les coefficients de corrélation de chaque variable avec la cible `Churn`.
        * **Variables les plus corrélées positivement :** Ce sont les plus grands facteurs de risque (ex: **`MonthlyCharges`**, **`NbContacts`**).
        * **Variables les plus corrélées négativement :** Ce sont les plus grands facteurs de protection (ex: **`tenure`**, **`TotalCharges`**).
        """
    )


st.markdown("---")

# ============================================================
# Facteurs de Risque Clés — Dashboard métier (Barplots horizontaux)
# ============================================================

st.header("Facteurs de Risque Clés — Dashboard métier")
st.markdown(
    "Visualisation de l'impact des variables clés sur le taux de Churn. "
    "Les barres rouges indiquent un risque supérieur à la moyenne, vert un risque inférieur à la moyenne."
)

mean_churn = df_clean['Churn'].mean()
st.info(f"Taux moyen global de churn : **{mean_churn:.2f}**")

# -------------------------
# 1 Variable numérique : MonthlyCharges
# -------------------------
num_feature = "MonthlyCharges"
if num_feature in df_clean.columns:
    st.subheader(f"{num_feature} vs Churn")
    # Discrétisation en 5 bins égaux (quantiles)
    churn_rate = df_clean.groupby(pd.qcut(df_clean[num_feature], 5, duplicates='drop'))['Churn'].mean().reset_index()
    churn_rate.columns = [num_feature, 'Taux Churn']

    # Palette rouge/vert selon risque
    palette_colors = ["#EA4335" if r > mean_churn else "#34A853" for r in churn_rate['Taux Churn']]

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x='Taux Churn', y=num_feature, data=churn_rate, palette=palette_colors, ax=ax)
    
    # Annoter chaque barre avec le taux exact
    for index, row in churn_rate.iterrows():
        ax.text(row['Taux Churn'] + 0.01, index, f"{row['Taux Churn']:.2f}", va='center')
    
    ax.axvline(mean_churn, color='#FBBC04', linestyle='--', label=f"Taux moyen ({mean_churn:.2f})")
    ax.set_xlabel("Taux de Churn")
    ax.set_ylabel(num_feature)
    ax.set_title(f"Taux de Churn par tranche de {num_feature}")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown(
        """
        **Analyse de `MonthlyCharges` (par tranches) :**
        * **Observation :** Regardez les tranches où la barre est **rouge**. Ce sont les segments de prix qui présentent le plus grand risque de *Churn*.
        * **Interprétation métier :** Une augmentation du risque dans les tranches relativement supérieures suggère que les clients paient trop cher par rapport à ce qu'ils reçoivent (problème de *Valeur*).
        * **Recommandation :** Mettre en place des offres de rétention ciblées (réduction temporaire ou ajout de services gratuits) pour les clients situés dans la tranche la plus à risque.
        """
    )


# -------------------------
# 2 Variables Catégorielles / Ordinales
# -------------------------
risk_features_cat = ["ContractType", "InternetService", "SatisfactionScore"]
for feature in risk_features_cat:
    if feature in df_clean.columns:
        st.subheader(f"{feature} vs Churn")
        
        # Taux de churn par catégorie, tri décroissant pour lisibilité
        churn_rate = df_clean.groupby(feature)['Churn'].mean().sort_values(ascending=False).reset_index()
        churn_rate.columns = [feature, 'Taux Churn']

        # Palette rouge/vert
        palette_colors = ["#EA4335" if r > mean_churn else "#34A853" for r in churn_rate['Taux Churn']]

        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x='Taux Churn', y=feature, data=churn_rate, palette=palette_colors, ax=ax)

        # Annoter chaque barre
        for index, row in churn_rate.iterrows():
            ax.text(row['Taux Churn'] + 0.01, index, f"{row['Taux Churn']:.2f}", va='center')
        
        ax.axvline(mean_churn, color='#FBBC04', linestyle='--', label=f"Taux moyen ({mean_churn:.2f})")
        ax.set_xlabel("Taux de Churn")
        ax.set_ylabel(feature)
        ax.set_title(f"Taux de Churn par catégorie de {feature}")
        ax.legend()
        st.pyplot(fig)
        
        # --- ANALYSE DÉTAILLÉE DES GRAPHIQUES CATÉGORIELS (INTEGRATION) ---
        if feature == "ContractType":
            st.markdown(
                """
                **Analyse de `ContractType` :**
                * **Observation :** Le contrat **`Month-to-month` (Mensuel)** présente presque toujours le taux de Churn le plus élevé. Les contrats d'un an et de deux ans agissent comme des facteurs protecteurs.
                * **Interprétation métier :** Moins d'engagement = plus de risque. La période mensuelle est celle où le client a le moins de friction pour partir.
                * **Recommandation :** Convertir activement les clients mensuels en contrats à long terme (1 ou 2 ans) en offrant une incitation financière (réduction).
                """
            )
        elif feature == "InternetService":
            st.markdown(
                """
                **Analyse de `InternetService` :**
                * **Observation :** Comparez les risques entre les types de services (DSL, Fiber Optic, No Internet). Le service **Fiber Optic** est souvent associé à un risque de Churn élevé, malgré sa performance.
                * **Interprétation métier :** La fibre optique, bien que rapide, est souvent synonyme de problèmes de fiabilité, d'un prix élevé, ou d'attentes non satisfaites (problème de **Qualité/Stabilité**).
                * **Recommandation :** Prioriser l'amélioration de la stabilité du service Fiber Optic et renforcer le support technique pour ce segment.
                """
            )
        elif feature == "SatisfactionScore":
            st.markdown(
                """
                **Analyse de `SatisfactionScore` :**
                * **Observation :** Plus le score de satisfaction est bas (notamment 2 ou pas de reponse), plus le risque de Churn est critique. Donc le groupe **`No Response`** (valeurs manquantes) peut aussi révéler un segment à risque passif.
                * **Interprétation métier :** La corrélation est directe : l'insatisfaction peut être un moteur de la résiliation.
                * **Recommandation :** Établir un processus de suivi immédiat pour tous les clients déclarant un score de satisfaction de 1 ou 2 après une interaction.
                """
            )
        # --- FIN ANALYSE DÉTAILLÉE ---


# ----------------------------
# 6) Préparation dataset pour ML
# ----------------------------
st.header("Préparation finale & pipeline ML")

target = "Churn"
# on ôte TotalCharges pour éviter multicolinéarité avec MonthlyCharges/AvgChargePerMonth si souhaité
drop_cols = ["customerID", "TotalCharges"] 
drop_cols = [c for c in drop_cols if c in df_clean.columns]

X = df_clean.drop(columns=[target] + drop_cols, errors="ignore")
y = df_clean[target].copy()

# Sécurités: remplissage rapide pour objets/numériques avant ColumnTransformer (car il est plus strict que le pipeline)
for c in X.select_dtypes(include=[np.number]).columns:
    if X[c].isnull().any():
        X[c] = X[c].fillna(X[c].median()) # Imputation temporaire avant le pipeline
for c in X.select_dtypes(include=["object"]).columns:
    X[c] = X[c].astype(str)
    if X[c].str.contains("nan|__missing_value__").any():
        X[c] = X[c].replace(["nan", "__missing_value__"], "Missing_Value")

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = [c for c in X.select_dtypes(include=["object"]).columns if c != "FeedbackText"]
text_feature = "FeedbackText" if "FeedbackText" in X.columns else None

st.write(f"Features numériques ({len(num_features)}): {num_features[:5]}...")
st.write(f"Features catégorielles ({len(cat_features)}): {cat_features[:5]}...")
if text_feature:
    st.write("FeedbackText présent -> sera traité en **TF-IDF**")

# ----------------------------
# 7) Train/Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
st.write(f"Split: train={len(X_train)} / test={len(X_test)}")

# ----------------------------
# 8) Pipeline : preprocessing + SMOTE + RF
# ----------------------------
# numeric pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # Redondant car fait plus tôt, mais sécurité
    ("scaler", StandardScaler())
])

# categorical pipeline
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # Redondant car fait plus tôt, mais sécurité
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# text pipeline
if text_feature:
    text_transformer = Pipeline([
        # Transformer la colonne DataFrame/Series en un tableau 1D pour TfidfVectorizer
        ("selector", FunctionTransformer(lambda X: X.squeeze(), validate=False)), 
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=200)) # Limité à 200 mots-clés
    ])
else:
    text_transformer = None

transformers = []
transformers.append(("num", numeric_transformer, num_features))
if cat_features:
    transformers.append(("cat", categorical_transformer, cat_features))
if text_feature:
    # ColumnTransformer passe la colonne textuelle sous forme de DataFrame (n_samples, 1)
    transformers.append(("text", text_transformer, [text_feature]))

preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

# RF parameters (définis manuellement ou via GridSearchCV)
best_rf_params = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42,
    "class_weight": "balanced_subsample"
}

rf = RandomForestClassifier(**best_rf_params)

# Pipeline complet avec suréchantillonnage (SMOTE appliqué uniquement sur le train set via ImbPipeline)
pipe = ImbPipeline(steps=[
    ("preproc", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", rf)
])

# ----------------------------
# 9) Entraînement (fit)
# ----------------------------
st.info("Entraînement du pipeline (préprocessing + SMOTE + RandomForest). Cela peut prendre quelques secondes...")
@st.cache_resource(show_spinner=False)
def train_model(_pipe, X_train, y_train):
    """Fonction pour entraîner le modèle et mettre en cache le résultat."""
    with st.spinner("Entraînement en cours..."):
        _pipe.fit(X_train, y_train)
    return _pipe

pipe = train_model(pipe, X_train, y_train)
st.success("Entraînement terminé.")

# probabilités & prédictions par défaut (threshold 0.5)
y_proba_test = pipe.predict_proba(X_test)[:, 1]
y_proba_train = pipe.predict_proba(X_train)[:, 1]

# ----------------------------
# 10) Interface seuil interactif & métriques
# ----------------------------
st.header("Evaluation & Ajustement du seuil")
st.markdown(
    """
    **Objectif :** L'ajustement du seuil de décision permet d'équilibrer le **Rappel (Recall)** et la **Précision (Precision)**.
    * **Augmenter le Seuil** (vers 0.9) : Priorise la Précision. Le modèle ne prédit un Churn que s'il est *très certain* (moins de faux positifs), mais il manquera plus de vrais churners (Rappel faible).
    * **Diminuer le Seuil** (vers 0.1) : Priorise le Rappel. Le modèle détecte plus de vrais churners (important pour les actions de rétention), mais il y aura plus de faux positifs (clients non-churners ciblés à tort).
    """
)

threshold = st.slider("Seuil de décision (probabilité) :", min_value=0.01, max_value=0.99, value=0.5, step=0.01)

y_pred_test = (y_proba_test >= threshold).astype(int)
y_pred_train = (y_proba_train >= threshold).astype(int)

def metrics(y_true, y_pred, y_prob):
    """Calcule les métriques de performance."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }

metrics_train = metrics(y_train, y_pred_train, y_proba_train)
metrics_test = metrics(y_test, y_pred_test, y_proba_test)

# affichage tableaux
metrics_df = pd.DataFrame([metrics_train, metrics_test], index=["train", "test"]).T.round(3)
st.subheader("Métriques (Train vs Test)")
st.dataframe(metrics_df.style.background_gradient(cmap="viridis"))

st.subheader("Classification report (Test)")
st.code(classification_report(y_test, y_pred_test))

# ----------------------------
# 11) Matrice de confusion & ROC visu
# ----------------------------
st.subheader("Matrice de confusion & ROC")

col1, col2 = st.columns(2)
with col1:
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-churn", "Churn"], yticklabels=["Non-churn", "Churn"])
    ax.set_title(f"Matrice de confusion (seuil={threshold:.2f})")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    st.pyplot(fig)
    st.markdown(
        """
        **Matrice de Confusion :**
        * **Vrais Positifs (TP, bas-droit) :** Clients qui ont *churné* et que le modèle a **bien prédit**. Ce sont nos succès de détection.
        * **Faux Négatifs (FN, bas-gauche) :** Clients qui ont *churné* mais que le modèle a **manqué**. C'est le coût d'opportunité d'une action de rétention manquée (impacte le **Rappel**).
        * **Faux Positifs (FP, haut-droit) :** Clients qui n'ont **pas** *churné* mais que le modèle a ciblés à tort. C'est le coût des actions de rétention gaspillées (impacte la **Précision**).
        """
    )


with col2:
    fpr, tpr, thr = roc_curve(y_test, y_proba_test)
    auc_score = roc_auc_score(y_test, y_proba_test)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="orange")
    # marker for current threshold
    if thr is not None:
        # Trouver l'index du seuil le plus proche
        idx = np.argmin(np.abs(thr - threshold))
        ax2.plot(fpr[idx], tpr[idx], "ro", label=f"Seuil {threshold:.2f} (FPR={fpr[idx]:.2f}, TPR={tpr[idx]:.2f})")
    ax2.plot([0, 1], [0, 1], "--", color="grey")
    ax2.set_xlabel("Faux Positif Rate (FPR)"); ax2.set_ylabel("Vrai Positif Rate (TPR)"); ax2.legend()
    ax2.set_title("Courbe ROC")
    st.pyplot(fig2)
    st.markdown(
        """
        **Courbe ROC (Receiver Operating Characteristic) :**
        * **AUC (Area Under the Curve) :** Une valeur proche de **1.0** (ex: 0.85 ou plus) indique une excellente capacité du modèle à distinguer les *churners* des *non-churners*, quel que soit le seuil.
        * **Le Marqueur Rouge :** Il montre la performance du seuil actuel (`FPR` et `TPR`). Plus le marqueur est proche du coin supérieur gauche (TPR élevé, FPR faible), meilleure est la performance à ce seuil.
        """
    )


# ============================================================
# Courbe d'Apprentissage (Learning Curve)
# ============================================================

st.header("Courbe d'Apprentissage : Train vs Test")

# On préfixe pipe avec "_" pour que Streamlit ne tente pas de le hasher
@st.cache_data(show_spinner="Calcul de la courbe d'apprentissage en cours...")
def compute_learning_curve(_pipe, X_train, y_train, scoring='roc_auc'):
    """Calcule les courbes d'apprentissage (Train/Validation) pour un pipeline donné."""
   
    train_sizes, train_scores, test_scores = learning_curve(
        _pipe,
        X_train,
        y_train,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6),
        random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    return train_sizes, train_mean, test_mean

# Calcul de la courbe avec pipe
train_sizes, train_mean, test_mean = compute_learning_curve(pipe, X_train, y_train, scoring='roc_auc')

# Visualisation
fig_lc, ax_lc = plt.subplots(figsize=(8, 5))
ax_lc.plot(train_sizes, train_mean, 'o-', label='Score Entraînement (AUC)', color='blue')
ax_lc.plot(train_sizes, test_mean, 'o-', label='Score Validation (AUC)', color='orange')
ax_lc.set_title("Courbe d'Apprentissage - AUC Train vs Validation")
ax_lc.set_xlabel("Taille de l'échantillon d'entraînement")
ax_lc.set_ylabel("Score AUC")
ax_lc.legend()
ax_lc.grid(True, linestyle='--')
st.pyplot(fig_lc)

# Interprétation automatique
if len(test_mean) > 0 and len(train_mean) > 0:
    gap = abs(train_mean[-1] - test_mean[-1])
    if test_mean[-1] < 0.75:
        st.warning(f"Le score de validation final ({test_mean[-1]:.3f}) est faible : le modèle semble sous-ajusté (underfitting).")
        st.markdown(
            """
            **Sous-ajustement (Underfitting) :** Les deux courbes (Train et Validation) stagnent à un score bas. Le modèle est **trop simple** pour capturer les tendances des données.
            * **Action :** Utiliser un modèle plus complexe (ex: plus de profondeur pour RF, ou un autre algorithme), ou intégrer plus de features.
            """
        )
    elif gap > 0.05:
        st.error(f"Grand écart de performance ({gap:.3f}) entre train et validation → surapprentissage (overfitting) probable.")
        st.markdown(
            """
            **Surapprentissage (Overfitting) :** La courbe Train est haute, mais la courbe Validation est significativement plus basse. Le modèle a **mémorisé** le bruit du jeu d'entraînement.
            * **Action :** Réduire la complexité du modèle (ex: moins de profondeur ou d'estimateurs pour RF), ou augmenter la taille du jeu de données.
            """
        )
    else:
        st.success("Les performances train/validation sont cohérentes : modèle bien généralisé.")
        st.markdown(
            """
            **Bonne Généralisation :** Les deux courbes se rejoignent et sont à un niveau élevé. Le modèle apprend bien et généralise correctement aux nouvelles données.
            """
        )

# ----------------------------
# 13) Feature importance (Top 20) - récupération des noms robustement
# ----------------------------
st.header("Top 20 features hors text (importance)")

clf = pipe.named_steps["clf"]
pre = pipe.named_steps["preproc"]

# Récupération des noms de features après preprocessing
feature_names = []
try:
    # Utiliser get_feature_names_out() si disponible (sklearn >= 1.0)
    feature_names = pre.get_feature_names_out()
except Exception:
    # Fallback: assembler les noms manuellement
    for name, trans, cols in pre.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            # try:
                # onehot
                encoder = trans.named_steps["onehot"]
                feature_names.extend(encoder.get_feature_names_out(cols))
        #     except Exception:
        #         feature_names.extend(cols)
        # elif name == "text":
        #     try:
        #         tfidf = trans.named_steps["tfidf"]
        #         names = tfidf.get_feature_names_out()
        #         feature_names.extend([f"text_{n}" for n in names])
        #     except Exception:
        #         feature_names.append("FeedbackText_NLP")

if len(feature_names) != len(clf.feature_importances_):
    # Gestion de l'incohérence de dimension (si des colonnes ont été droppées/ajoutées de manière inattendue)
    minlen = min(len(feature_names), len(clf.feature_importances_))
    feature_names = feature_names[:minlen]
    importances = clf.feature_importances_[:minlen]
else:
    importances = clf.feature_importances_

# Affichage des importances
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
sns.barplot(x=imp_series.values, y=imp_series.index, palette="viridis", ax=ax_imp)
ax_imp.set_title("Top 20 features - RandomForest importance")
ax_imp.set_xlabel("Importance")
ax_imp.set_ylabel("Feature Name")
st.pyplot(fig_imp)

st.markdown(
    """
    ### Interprétation des Importances (Random Forest)
    Ce graphique montre quelles features le modèle Random Forest a jugé les plus pertinentes pour prédire le Churn.
    * **Priorité Absolue :** Les variables en haut de la liste (ex: **`tenure`**, **`InternetService`**, **`ContractType_Month-to-month`**, **`SatisfactionScore`**, **`NbContacts`**) sont les leviers d'action les plus efficaces pour la rétention.
    * **Révélaations Spécifiques :**
        * Si **`tenure`** est en tête, l'ancienneté est le meilleur indicateur (plus faible tenure = haut risque).
        * Si **`NbContacts`** est très haut, le support client et la résolution de problèmes sont la priorité.
        * Les termes de **`text_`** (FeedbackText) dans le top 20 confirment l'importance des plaintes clients.
    """
)

# ============================================================
# Analyse textuelle : FeedbackText (Churn vs Non-Churn) - anglais
# ============================================================

st.header("Analyse textuelle : FeedbackText")

if text_feature:
    st.subheader("WordClouds : churners vs non-churners (anglais)")
    
    # Stopwords anglais
    english_stopwords = set(STOPWORDS)

    # Séparer les données
    df_text = df_clean[[text_feature, target]].copy()
    # Filtrer les lignes avec du texte non vide
    df_text = df_text[df_text[text_feature].str.strip() != ""] 
    df_0 = df_text[df_text[target] == 0]
    df_1 = df_text[df_text[target] == 1]

    col_wc1, col_wc2 = st.columns(2)

    # --- WordCloud Non-churners ---
    if not df_0.empty:
        text_0 = " ".join(df_0[text_feature])
        wc_0 = WordCloud(stopwords=english_stopwords, width=600, height=400, 
                          background_color="white", colormap="summer").generate(text_0)
        
        with col_wc1:
            fig0, ax0 = plt.subplots(figsize=(8, 5))
            ax0.imshow(wc_0, interpolation='bilinear')
            ax0.axis("off")
            ax0.set_title("Feedback Non-Churners (label=0)")
            st.pyplot(fig0)
    else:
        with col_wc1: st.info("Pas de feedback disponible pour les Non-Churners.")

    # --- WordCloud Churners ---
    if not df_1.empty:
        text_1 = " ".join(df_1[text_feature])
        wc_1 = WordCloud(stopwords=english_stopwords, width=600, height=400, 
                          background_color="white", colormap="autumn").generate(text_1)

        with col_wc2:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.imshow(wc_1, interpolation='bilinear')
            ax1.axis("off")
            ax1.set_title("Feedback Churners (label=1)")
            st.pyplot(fig1)
    else:
        with col_wc2: st.info("Pas de feedback disponible pour les Churners.")


    st.markdown("""
    **Analyse du WordCloud :**
    * **Churners (Rouge) :** Les mots dominants sont souvent liés à la **frustration, aux problèmes de service, et au coût** (par ex. `price`, `slow`, `support`, `bad`, `terrible`, `unacceptable`). Ils pointent directement les déficiences du service.
    * **Non-Churners (Vert) :** Les mots dominants reflètent souvent la **satisfaction et la qualité perçue** (par ex. `great`, `satisfied`, `good`, `helpful`, `fine`).
    * **Recommandation métier :** Utiliser les thèmes dominants chez les churners pour créer des **initiatives d'amélioration de service ciblées** (ex: si `slow` est dominant, investir dans la bande passante).
    """)
else:
    st.info("Pas de colonne FeedbackText disponible pour l'analyse textuelle.")


# ============================================================
# 14) Segments à risque : Barplots horizontaux par feature
# ============================================================

st.header("Impact des features sur le churn — Segments à risque")

mean_churn_rate = df_clean['Churn'].mean()
st.info(f"Taux de Churn Moyen Global : **{mean_churn_rate:.2f}**")

# Features continues pour l'analyse
continuous_features = [c for c in numeric_cols_no_target if c != "Churn"]

# Nombre de bins pour discrétisation
n_bins = st.slider("Nombre de bins pour chaque feature :", 3, 15, 6)

# Seuil pour alerter sur un segment
churn_risk_threshold = st.slider("Seuil churn pour alerter un segment :", 0.0, 1.0, max(0.3, mean_churn_rate * 1.5), 0.05)
st.write(f"Segments considérés à risque si taux de churn ≥ **{churn_risk_threshold:.2f}**")

risk_data_list = []

for feature in continuous_features:
    # 1. Discrétisation en n_bins
    df_clean[f"{feature}_bin"] = pd.cut(df_clean[feature], bins=n_bins, include_lowest=True)

    # 2. Calcul taux churn et non-churn par bin
    churn_by_bin = df_clean.groupby(f"{feature}_bin")["Churn"].mean().fillna(0)
    non_churn_by_bin = 1 - churn_by_bin

    # 3. Taille des bins (pour poids population)
    count_by_bin = df_clean.groupby(f"{feature}_bin")[feature].count().fillna(0)
    count_percent = count_by_bin / len(df_clean) * 100

    # Préparer le DataFrame pour le barplot horizontal
    plot_df = pd.DataFrame({
        'Segment': churn_by_bin.index.astype(str),
        'Churn': churn_by_bin.values,
        'Non-Churn': non_churn_by_bin.values,
        'Population (%)': count_percent.values
    })

    # --- Barplot horizontal ---
    fig, ax = plt.subplots(figsize=(8,4))
    
    # Rouge pour churn, vert pour non-churn (stacked)
    ax.barh(plot_df['Segment'], plot_df['Churn'], color="#EA4335", label="Churn")
    ax.barh(plot_df['Segment'], plot_df['Non-Churn'], left=plot_df['Churn'], color="#34A853", label="Non-Churn")
    
    # Annoter chaque barre rouge avec taux churn
    for idx, row in plot_df.iterrows():
        # Annotation du taux de Churn (partie gauche en rouge)
        ax.text(row['Churn']/2, idx, f"{row['Churn']:.2f}", va='center', ha='center', color='white', fontsize=9)
        # Annotation du taux de Non-Churn (partie droite en vert)
        ax.text(row['Churn'] + row['Non-Churn']/2, idx, f"{row['Non-Churn']:.2f}", va='center', ha='center', color='white', fontsize=9)
    
    ax.set_xlabel("Proportion")
    ax.set_ylabel(feature)
    ax.set_title(f"Taux de Churn par segment de {feature}")
    ax.axvline(mean_churn_rate, color="#FBBC04", linestyle="--", label=f"Taux moyen ({mean_churn_rate:.2f})")
    ax.legend(loc='lower right')
    
    st.pyplot(fig)
    
    # 4. Créer recommandations pour segments à risque
    high_risk_bins = churn_by_bin[churn_by_bin >= churn_risk_threshold]
    for b, rate in high_risk_bins.items():
        pop_percent = count_percent.loc[b]
        risk_data_list.append({
            "Feature": feature,
            "Segment (Intervalle)": str(b),
            "Taux Churn Segment": rate,
            "Poids Population (%)": pop_percent,
            "Recommendation": f"Prioriser clients {feature} ∈ {b}"
        })
    


# ----------------------------
# 15) Export / recommandations / résumé
# ----------------------------
st.header("Résumé & Recommandations métier")

# ============================================================
# RECOMMANDATIONS OPÉRATIONNELLES DÉTAILLÉES
# ============================================================

st.header(" Recommandations Opérationnelles Détaillées")

st.markdown("""
---

##  **A. Stratégie de Modélisation et de Seuil (Tactique)**

### **Recommandation I.A : Ajuster le Seuil pour le Rappel**
- **Action :** Fixez un nouveau **seuil de décision** (par exemple, entre `0.15` et `0.25`) dans votre dashboard Streamlit afin d’augmenter le **rappel** à un niveau jugé acceptable (ex. > **85%**).  
- **Justification Métier :** En rétention, il est souvent plus rentable **d'accepter quelques faux positifs** (cibler des non-churners, coût marketing) que **de rater de vrais churners** (perte de revenu client).

### **Recommandation I.B : Suivi et Ré-entraînement**
- **Action :** Mettre le modèle en **production (via API)** et établir un **processus de ré-entraînement mensuel** pour capter l’évolution des comportements clients (notamment via les poids du `FeedbackText`).  

---

##  **B. Programme d’Engagement (Rétention Proactive)**

### **Recommandation II.A : Programme “Tenure”**
- **Action :** Mettre en place un **programme intensif et personnalisé** pour les clients dans le segment **[0, 6] mois**, identifiés comme les plus à risque de churn.  
- **Exemples :**  
  - Appels de suivi proactifs du support client  
  - Tutoriels personnalisés sur les fonctionnalités premium  
  - Offre d’un mois gratuit à l’approche du 6ᵉ mois  
- **Objectif :** Transformer la période critique d’essai en une **phase d’ancrage relationnel**.

### **Recommandation II.B : Conversion Contractuelle**
- **Action :** Cibler agressivement les clients en **contrat “Month-to-Month”** (notamment ceux avec un **SatisfactionScore faible**) avec des **incitations financières** pour migrer vers un contrat **1 an ou 2 ans**.  
- **Objectif :** Utiliser les **contrats longs comme leviers de fidélisation** et réduire le risque de churn à court terme.  

---

##  **C. Amélioration du Service (Qualité Produit et Support)**

### **Recommandation III.A : “Fiber Optic”**
- **Action :** Les équipes **techniques** et **produit** doivent prioriser l’amélioration de la **stabilité et de la qualité** pour les abonnés **Fiber Optic**.  
- **Justification :** Ces clients paient pour une performance élevée qu’ils **n’obtiennent pas toujours** — source majeure de désengagement.

### **Recommandation III.B : Alerte de Frustration (Support)**
- **Action :** Déployer une **règle métier automatique** :  
  - Tout client ayant **≥ 3 contacts support en 30 jours**  
  - Ou un **SatisfactionScore ≤ 2**  
  doit être **transféré en temps réel à un manager de rétention senior** pour une résolution immédiate et personnalisée.  
- **Objectif :** Intervenir **avant que la frustration ne devienne une résiliation**.

---

##  **D. Optimisation des Prix (Pricing & Marketing)**

### **Recommandation IV.A : Audit de la Valeur pour les Hauts Payeurs**
- **Action :** Réaliser un **audit des services** associés aux forfaits les plus chers.  
  Pour les clients à **MonthlyCharges élevés et identifiés à risque**, offrir **des fonctionnalités premium gratuites** afin de restaurer la perception de valeur.  
- **Justification :** Un **risque élevé lié au prix** indique souvent une **comparaison active avec la concurrence** — il faut **revaloriser l’offre avant la résiliation**.

---

 **Synthèse :**
> Ces recommandations traduisent les révélations du modèle IA en **actions métier concrètes** :  
> - Ajuster les seuils de décision pour maximiser la détection de churners  
> - Renforcer la rétention proactive sur les segments sensibles  
> - Améliorer le service et la satisfaction client avant la rupture  
> - Revaloriser la perception prix des clients premium
""")
