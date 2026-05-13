# ecs7019u-group4

# UK Demographic Marketing Dashboard

This project is a Streamlit-based data science application that uses clustering and generative AI to segment UK postcode areas and generate personalised marketing strategies.

It combines:
- KMeans, GMM, and Hybrid clustering models
- UMAP-based visualisation
- Interactive Streamlit dashboard
- Groq LLaMA 3.1 API for AI-generated marketing recommendations

---

##  Project Aim

The aim of this project is to:
- Segment UK postcode areas into meaningful demographic clusters
- Analyse socioeconomic patterns across clusters
- Generate AI-driven marketing strategies tailored to each segment
- Provide an interactive dashboard for exploring results

---

##  System Overview

The system works as follows:

1. User enters a postcode and product description
2. Postcode is mapped to a demographic cluster
3. Cluster statistics are retrieved
4. AI model generates marketing strategy using Groq API
5. Results are displayed in the Streamlit dashboard

Two approaches are used:
- **KMeans-based strategy** (single cluster)
- **Hybrid strategy** (probabilistic cluster blend)

---

##  Groq API Setup (REQUIRED)

This project uses the Groq API for generating AI marketing recommendations.

### Step 1: Create API Key
https://console.groq.com/

### Step 2: Add to Streamlit/secret.toml

### GROQ_API_KEY = "your_api_key_here"
### the application wont run without the API key
---
## Installation
### dependencies need to be installed manually:
pip install pandas numpy scikit-learn streamlit plotly joblib groq
---
## Dataset setup

before running any commands ensure the file notebooks/processed/features_by_oa.csv exists if not you can download it from 
https://drive.google.com/file/d/14Z9nwQdCYclowMATAxulcI90y69wvarn/view?usp=drivesdk 
and upload in notebooks/processed/
---
## Model training pipeline
the clustering models must be trained before running the pipeline as follows;
1. python scripts/train_kmeans.py --kmin 2 --kmax 15 --k 4
2. python scripts/train_gmm.py --kmin 2 --kmax 15 --k 4
3. python scripts/train_hybrid.py
---
## After the training is complete launch the streamlit app by:
streamlit run app.py
---
## You can:
1. Enter the postcode
2. Enter the product description
3. View AI generated marejting stratergies and communication channels based on the cluster information pased to it as prompts.
---
## Authors/Group Members
This project was developed as group effort by:
1. Keerthana Jayakumar Natarajan
2. Aleksander Skutnik
3. Normala Binti Shamsuddin
4. Aryaa Abhay Choudhari
