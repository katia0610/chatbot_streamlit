import pandas as pd
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore as qd
from langchain.document_loaders import CSVLoader
from qdrant_client.models import Distance, VectorParams
import os

# Constants
API_TOKEN = os.getenv('API_TOKEN')
MBD_MODEL = 'intfloat/multilingual-e5-large'
QDRANT_URL = os.getenv('QDRANT_URL')
COLLECTION_NAME  = os.getenv('COLLECTION_NAME')
API_KEY=os.getenv('API_KEY')

# Charger les données du CSV avec CSVLoader
loader = CSVLoader(
    file_path='../data/icecat.csv',
    metadata_columns=["Marque", "Categorie"],
    csv_args={"delimiter": ","},
    encoding="utf-8"
)
documents = loader.load()

# Initialisez les embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
# Définir les paramètres de la collection
vector_params = VectorParams(
    size=1024,  # Remplacez par la taille des vecteurs que vous utilisez
    distance=Distance.COSINE
)
db = qd.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=API_KEY,
    collection_name=COLLECTION_NAME,
    prefer_grpc=True,
    vector_name='vector_params'  # Utilisez le nom du vecteur dense configuré
)

# Ajouter les documents à la collection
db.add_documents(documents)
print("Documents ajoutés à la collection Qdrant.")