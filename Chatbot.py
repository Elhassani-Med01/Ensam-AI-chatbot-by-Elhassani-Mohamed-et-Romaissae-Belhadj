import os
import streamlit as st
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA



# Charger les variables d'environnement
load_dotenv()

# Initialiser les modèles LLM et embeddings
llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434")
embed_model = OllamaEmbeddings(model="mistral", base_url="http://127.0.0.1:11434")  # Embedding model updated here

file_path = 'C:/Users/nelba/Desktop/Chatbot/Chatbot/New.txt'

# Read the content of the text file
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()
# Diviser le texte en morceaux
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1350, chunk_overlap=110)
chunks = text_splitter.split_text(file_content)  # Use file content here
vector_store_path='C:/Users/nelba/Desktop/Chatbot/Chatbot/VDB'
if not os.path.exists(vector_store_path):
    os.makedirs(vector_store_path)
    print("Dossier pour la base de vecteurs créé avec succès !")

# Charger ou créer une base de vecteurs
try:
    if os.listdir(vector_store_path):  # Vérifie si le dossier contient des fichiers
        # Charger les vecteurs sauvegardés
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embed_model)
        print("Base de vecteurs chargée avec succès !")
    else:
        # Créer une nouvelle base de vecteurs si le dossier est vide
        vector_store = Chroma.from_texts(chunks, embed_model, persist_directory=vector_store_path)
        vector_store.persist()  # Sauvegarder la base de vecteurs
        print("Base de vecteurs créée et sauvegardée avec succès !")
except Exception as e:
    print(f"Une erreur s'est produite lors du chargement ou de la création de la base de vecteurs : {e}")


# Créer un retrieveur à partir de la base
retriever = vector_store.as_retriever()

# Créer la chaîne RetrievalQA avec une gestion des réponses inconnues
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True  # Include this to inspect sources if needed
)

# Interface utilisateur avec Streamlit
st.title("Application Interactive - Questions sur ENSAM")
st.write("Posez une question et obtenez une réponse pertinente !")

# Champ de saisie pour les questions
user_question = st.text_input("Votre question :", placeholder="Posez une question ici...")

if st.button("Obtenir la réponse"):
    if user_question.strip():
        with st.spinner("Traitement de la question..."):
            try:
                # Construire le prompt avec les instructions
                prompt = f""""Tu es un assistant virtuel spécialisé dans les questions relatives à l'ENSAM (École Nationale Supérieure d'Arts et Métiers). Répond en français uniquement en te basant sur les informations disponibles dans la base de données fournie. Si la réponse à une question n'est pas dans la base de données, indique clairement que tu ne disposes pas de l'information. Reste concis et précis dans tes réponses.

                        Question de l'utilisateur : {user_question}"
                 """

                # Exécuter la chaîne avec le prompt construit
                response = qa_chain({"query": prompt.strip()})  # Utilise le prompt construit ici
                result = response.get("result", "").strip()
                source_docs = response.get("source_documents", [])

                if result:
                    st.success("Voici la réponse :")
                    st.write(result)
                    if source_docs:
                        st.subheader("Documents Source")
                        for i, doc in enumerate(source_docs, 1):
                            st.write(f"Source {i}: {doc.page_content}")
                else:
                    st.warning("Désolé, je ne peux pas répondre à cette question en fonction des données fournies.")
            except Exception as e:
                st.error(f"Une erreur s'est produite : {e}")
    else:
        st.error("Veuillez entrer une question valide.")



