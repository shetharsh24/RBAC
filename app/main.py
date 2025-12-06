from typing import Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama


app = FastAPI()

# serve the frontend static files from app/static
current_dir = Path(__file__).parent.resolve()
static_dir = current_dir / "static"
print(f"[startup] static_dir={static_dir} exists={static_dir.exists()}")


static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

security = HTTPBasic()

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"}
}


# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    user = users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": user["role"]}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

#Load Documents
def load_documents(role: str):
    current_dir = Path(__file__).parent.parent
    base_path = current_dir / "resources" / "data" / role
    
    print(f"Looking for documents in: {base_path}")
    print(f"Path exists: {base_path.exists()}")
    documents = []

    if not base_path.exists():
        return f"No documents found for role: {role}"
    
    try:
        file_patterns = ["*.txt", "*.pdf", "*.docx", "*.md", "*.json", "*.csv"]

        for pattern in file_patterns:
            for file_path in base_path.rglob(pattern):
                try:
                    content = read_file(file_path)
                    documents.append(f"[{file_path.name}]\n{content}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
    except Exception as e:
        return f"Error loading documents: {e}"
    
    return "\n\n---\n\n".join(documents) if documents else f"No documents found for role: {role}"


def read_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()
    
    if extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif extension == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif extension == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif extension == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif extension == ".pdf":
        return "[PDF file detected - requires PyPDF2 or pdfplumber library]"
    
    elif extension == ".docx":
        return "[DOCX file detected - requires python-docx library]"
    
    else:
        return f"[Unsupported file type: {extension}]"


def create_rag_response(documents: str, user_message: str) -> str:

    try:
    # Initialize text splitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_text(documents)

    # Initialize embeddings and vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_texts(chunks, embeddings)

    # Initialize LLM and generate response

        # Create RAG response
        relevant_docs = vector_store.similarity_search(user_message, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])


        llm = Ollama(model="mistral")
        prompt = f"""Based on the following context, answer the user's question.
        
        Context:
        {context}

        User Question: {user_message}

        Answer:"""
        
        response = llm.invoke(prompt)
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Protected chat endpoint
@app.post("/chat")
def query(user=Depends(authenticate), message: str = "Hello"):
    username = user['username']
    role = user['role']

    documents = load_documents(role)

    if isinstance(documents, str) and "No documents" in documents:
        response = documents
    else:
        response = create_rag_response(documents, message)

    return {
            "username": username,
            "role": role,
            "message": message,
            "response": response
        }