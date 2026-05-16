import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from rag.loader import load_and_chunk_pdf
from rag.embedder import create_vector_store
from rag.chain import build_qa_chain, ask_question

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global vector store — in production you'd persist this per session/user
vector_store = None
qa_chain = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts a PDF file, chunks it, embeds it, and stores in FAISS.
    Returns a success/error JSON response.
    """
    global vector_store, qa_chain

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # Step 1: Load PDF and split into chunks
        chunks = load_and_chunk_pdf(filepath)

        # Step 2: Embed chunks and store in FAISS
        vector_store = create_vector_store(chunks)

        # Step 3: Build the QA chain with the vector store
        qa_chain = build_qa_chain(vector_store)

        return jsonify({
            "message": f"Successfully processed '{filename}'",
            "chunks": len(chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts a JSON body with a 'question' field.
    Returns the LLM answer and the source chunks used.
    """
    global qa_chain

    if qa_chain is None:
        return jsonify({"error": "Please upload a document first"}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        result = ask_question(qa_chain, question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 0.0.0.0 is required for deployment — don't use localhost
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)