from flask import Flask, jsonify, request, send_from_directory

from main import gerar_resposta

PDF_DIR = "files"
PDF_NAME = "manual_fazerfz25abs_2023.pdf"

app = Flask(__name__, static_folder="static")


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/pdf")
def pdf():
    return send_from_directory(PDF_DIR, PDF_NAME)


@app.post("/api/chat")
def chat():
    pergunta = (request.json or {}).get("pergunta", "").strip()
    if not pergunta:
        return jsonify(erro="Pergunta vazia"), 400
    texto, resultados, uso = gerar_resposta(pergunta)
    refs = [{"pagina": c["pagina"], "score": round(s, 3), "trecho": c["texto"][:220]}
            for c, s in resultados]
    return jsonify(resposta=texto, referencias=refs, uso=uso)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
