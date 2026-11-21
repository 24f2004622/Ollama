import os
import re
import json
import ast
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from solver import solve, get_cache

app = Flask(__name__)
CORS(app)

# 10MB limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def parse_relaxed_json(text_data):
    if not text_data:
        return None
    if isinstance(text_data, bytes):
        text_data = text_data.decode('utf-8', errors='replace')
    text_data = text_data.strip()
    if not text_data:
        return None
    # 1. Standard JSON
    try:
        return json.loads(text_data)
    except Exception:
        pass
    # 2. JSON with trailing commas stripped
    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', text_data)
        return json.loads(cleaned)
    except Exception:
        pass
    # 3. Python-style literal eval (handles single quotes, True/False/None)
    try:
        res = ast.literal_eval(text_data)
        if isinstance(res, (dict, list)):
            return res
    except Exception:
        pass
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "dual-embedding-solver",
        "models": ["nomic-embed-text", "mxbai-embed-large"],
        "metric": "Spearman rank correlation (rho)"
    })

@app.route('/api/default', methods=['GET'])
def get_default():
    try:
        cache = get_cache()
        return jsonify({
            "status": "success",
            "sentences": cache["sentences"],
            "queries": cache["queries"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/solve', methods=['POST'])
def handle_solve():
    try:
        payload = None

        # Check if file was uploaded
        if 'file' in request.files:
            uploaded = request.files['file']
            if uploaded.filename != '':
                raw = uploaded.read()
                payload = parse_relaxed_json(raw)
        elif 'dataset' in request.files:
            uploaded = request.files['dataset']
            if uploaded.filename != '':
                raw = uploaded.read()
                payload = parse_relaxed_json(raw)

        # Or JSON in body
        if payload is None:
            if request.is_json:
                payload = request.get_json(silent=True)
            if payload is None and request.data:
                payload = parse_relaxed_json(request.data)
            if payload is None and request.form and 'jsonInput' in request.form:
                payload = parse_relaxed_json(request.form['jsonInput'])

        result = solve(payload)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
