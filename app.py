import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from solver import solve, get_cache

app = Flask(__name__)
CORS(app)

# 10MB limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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
                content = uploaded.read().decode('utf-8')
                payload = json.loads(content)
        elif 'dataset' in request.files:
            uploaded = request.files['dataset']
            if uploaded.filename != '':
                content = uploaded.read().decode('utf-8')
                payload = json.loads(content)

        # Or JSON in body
        if payload is None:
            if request.is_json:
                payload = request.get_json()
            elif request.data:
                try:
                    payload = json.loads(request.data.decode('utf-8'))
                except Exception:
                    payload = None

        result = solve(payload)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
