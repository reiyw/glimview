import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from glimview.ModelKB import Model

app = Flask(__name__, template_folder="./dist")
app.config.from_mapping(
    MODEL=Model(
        os.environ["VOCAB_ENT"], os.environ["VOCAB_REL"], os.environ["MODEL_DIR"]
    )
)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/entities")
def get_entities():
    return jsonify(app.config["MODEL"].list_word)


@app.route("/api/relations")
def get_relations():
    return jsonify(app.config["MODEL"].list_role)


@app.route("/api/query", methods=["POST"])
def query():
    model: Model = app.config["MODEL"]
    expr = []
    for i, triple in enumerate(request.json):
        if i > 0:
            expr.append(" + ")
        expr.append(f"trans({triple['head']}, {triple['relation']})")
        if triple["tail"]:
            expr.append(f" + {triple['tail']}")

    tsim = model.tvecs.dot(model.calc("".join(expr)))
    top_inds = np.argpartition(tsim, -20)[-20:]
    most_similar = sorted(
        [(tsim[i], model.list_word[i]) for i in top_inds], reverse=True
    )
    return jsonify([dict(target=tgt, similarity=sim) for sim, tgt in most_similar])


@app.route("/")
def hello():
    return render_template("index.html")
