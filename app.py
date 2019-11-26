import os

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from glimview.ModelKB import Model

app = Flask(__name__, static_url_path="", static_folder="dist", template_folder="dist")
app.config.from_mapping(
    MODEL=Model(
        os.environ["VOCAB_ENT"],
        os.environ["VOCAB_REL"],
        os.environ["MODEL_DIR"],
        os.environ["PATH_FILE"],
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
    triples = [(tri["head"], tri["relation"], tri["tail"]) for tri in request.json]
    vec = model.calc_phrase_from_triples(triples)
    targets = model.list_word
    sims = model.tvecs.dot(vec)
    if model.sim_with_path:
        path_sims = model.path_vecs.dot(vec)
        sims = np.hstack((sims, path_sims))
        targets.extend(model.paths)
    top_inds = np.argpartition(sims, -20)[-20:]
    most_similar = sorted([(sims[i], targets[i]) for i in top_inds], reverse=True)
    return jsonify([dict(target=tgt, similarity=sim) for sim, tgt in most_similar])


@app.route("/")
def hello():
    return render_template("index.html")
