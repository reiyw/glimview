from __future__ import absolute_import, division, print_function

import json
import os
import sys

import numpy as np

from .utilityFuncs import readerLine, show_top, split_wrt_brackets
from .util import build_path_expr


class Model(object):
    def __init__(self, words, roles, path, path_file):
        # load lexicon
        self.list_word = [line.split("\t", 1)[0] for line in readerLine(words)]
        self.dict_word = dict((s, i) for i, s in enumerate(self.list_word))

        list_role_pre = [line.split("\t", 1)[0] for line in readerLine(roles)]
        self.list_role = [x + ">" for x in list_role_pre] + [
            x + "<" for x in list_role_pre
        ]

        self.dict_role = dict((s, i) for i, s in enumerate(self.list_role))

        # vecs & mats
        if os.path.exists(path + "tvecs.npy"):
            tvecs = np.load(path + "tvecs.npy")
        else:
            tvecs = np.loadtxt(path + "tvecs.txt")
            np.save(path + "tvecs.npy", tvecs)
        tvecs /= np.sqrt(np.sum(np.square(tvecs), axis=1, keepdims=True))
        dim = tvecs.shape[1]

        if os.path.exists(path + "cvecs.npy"):
            cvecs = np.load(path + "cvecs.npy")
        else:
            cvecs = np.loadtxt(path + "cvecs.txt")
            np.save(path + "cvecs.npy", cvecs)
        with open(path + "params.json") as params_file:
            params = json.load(params_file)
        cvecs /= np.expand_dims(
            1.0
            + np.loadtxt(path + "vsteps.txt").astype("float32")[: cvecs.shape[0]]
            * params["vEL"],
            axis=1,
        )

        if os.path.exists(path + "mats.npy"):
            mats = np.load(path + "mats.npy")
        else:
            # np.loadtxt cannot load 3d array, so reshape after loading
            mats = np.loadtxt(path + "mats.txt").reshape(
                -1, params["dim"], params["dim"]
            )
            np.save(path + "mats.npy", mats)
        mats *= np.sqrt(dim / np.sum(np.square(mats), axis=(1, 2), keepdims=True))

        self.tvecs = tvecs
        self.dim = dim
        self.mats = mats
        self.cvecs = cvecs
        denc_scal = 1.0 / (
            1.0 + np.loadtxt(path + "dstep.txt").astype("float32") * params["autoEL"]
        )

        if os.path.exists(path + "encoder.npy"):
            self.encoder = (
                np.load(path + "encoder.npy").reshape((-1, dim * dim)) * denc_scal
            )
        else:
            enc = np.loadtxt(path + "encoder.txt")
            np.save(path + "encoder.npy", enc)
            self.encoder = enc.reshape((-1, dim * dim)) * denc_scal

        if os.path.exists(path + "decoder.npy"):
            self.decoder = (
                np.load(path + "decoder.npy").reshape((-1, dim * dim)) * denc_scal
            )
        else:
            dec = np.loadtxt(path + "decoder.txt")
            np.save(path + "decoder.npy", dec)
            self.decoder = dec.reshape((-1, dim * dim)) * denc_scal

        self.msteps = np.loadtxt(path + "msteps.txt", dtype=np.uint)

        if os.path.exists(path_file):
            paths = []
            path_vecs = []
            for path_repr in open(path_file):
                path_repr = path_repr.rstrip()
                expr = build_path_expr(path_repr.split("\t"))
                vec = self.calc(expr)
                paths.append(path_repr)
                path_vecs.append(vec)
            self.paths = paths
            self.path_vecs = np.asarray(path_vecs)
            self.sim_with_path = True
        else:
            self.sim_with_path = False

        print(
            "Loaded model. # of relations: {}  # of entities: {}".format(
                len(list_role_pre), len(self.list_word)
            ),
            file=sys.stderr,
        )

    def get_word_vec(self, word):
        return self.tvecs[self.dict_word[word]]

    def trans(self, v, role):
        m = self.mats[self.dict_role[role]]
        v = m.dot(v)
        v /= np.sqrt(np.sum(np.square(v)))
        return v

    def calc(self, expr):
        tosum = []
        # TODO: "＋" の代わりの記号／仕組み
        for pre in split_wrt_brackets(expr, "＋"):
            s = pre.strip()
            if s.startswith("trans(") and s.endswith(")"):
                ss, r = s[len("trans(") : -len(")")].rsplit(", ", 1)
                tosum.append(self.trans(self.calc(ss), r))
            elif s.startswith("(") and s.endswith(")"):
                tosum.append(self.calc(s[1:-1]))
            else:
                tosum.append(self.get_word_vec(s))
        ret = np.sum(tosum, axis=0)
        ret /= np.sqrt(np.sum(np.square(ret)))
        return ret

    def show_v(self, v, k):
        tsim = self.tvecs.dot(v)
        print("Similar Targets:")
        show_top(k, tsim, self.list_word)
        print()
        csim = self.cvecs.dot(v)
        print("Strong Contexts:")
        show_top(k, csim, self.list_word)
        print()

    def show_m(self, r, k):
        prj = self.mats[self.dict_role[r]]

        def calc_deform(x):
            mean = x.trace() / self.dim
            y = x - np.diagflat(np.full(self.dim, mean, dtype=np.float32))
            return np.sqrt(np.sum(np.square(y))), mean

        p_dfm, p_mtr = calc_deform(prj)
        prj_dfm = calc_deform(np.dot(prj, prj.T))[0]

        def code_relu(x):
            code = np.minimum(x, 4.0 * np.sqrt(self.dim))
            code_hinge = np.maximum(0.5 + 0.25 * code, 0.0)
            code_grad = np.minimum(code_hinge, 1.0)
            return code_grad * np.maximum(2.0 * code_hinge, code)

        prj_code = code_relu(self.encoder.dot(prj.flatten()))
        prj_dec = self.decoder.transpose().dot(prj_code)
        prj_dec_norm = np.sqrt(self.dim / np.sum(np.square(prj_dec)))
        prj_dec *= prj_dec_norm
        prj_err = prj_dec.dot(prj.flatten()) / self.dim

        prj_sim = (
            self.mats.reshape((-1, self.dim * self.dim)).dot(prj.flatten()) / self.dim
        )

        print("Matrix non-diagonal:  " + str(p_dfm))
        print("Matrix diagonal:    " + str(p_mtr))
        print("Skewness of Matrix:   " + str(prj_dfm))
        print("Dec norm:       " + str(prj_dec_norm))
        print("Decoding cos:     " + str(prj_err))
        print()
        print("Matrix code:")
        print(prj_code)
        print()
        print("Step:         " + str(self.msteps[self.dict_role[r]]))
        print()

        print("Similar Roles:")
        show_top(k, prj_sim, self.list_role)
        print()

    def get_score(self, head, relation, direction):
        ti = self.dict_word[head]
        ri = self.dict_role[relation + (">" if direction else "<")]
        vec = self.mats[ri].dot(self.tvecs[ti])
        return self.cvecs.dot(vec)

    def show_mm(self, r1, r2, k):
        m = self.mats[self.dict_role[r1]].dot(self.mats[self.dict_role[r2]])
        sim = self.mats.reshape((-1, self.dim * self.dim)).dot(m.flatten()) / self.dim
        print("Similar Roles:")
        show_top(k, sim, self.list_role)
        print()

    def mm_rank(self, r1, r2, r):
        from scipy.stats import rankdata

        m = self.mats[self.dict_role[r1]].dot(self.mats[self.dict_role[r2]])
        sim = self.mats.reshape((-1, self.dim * self.dim)).dot(
            m.flatten()
        )  # / self.dim
        return rankdata(-sim)[self.dict_role[r]]

    def code_of(self, r):
        def code_relu(x):
            code = np.minimum(x, 4.0 * np.sqrt(self.dim))
            code_hinge = np.maximum(0.5 + 0.25 * code, 0.0)
            code_grad = np.minimum(code_hinge, 1.0)
            return code_grad * np.maximum(2.0 * code_hinge, code)

        prj = self.mats[self.dict_role[r]]
        prj_code = code_relu(self.encoder.dot(prj.flatten()))
        return prj_code
