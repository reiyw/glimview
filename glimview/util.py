#%%
from itertools import islice
from functools import partial
from typing import List

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def chunked(iterable, n):
    return iter(partial(take, n, iter(iterable)), [])

def build_path_expr(path: List[str]) -> str:
    expr = path[0]
    for rel, ent in chunked(path[1:], 2):
        # TODO: "＋" の代わりの記号／仕組み
        expr = f'trans({expr}, {rel}) ＋ {ent}'
    return expr