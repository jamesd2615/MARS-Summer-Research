"""
GP tree generation with relaxed height constraints for strongly-typed GP.
Adapted from Ying Bi's BookCode/FlexGP implementation.
"""
import random
import sys
from inspect import isclass

__type__ = object


def genFull(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height
    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height or depth >= min_
    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=__type__):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        if len(expr) > 80:
            expr = []
            type_ = pset.ret
            stack = [(0, type_)]
            height = random.randint(min_, max_)
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
                if isclass(term):
                    term = term()
                expr.append(term)
            except (IndexError, KeyError):
                try:
                    depth -= 1
                    prim = random.choice(pset.primitives[type_])
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth, arg))
                except (IndexError, KeyError):
                    _, _, traceback = sys.exc_info()
                    raise IndexError(
                        "Cannot add a primitive of type '%s'." % (type_,)
                    ).with_traceback(traceback)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            except (IndexError, KeyError):
                try:
                    term = random.choice(pset.terminals[type_])
                    if isclass(term):
                        term = term()
                    expr.append(term)
                except (IndexError, KeyError):
                    _, _, traceback = sys.exc_info()
                    raise IndexError(
                        "Cannot add a terminal of type '%s'." % (type_,)
                    ).with_traceback(traceback)
    return expr


def genHalfAndHalfMD(pset, min_, max_, type_=None):
    expr = genHalfAndHalf(pset, min_, max_, type_=None)
    attempts = 0
    while len(expr) > 80 and attempts < 50:
        expr = genHalfAndHalf(pset, min_, max_, type_=None)
        attempts += 1
    return expr
