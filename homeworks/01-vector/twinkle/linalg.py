def vector_norm(vec, ord=2):
    return sum([abs(x) ** ord for x in vec]) ** (1/ord)
