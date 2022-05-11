def vector_norm(self, ord=2):
    return sum([abs(x) ** ord for x in self._data]) ** (1/ord)
