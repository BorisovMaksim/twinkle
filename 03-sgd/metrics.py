def mean_absolute_error(y_true, y_pred):
    return (abs(y_true - y_pred)).mean()


def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) **2).mean()

def r2_score(y_true, y_pred):
    raise NotImplementedError()
