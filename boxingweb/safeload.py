import pickle, numpy as np
class Safe(pickle.Unpickler):
    OK = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy._core.multiarray', '_reconstruct')}
    def find_class(self, module, name):
        if (module, name) not in self.OK:
            raise pickle.UnpicklingError(f'blocked {module}.{name}')
        return super().find_class(module, name)
def load(path):
    with open(path, 'rb') as f:
        return Safe(f).load()
