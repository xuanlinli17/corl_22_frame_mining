import pickle, importlib
from .base import BaseFileHandler
from ....meta import get_filename_suffix


class PickleProtocol:
    def __init__(self, level):
        self.previous = pickle.HIGHEST_PROTOCOL
        self.level = level

    def __enter__(self):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.level

    def __exit__(self, *exc):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.previous


class PickleHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('protocol', 5)
        pickle.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('protocol', 5)
        return pickle.dumps(obj, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        file_suffix = get_filename_suffix(filepath)
        if file_suffix == 'pkl':
            with open(filepath, 'rb') as f:
                return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        file_suffix = get_filename_suffix(filepath)
        if file_suffix == 'pkl':
            with open(filepath, 'wb') as f:
                return self.dump_to_fileobj(obj, f, **kwargs)
