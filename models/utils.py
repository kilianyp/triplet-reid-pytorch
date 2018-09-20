import sys

def import_from_path(path, args):
    """Imports a model from a specific path. Returns None if no model is found.

    path: File path to the directory where the .py file is placed.
    args: dictionary with model arguments.
    """
    sys.path.append(path)
    try:
        module = __import__(args["model"])
    except ModuleNotFoundError:
        return None, None
    init_fn = getattr(module, args["model"])
    model_params = getattr(module, "model_parameters")
    model_params.update(args)
    # clean up
    sys.path.remove(path)
    del module
    del sys.modules[args["model"]]
    return init_fn(**model_params)
