"""Stand-in for rembg, which TripoSR's tsr/utils.py imports: app/single3d.py cuts the object out itself
(app/segment.py's masks), so nothing here is ever called."""


def new_session(*args, **kwargs):
    raise NotImplementedError("rembg is not installed; app/single3d.py removes the background itself")


def remove(*args, **kwargs):
    raise NotImplementedError("rembg is not installed; app/single3d.py removes the background itself")
