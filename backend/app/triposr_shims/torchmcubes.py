"""Stand-in for torchmcubes (needs compiling on Windows), which TripoSR's tsr/models/isosurface.py imports:
app/single3d.py runs marching cubes itself (scikit-image), so this is never called."""


def marching_cubes(*args, **kwargs):
    raise NotImplementedError("torchmcubes is not installed; app/single3d.py extracts the surface itself")
