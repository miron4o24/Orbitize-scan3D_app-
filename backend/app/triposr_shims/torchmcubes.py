"""Заместител на torchmcubes (изисква компилиране на Windows), който tsr/models/isosurface.py на TripoSR импортира:
app/single3d.py сам изпълнява marching cubes (scikit-image), затова това никога не се извиква."""


def marching_cubes(*args, **kwargs):
    raise NotImplementedError("torchmcubes is not installed; app/single3d.py extracts the surface itself")
