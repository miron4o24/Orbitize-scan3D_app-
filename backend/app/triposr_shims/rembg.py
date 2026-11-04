"""Заместител на rembg, който tsr/utils.py на TripoSR импортира: app/single3d.py сам изрязва предмета (маските на
app/segment.py), затова нищо тук никога не се извиква."""


def new_session(*args, **kwargs):
    raise NotImplementedError("rembg is not installed; app/single3d.py removes the background itself")


def remove(*args, **kwargs):
    raise NotImplementedError("rembg is not installed; app/single3d.py removes the background itself")
