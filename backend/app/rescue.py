"""Дълбочина от Depth Anything 3 (DA3-Base, Apache-2.0, на процесора) за кадрите, чиято AI дълбочина не пасва на
измерената (app/fuse.py completed): лъскава тъмна пластмаса, кадри почти на нивото на масата, блясък от прозорец.
Тогава измерената дълбочина на OpenMVS е шум и сливането пропускаше тези кадри, а с тях цяла страна на предмета
(ролетка, 2026-11-08: 16 от 35 кадъра). DA3 вижда няколко кадъра наведнъж с известните им положения и дава
съгласувана дълбочина за всеки; сливането я взима само за пропуснатите кадри (страната, която иначе е само от
очертанията). При Фродо, където всички кадри пасват, не се пуска: там измерената дълбочина е по-точна (сравнено на кадри
извън сливането DA3 вместо Depth Anything V2 не променя нищо: 0,507% срещу 0,503% от размера).

Като собствен процес (стъпка "rescuing depth"): python rescue.py <папка с изображения> <inputs.npz> <out.npz>.
inputs.npz: names, R (n, 3, 3), t (n, 3), K (n, 4) fx fy cx cy (COLMAP, изображенията в папката), pick (индексите за
DA3), want (тези от тях, чиято дълбочина трябва). out.npz: depth_i (h, w) за всеки i от want, за цялото изображение
при обработваната разделителна способност (app/fuse.py read_views я пренася в изрязването)."""

import os
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from . import depth as D
    from . import fuse as F
    from . import parallel as PAR
else:  # изпълнение като скрипт: импортира пакета по пътя му
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import depth as D
    from app import fuse as F
    from app import parallel as PAR

TOOLS = Path(os.environ.get("DA3_DIR", Path(__file__).resolve().parent.parent / "tools" / "da3"))
CODE = TOOLS / "code" / "src"  # src/ на хранилището ByteDance-Seed/Depth-Anything-3 (scripts/models.ps1)
WEIGHTS = TOOLS / "DA3-BASE"  # config.json и model.safetensors
RES = 504  # дългата страна на кадъра, която DA3 вижда (по подразбиране при него)
MOST = 18  # най-много кадри наведнъж (време: около 3 минути на лаптоп с 6 нишки, по-малко на сървъра)
CONTEXT = 6  # поне толкова пасващи кадри за опора, разпределени по видеото
MIN_LEFT, MIN_SHARE = 3, 0.1  # пуска се, когато поне толкова (и такъв дял) кадри не пасват


def ready() -> bool:
    return (CODE / "depth_anything_3" / "api.py").exists() and (WEIGHTS / "model.safetensors").exists()


def failing(dense: Path, names: list[str], crops_file: Path) -> list[int]:
    """Индексите на кадрите, чиято AI дълбочина няма да пасне на измерената (правилото на fuse.completed, без
    обвивката: изрязването и вътрешността на маската)."""
    maps = F.dmap_files(dense)
    with np.load(crops_file) as crops:
        todo = [(i, maps[n], crops[f"disp_{i}"], tuple(int(v) for v in crops[f"box_{i}"]), crops[f"inner_{i}"])
                for i, n in enumerate(names) if f"disp_{i}" in crops and n in maps]  # fmt: skip

    def one(item):
        i, path, disp, (x0, y0, x1, y1), inner = item
        dm = F.at_image_size(F.read_dmap(path))
        dep = dm.depth[y0:y1, x0:x1]
        if dep.shape != disp.shape:
            return None
        conf = dm.conf[y0:y1, x0:x1] if dm.conf is not None else np.ones_like(dep)
        on = inner & (dep > 0) & (conf >= F.MIN_CONF)
        if on.sum() < 500:
            return i
        (a, _), err = D.fit_scale(disp[on], 1 / dep[on], np.ones(int(on.sum())))
        return i if a <= 0 or err > D.MAX_ERROR else None

    return sorted(i for i in PAR.each(one, todo) if i is not None)


def chosen(n: int, bad: list[int]) -> list[int]:
    """Кадрите за DA3: неподходящите (най-много MOST - CONTEXT, равномерно) и пасващи за опора до MOST."""
    bad = [bad[k] for k in np.linspace(0, len(bad) - 1, min(len(bad), MOST - CONTEXT)).round().astype(int)]
    good = [i for i in range(n) if i not in set(bad)]
    extra = MOST - len(bad)
    good = [good[k] for k in np.linspace(0, len(good) - 1, min(len(good), extra)).round().astype(int)] if good else []
    return sorted(set(bad) | set(good))


def vision_stubs(force: bool = False) -> None:
    """torchvision и cv2 са нужни на DA3 само за подготовката на кадъра (нормализиране, изрязване, смяна на размера).
    Без тях (сървърът не ги инсталира: torchvision би сменил вече заредения torch при обновяване) ги заместват тези."""
    import torch

    try:
        if force:
            raise ImportError
        import torchvision.transforms  # noqa: F401
    except ImportError:
        tv = types.ModuleType("torchvision")
        T = types.ModuleType("torchvision.transforms")

        class ToTensor:
            def __call__(self, img):
                return torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1).copy() / 255)

        class Normalize:
            def __init__(self, mean, std):
                self.mean = torch.tensor(mean).view(-1, 1, 1)
                self.std = torch.tensor(std).view(-1, 1, 1)

            def __call__(self, t):
                return (t - self.mean) / self.std

        class CenterCrop:
            def __init__(self, size):
                self.h, self.w = size

            def __call__(self, t):
                H, W = t.shape[-2:]
                y, x = (H - self.h) // 2, (W - self.w) // 2
                return t[..., y : y + self.h, x : x + self.w]

        T.ToTensor, T.Normalize, T.CenterCrop = ToTensor, Normalize, CenterCrop
        tv.transforms = T
        sys.modules["torchvision"], sys.modules["torchvision.transforms"] = tv, T
    try:
        if force:
            raise ImportError
        import cv2  # noqa: F401
    except ImportError:
        cv = types.ModuleType("cv2")
        cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA = 1, 2, 3

        def resize(arr, size, interpolation=1):
            way = {2: Image.BICUBIC, 3: Image.BOX}.get(interpolation, Image.BILINEAR)
            return np.asarray(Image.fromarray(np.asarray(arr)).resize(tuple(int(s) for s in size), way))

        cv.resize = resize
        sys.modules["cv2"] = cv


def model():
    """DA3 от tools/da3: кодът му внася модули само за износ и за подравняване с evo, които тук не трябват."""
    vision_stubs(os.environ.get("DA3_STUBS") == "1")
    sys.path.insert(0, str(CODE))
    traj = types.ModuleType("evo.core.trajectory")

    class PosePath3D:  # подравняването на положенията (Umeyama), само колкото го ползва DA3
        def __init__(self, poses_se3):
            self.poses_se3 = [p.copy() for p in poses_se3]

        def align(self, ref, correct_scale=True):
            x = np.stack([p[:3, 3] for p in self.poses_se3])
            y = np.stack([p[:3, 3] for p in ref.poses_se3])
            mx, my = x.mean(0), y.mean(0)
            U, Dg, Vt = np.linalg.svd((y - my).T @ (x - mx) / len(x))
            S = np.eye(3)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                S[2, 2] = -1
            r = U @ S @ Vt
            s = np.trace(np.diag(Dg) @ S) / (((x - mx) ** 2).sum() / len(x)) if correct_scale else 1.0
            t = my - s * r @ mx
            for p in self.poses_se3:
                p[:3, :3] = r @ p[:3, :3]
                p[:3, 3] = r @ (s * p[:3, 3]) + t
            return r, t, s

    traj.PosePath3D = PosePath3D
    sys.modules.setdefault("evo", types.ModuleType("evo"))
    sys.modules.setdefault("evo.core", types.ModuleType("evo.core"))
    sys.modules["evo.core.trajectory"] = traj
    export = types.ModuleType("depth_anything_3.utils.export")
    export.export = lambda *a, **k: None
    sys.modules["depth_anything_3.utils.export"] = export
    import torch

    from depth_anything_3.api import DepthAnything3

    torch.set_num_threads(PAR.CPUS)
    return DepthAnything3.from_pretrained(str(WEIGHTS)).eval()


def main(images: Path, inputs: Path, out: Path) -> None:
    import torch

    z = np.load(inputs)
    names, pick, want = [str(n) for n in z["names"]], [int(i) for i in z["pick"]], {int(i) for i in z["want"]}
    frames, ext, intr = [], [], []
    for i in pick:
        with Image.open(images / names[i]) as im:
            frames.append(np.asarray(im.convert("RGB")))
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = z["R"][i], z["t"][i]
        fx, fy, cx, cy = z["K"][i]
        ext.append(E)
        intr.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    started = time.time()
    net = model()
    with torch.inference_mode():
        pred = net.inference(frames, extrinsics=np.stack(ext).astype(np.float32),
                             intrinsics=np.stack(intr).astype(np.float32), process_res=RES)  # fmt: skip
    print(f"DA3: {len(pick)} frames in {time.time() - started:.0f} s", flush=True)
    np.savez(out, **{f"depth_{i}": pred.depth[k].astype(np.float32) for k, i in enumerate(pick) if i in want})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
