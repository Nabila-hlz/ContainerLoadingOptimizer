from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
import threading
import time
import contextlib
import io
import json
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ast
import math
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from copy import deepcopy

NOTEBOOK_PATH = Path(__file__).parent / "notebooks" / "NoteBook.ipynb"

def _node_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(dec.lineno for dec in node.decorator_list)
    end = node.end_lineno
    return "\n".join(lines[start - 1:end])

@dataclass
class Box:
    id: int
    length: float
    width: float
    height: float
    weight_kg: float
    fragile: bool = False

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def get_orientations(self) -> List[Tuple[float, float, float]]:
        l, w, h = self.length, self.width, self.height
        if self.fragile:
            return [(l, w, h), (w, l, h)]
        return [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l),
        ]

    def orientation_labels(self) -> List[str]:
        return [f"{o[0]}x{o[1]}x{o[2]} cm" for o in self.get_orientations()]


@dataclass
class Container:
    name: str
    length: float
    width: float
    height: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


PRESET_CONTAINERS: List[Container] = [
    Container("ISO 20ft Container", 589.0, 235.0, 239.0),
    Container("ISO 40ft Container", 1203.0, 235.0, 239.0),
    Container("Standard Truck", 600.0, 240.0, 250.0),
    Container("Delivery Van", 250.0, 160.0, 160.0),
    Container("Pallet Box", 120.0, 80.0, 100.0),
]


def _should_keep_cell(source: str) -> bool:
    stripped = source.lstrip()
    return stripped.startswith(("class ", "def ", "@dataclass", "decode_sequence"))


def _import_block(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if line.startswith(("from ", "import ")) or line.startswith("sns.set_theme"):
            lines.append(line)
    return "\n".join(lines).strip()


@lru_cache(maxsize=1)
def _load_notebook_namespace() -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    blocks: List[str] = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        stripped = source.lstrip()
        if not stripped:
            continue
        if stripped.startswith(("from ", "import ")):
            block = _import_block(source)
            if block:
                blocks.append(block)
        elif _should_keep_cell(source):
            blocks.append(source)

    module_name = "notebook_backend_runtime"
    runtime_module = types.ModuleType(module_name)
    runtime_module.__file__ = str(NOTEBOOK_PATH)
    sys.modules[module_name] = runtime_module

    namespace = runtime_module.__dict__
    namespace["__name__"] = module_name
    compiled = compile("\n\n".join(blocks), str(NOTEBOOK_PATH), "exec")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compiled, namespace)
    return namespace


def _nb_box(box: Box):
    ns = _load_notebook_namespace()
    return ns["Box"](
        id=box.id,
        length=box.length,
        width=box.width,
        height=box.height,
        weight_kg=box.weight_kg,
        fragile=box.fragile,
    )


def _nb_container(container: Container):
    ns = _load_notebook_namespace()
    return ns["Container"](
        length=container.length,
        width=container.width,
        height=container.height,
    )


def _nb_problem(boxes: List[Box], container: Container):
    ns = _load_notebook_namespace()
    return ns["CLOProblem"](
        container=_nb_container(container),
        seq_boxes=[_nb_box(box) for box in boxes],
    )


def _to_app_placements(result) -> List[dict]:
    placed = []
    for p in result.placed_boxes:
        placed.append({
            "id": p.box.id,
            "pos": (p.x, p.z, p.y),
            "dim": (p.l, p.w, p.h),
            "weight": p.box.weight_kg,
            "fragile": p.box.fragile,
        })
    return placed

def _check_overlap(p1: Tuple[float, float, float], d1: Tuple[float, float, float],
                   p2: Tuple[float, float, float], d2: Tuple[float, float, float]) -> bool:
    x1, y1, z1 = p1
    l1, w1, h1 = d1
    x2, y2, z2 = p2
    l2, w2, h2 = d2
    return not (
        x1 + l1 <= x2 or x2 + l2 <= x1 or
        y1 + w1 <= y2 or y2 + w2 <= y1 or
        z1 + h1 <= z2 or z2 + h2 <= z1
    )


def _is_supported(x: float, y: float, z: float, l: float, w: float, placed: List[dict]) -> bool:
    if z < 1e-6:
        return True
    for pb in placed:
        px, py, pz = pb["pos"]
        pl, pw, ph = pb["dim"]
        if abs((pz + ph) - z) < 0.01:
            ox = min(x + l, px + pl) - max(x, px)
            oy = min(y + w, py + pw) - max(y, py)
            if ox > 0 and oy > 0:
                return True
    return False


def pack_sequence_with_forced(
    sequence: List[Box],
    container: Container,
    forced_orientations: Dict[int, Tuple[float, float, float]],
) -> Tuple[List[dict], float]:
    placed: List[dict] = []
    candidates = [(0.0, 0.0, 0.0)]
    vol_packed = 0.0

    for box in sequence:
        oris = ([forced_orientations[box.id]]
                if box.id in forced_orientations else box.get_orientations())
        best_pos = best_ori = None
        best_score = float("inf")

        for ori in oris:
            bl, bw, bh = ori
            for cx, cy, cz in candidates:
                if (cx + bl > container.length or
                        cy + bw > container.width or
                        cz + bh > container.height):
                    continue
                if not _is_supported(cx, cy, cz, bl, bw, placed):
                    continue
                if any(_check_overlap((cx, cy, cz), (bl, bw, bh), pb["pos"], pb["dim"])
                       for pb in placed):
                    continue
                score = cx + cy + cz + cz * box.weight_kg
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_ori = ori

        if best_pos:
            bx, by, bz = best_pos
            bl, bw, bh = best_ori
            placed.append({
                "id": box.id,
                "pos": best_pos,
                "dim": best_ori,
                "weight": box.weight_kg,
                "fragile": box.fragile,
                "original_box": box,
            })
            vol_packed += bl * bw * bh
            candidates.extend([
                (bx + bl, by, bz),
                (bx, by + bw, bz),
                (bx, by, bz + bh),
            ])

    util_pct = (vol_packed / container.volume) * 100.0 if container.volume > 0 else 0.0
    return placed, util_pct


def draw_box_3d(ax, pos, dim, color, alpha=0.55, highlight=False):
    x, y, z = pos
    l, w, h = dim

    verts = [
        (x, y, z),
        (x + l, y, z),
        (x + l, y + w, z),
        (x, y + w, z),
        (x, y, z + h),
        (x + l, y, z + h),
        (x + l, y + w, z + h),
        (x, y + w, z + h),
    ]
    faces = [
        [verts[i] for i in [0, 1, 2, 3]],
        [verts[i] for i in [4, 5, 6, 7]],
        [verts[i] for i in [0, 1, 5, 4]],
        [verts[i] for i in [2, 3, 7, 6]],
        [verts[i] for i in [1, 2, 6, 5]],
        [verts[i] for i in [0, 3, 7, 4]],
    ]
    edge_color = "gold" if highlight else "black"
    edge_width = 1.6 if highlight else 0.3
    poly = Poly3DCollection(
        faces,
        alpha=alpha,
        linewidths=edge_width,
        edgecolors=edge_color,
        facecolors=color,
    )
    ax.add_collection3d(poly)

class _SpatialGrid:
    GRID = 12

    def __init__(self, cl, cw, ch):
        self.cl, self.cw, self.ch = cl, cw, ch
        self.cell_l = cl / self.GRID
        self.cell_w = cw / self.GRID
        self.cell_h = ch / self.GRID
        self._grid: Dict[Tuple[int, int, int], List[dict]] = {}

    def add(self, pb: dict):
        x, y, z = pb["pos"]
        l, w, h = pb["dim"]

        ix0 = max(0, int(x / self.cell_l))
        ix1 = min(self.GRID - 1, int((x + l - 1e-7) / self.cell_l))
        iy0 = max(0, int(y / self.cell_w))
        iy1 = min(self.GRID - 1, int((y + w - 1e-7) / self.cell_w))
        iz0 = max(0, int(z / self.cell_h))
        iz1 = min(self.GRID - 1, int((z + h - 1e-7) / self.cell_h))

        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    cell = (ix, iy, iz)
                    self._grid.setdefault(cell, []).append(pb)

    def any_overlap(self, x, y, z, l, w, h) -> bool:
        x2, y2, z2 = x + l, y + w, z + h
        ix0 = max(0, int(x / self.cell_l))
        ix1 = min(self.GRID - 1, int((x + l - 1e-7) / self.cell_l))
        iy0 = max(0, int(y / self.cell_w))
        iy1 = min(self.GRID - 1, int((y + w - 1e-7) / self.cell_w))
        iz0 = max(0, int(z / self.cell_h))
        iz1 = min(self.GRID - 1, int((z + h - 1e-7) / self.cell_h))

        seen_ids = set()
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    for pb in self._grid.get((ix, iy, iz), []):
                        pid = pb["id"]
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)
                        px, py, pz = pb["pos"]
                        pl, pw, ph = pb["dim"]
                        if x2 <= px + 1e-5 or px + pl <= x + 1e-5:
                            continue
                        if y2 <= py + 1e-5 or py + pw <= y + 1e-5:
                            continue
                        if z2 <= pz + 1e-5 or pz + ph <= z + 1e-5:
                            continue
                        return True
        return False


def pack_sequence(
    sequence: List[Box],
    container: Container,
) -> Tuple[List[dict], float]:
    placed: List[dict] = []
    packed_volume = 0.0
    cl, cw, ch = container.length, container.width, container.height

    ep_set = {(0.0, 0.0, 0.0)}
    tops: Dict[float, List[dict]] = {}
    grid = _SpatialGrid(cl, cw, ch)

    def _supported(x: float, y: float, z: float, l: float, w: float) -> bool:
        if z < 1e-7:
            return True
        z_round = round(z, 5)
        if z_round not in tops:
            return False
        x_limit = x + l - 1e-5
        y_limit = y + w - 1e-5
        for pb in tops[z_round]:
            px, py = pb["pos"][0], pb["pos"][1]
            pl, pw = pb["dim"][0], pb["dim"][1]
            if px < x_limit and px + pl > x + 1e-5 and py < y_limit and py + pw > y + 1e-5:
                return True
        return False

    def _add_eps(x: float, y: float, z: float, l: float, w: float, h: float) -> None:
        raw_candidates = ((x + l, y, z), (x, y + w, z), (x, y, z + h))
        for cx, cy, cz in raw_candidates:
            if cx > cl - 1e-5 or cy > cw - 1e-5 or cz > ch - 1e-5:
                continue
            ep_set.add((round(cx, 5), round(cy, 5), round(cz, 5)))
            gz = 0.0
            for z_top, pbs in tops.items():
                if z_top > cz + 1e-5:
                    continue
                for pb in pbs:
                    px, py = pb["pos"][0], pb["pos"][1]
                    pl, pw = pb["dim"][0], pb["dim"][1]
                    if px < cx + 1e-5 and px + pl > cx + 1e-5 and py < cy + 1e-5 and py + pw > cy + 1e-5:
                        if z_top > gz:
                            gz = z_top
            if gz < cz - 1e-5:
                ep_set.add((round(cx, 5), round(cy, 5), round(gz, 5)))

            px_best = 0.0
            for pb in placed:
                px, py, pz = pb["pos"]
                pl, pw, ph = pb["dim"]
                if px + pl <= cx + 1e-5 and py < cy + 1e-5 and py + pw > cy + 1e-5 and pz < cz + 1e-5 and pz + ph > cz + 1e-5:
                    if px + pl > px_best:
                        px_best = px + pl
            if px_best > 1e-5:
                ep_set.add((round(px_best, 5), round(cy, 5), round(cz, 5)))

            py_best = 0.0
            for pb in placed:
                px, py, pz = pb["pos"]
                pl, pw, ph = pb["dim"]
                if py + pw <= cy + 1e-5 and px < cx + 1e-5 and px + pl > cx + 1e-5 and pz < cz + 1e-5 and pz + ph > cz + 1e-5:
                    if py + pw > py_best:
                        py_best = py + pw
            if py_best > 1e-5:
                ep_set.add((round(cx, 5), round(py_best, 5), round(cz, 5)))

    for box in sequence:
        best_pos = None
        best_ori = None
        best_score = float("inf")
        orientations = box.get_orientations()

        for (cx, cy, cz) in ep_set:
            for (bl, bw, bh) in orientations:
                if cx + bl > cl + 1e-5 or cy + bw > cw + 1e-5 or cz + bh > ch + 1e-5:
                    continue
                if not _supported(cx, cy, cz, bl, bw):
                    continue
                if grid.any_overlap(cx, cy, cz, bl, bw, bh):
                    continue

                score = (cz * 3.0) + (cy * 1.5) + (cx * 1.0) + ((cx + bl) / cl * 0.1)
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_ori = (bl, bw, bh)

        if best_pos:
            x, y, z = best_pos
            l, w, h = best_ori
            pb = {
                "id":      box.id,
                "pos":     (x, y, z),
                "dim":     (l, w, h),
                "weight":  box.weight_kg,
                "fragile": box.fragile,
            }
            placed.append(pb)
            packed_volume += l * w * h
            tops.setdefault(round(z + h, 5), []).append(pb)
            grid.add(pb)
            ep_set.discard(best_pos)
            _add_eps(x, y, z, l, w, h)

    util = (packed_volume / container.volume) * 100 if container.volume > 0 else 0
    return placed, util


def greedy_pack(
    boxes: List[Box],
    container: Container,
) -> Tuple[List[dict], float, float]:
    start = time.time()
    sorted_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
    placed, util = pack_sequence(sorted_boxes, container)
    return placed, util, (time.time() - start) * 1000


def genetic_algorithm(
    boxes: List[Box],
    container: Container,
    *,
    pop_size: int = 30,
    generations: int = 50,
    mutation_rate: float = 0.10,
    elite_frac: float = 0.25,
    progress_cb=None,
) -> Tuple[List[dict], float, float, List[float]]:
    # Use an internal adaptive GA implementation (imported from app_engine1)
    # This GA operates on packing sequences and uses the module's `pack_sequence`.
    start_time = time.time()
    time_budget_s = 25.0

    if len(boxes) < 2:
        placed, util = pack_sequence(boxes, container)
        return placed, util, 0.0, [util]

    _cache: Dict[Tuple[int, ...], float] = {}

    def evaluate(seq):
        key = tuple(b.id for b in seq)
        if key not in _cache:
            _, util = pack_sequence(seq, container)
            _cache[key] = util
        return _cache[key]

    def crossover_ox1(pa, pb):
        n = len(pa)
        i, j = sorted(random.sample(range(n), 2))
        seg = pa[i:j+1]
        seg_ids = {b.id for b in seg}
        rest = [b for b in pb if b.id not in seg_ids]
        child = [None] * n
        child[i:j+1] = seg
        it = iter(rest)
        for k in list(range(j+1, n)) + list(range(0, i)):
            child[k] = next(it)
        return child

    def mutate_adaptive(seq, placed_ids):
        unplaced = [b for b in seq if b.id not in placed_ids]
        placed_objs = [b for b in seq if b.id in placed_ids]

        if not unplaced:
            res = seq[:]
            i, j = random.sample(range(len(res)), 2)
            res[i], res[j] = res[j], res[i]
            return res

        if random.random() < 0.7:
            random.shuffle(unplaced)
            return unplaced + placed_objs
        else:
            res = seq[:]
            i, j = sorted(random.sample(range(len(res)), 2))
            res[i:j+1] = reversed(res[i:j+1])
            return res

    seeds = [
        sorted(boxes, key=lambda b: b.volume, reverse=True),
        sorted(boxes, key=lambda b: b.length * b.width, reverse=True),
        sorted(boxes, key=lambda b: max(b.length, b.width, b.height), reverse=True),
    ]

    population = []
    for s in seeds:
        if len(population) < pop_size:
            population.append(s[:])

    while len(population) < pop_size:
        shuffled = seeds[0][:]
        random.shuffle(shuffled)
        population.append(shuffled)

    fitnesses = [evaluate(seq) for seq in population]
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_seq = population[best_idx][:]
    best_util = fitnesses[best_idx]

    history = [best_util]
    no_improve = 0

    deadline = start_time + time_budget_s

    for gen in range(generations):
        if time.time() > deadline:
            break

        sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        elite_size = max(1, int(pop_size * elite_frac))
        next_gen = [population[i][:] for i in sorted_idx[:elite_size]]

        placed_res, _ = pack_sequence(best_seq, container)
        placed_ids = {p["id"] for p in placed_res}

        while len(next_gen) < pop_size:
            idx1 = random.sample(range(len(population)), min(3, len(population)))
            p1 = population[max(idx1, key=lambda i: fitnesses[i])][:]
            idx2 = random.sample(range(len(population)), min(3, len(population)))
            p2 = population[max(idx2, key=lambda i: fitnesses[i])][:]

            child = crossover_ox1(p1, p2) if random.random() < 0.8 else p1[:]
            if random.random() < mutation_rate:
                child = mutate_adaptive(child, placed_ids)
            next_gen.append(child)

        population = next_gen
        fitnesses = [evaluate(seq) for seq in population]

        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_util:
            best_util = fitnesses[gen_best_idx]
            best_seq = population[gen_best_idx][:]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 6:
            for sc in range(elite_size, len(population)):
                population[sc] = mutate_adaptive(best_seq, placed_ids)
            fitnesses = [evaluate(seq) for seq in population]
            no_improve = 0

        history.append(best_util)
        if progress_cb:
            progress_cb(gen + 1, generations, best_util)

    placed, final_util = pack_sequence(best_seq, container)
    return placed, final_util, (time.time() - start_time) * 1000, history


def simulated_annealing(
    boxes: List[Box],
    container: Container,
    *,
    initial_sequence: Optional[List[Box]] = None,
    T_start: float = 1000.0,
    T_end: float = 0.1,
    cooling: float = 0.995,
    iters_per_step: int = 30,
    progress_cb=None,
) -> Tuple[List[dict], float, float]:
    """
    Standard simulated annealing wrapper.
    Uses the already-implemented interactive SA engine internally.
    """

    placed_boxes, util, exec_time = simulated_annealing_interactive(
        boxes=boxes,
        container=container,
        initial_sequence=initial_sequence,
        T_start=T_start,
        T_end=T_end,
        cooling=cooling,
        iters_per_step=iters_per_step,
        progress_cb=progress_cb,
    )

    # convert placed objects into UI dictionaries
    converted = []

    for pb in placed_boxes:

        converted.append({
            "id": pb.box.id,
            "pos": (pb.x, pb.z, pb.y),
            "dim": (pb.l, pb.w, pb.h),
            "weight": getattr(pb.box, "weight", pb.box.weight_kg),
            "fragile": pb.box.fragile,
        })

    return converted, util, exec_time


def _to_optimizer_box(box: Box):
    """
    Convert app Box -> notebook/runtime Box.
    Avoids importing nonexistent optimizer.py
    """

    ns = _load_notebook_namespace()

    return ns["Box"](
        id=box.id,
        length=box.length,
        width=box.width,
        height=box.height,
        weight_kg=box.weight_kg,
        fragile=box.fragile,
    )


def _to_optimizer_container(container: Container):
    """
    Convert app Container -> notebook/runtime Container.
    Avoids importing nonexistent optimizer.py
    """

    ns = _load_notebook_namespace()

    return ns["Container"](
        length=container.length,
        width=container.width,
        height=container.height,
    )

def smart_greedy_pack(boxes, container, progress_cb=None):
    """
    Multi-strategy greedy that tries multiple sorting strategies and returns the best.
    Uses greedy_pack which is already working.
    """
    from app_engine import greedy_pack
    
    strategies = [
        ("Volume (largest first)", lambda b: b.volume),
        ("Volume (smallest first)", lambda b: -b.volume),
        ("Max dimension", lambda b: max(b.length, b.width, b.height)),
        ("Min dimension (small boxes last)", lambda b: -min(b.length, b.width, b.height)),
        ("Area (footprint)", lambda b: b.length * b.width),
        ("Perimeter", lambda b: b.length + b.width + b.height),
        ("Surface area", lambda b: 2*(b.length*b.width + b.length*b.height + b.width*b.height)),
        ("Volume × Density (heavy + large)", lambda b: b.volume * b.weight_kg),
        ("Density (heavy first)", lambda b: b.weight_kg / b.volume if b.volume > 0 else 0),
        ("Fragile first", lambda b: (0 if b.fragile else 1, -b.volume)),
        ("Original order", None),
    ]
    
    best_placed = None
    best_util = 0
    best_strategy = None
    
    total = len(strategies)
    for idx, (strategy_name, key_func) in enumerate(strategies):
        if progress_cb:
            progress_cb(idx, total, strategy_name)
        
        # Sort boxes according to strategy
        if key_func is None:
            sorted_boxes = boxes[:]
        else:
            sorted_boxes = sorted(boxes, key=key_func, reverse=True)
        
        # Use greedy_pack - this already works with your notebook
        try:
            placed, util, _ = greedy_pack(sorted_boxes, container)
        except:
            # Fallback to pack_sequence if greedy_pack fails
            from app_engine import pack_sequence
            placed, util = pack_sequence(sorted_boxes, container)
        
        if util > best_util:
            best_util = util
            best_placed = placed
            best_strategy = strategy_name
    
    if progress_cb:
        progress_cb(total, total, f"Best: {best_strategy}")
    
    return best_placed, best_util, best_strategy


"""def simulated_annealing_interactive(
    boxes,
    container,
    initial_sequence=None,
    T_start=500.0,
    T_end=5.0,
    cooling=0.97,
    iters_per_step=6,
    target_pct=80.0,
    progress_cb=None,
    user_ask_cb=None
):

    ns = _load_notebook_namespace()

    SpaceManager = ns["SpaceManager"]

    if initial_sequence is None:
        initial_sequence = sorted(
            boxes,
            key=lambda b: b.volume,
            reverse=True
        )

    current_seq = initial_sequence[:]
    random.shuffle(current_seq)

    sm = SpaceManager(_nb_container(container))

    for box in current_seq:

        nb_box = _nb_box(box)

        space, dims = sm.find_placement(
            nb_box,
            strategy="bottom"
        )

        if space and dims:
            sm.place_box(nb_box, space, dims)

    current_score = sm.packed_volume

    best_seq = current_seq[:]
    best_score = current_score

    theoretical_max = sum(b.volume for b in boxes)

    target_volume = theoretical_max * (target_pct / 100.0)

    target_reached = False
    should_continue = True

    T = T_start
    step = 0

    total_iterations = 0
    no_improvement_count = 0

    start_time = time.time()

    while T > T_end:

        for _ in range(iters_per_step):

            total_iterations += 1

            if not target_reached and best_score >= target_volume:

                target_reached = True

                current_util = best_score 

                pct_of_max = (
                    (best_score / 100) * (container_vol / theoretical_max) * 100
                )

                if user_ask_cb:

                    should_continue = user_ask_cb(
                        current_util,
                        pct_of_max,
                        total_iterations
                    )

                    if not should_continue:
                        break

                else:
                    print(
                        f"Target reached at iteration "
                        f"{total_iterations}: "
                        f"{current_util:.1f}%"
                    )

            new_seq = current_seq[:]

            i, j = random.sample(
                range(len(new_seq)),
                2
            )

            new_seq[i], new_seq[j] = (
                new_seq[j],
                new_seq[i]
            )

            sm2 = SpaceManager(_nb_container(container))

            for box in new_seq:

                nb_box = _nb_box(box)

                space, dims = sm2.find_placement(
                    nb_box,
                    strategy="bottom"
                )

                if space and dims:
                    sm2.place_box(nb_box, space, dims)

            new_score = sm2.packed_volume

            delta = new_score - current_score

            if (
                delta > 0 or
                (
                    T > 1e-10 and
                    random.random() < math.exp(delta / T)
                )
            ):

                current_seq = new_seq
                current_score = new_score

                no_improvement_count = 0

            else:
                no_improvement_count += 1

            if current_score > best_score:

                best_score = current_score
                best_seq = current_seq[:]

                no_improvement_count = 0

            if (
                progress_cb and
                total_iterations % 10 == 0
            ):

                progress_cb(
                    T,best_score , total_iterations
                )

            if no_improvement_count > 200:
                break

        if target_reached and not should_continue:
            break

        T *= cooling
        step += 1

    sm_best = SpaceManager(_nb_container(container))

    for box in best_seq:

        nb_box = _nb_box(box)

        space, dims = sm_best.find_placement(
            nb_box,
            strategy="bottom"
        )

        if space and dims:
            sm_best.place_box(
                nb_box,
                space,
                dims
            )

    execution_time = time.time() - start_time

    return placed, best_score, execution_time"""

def simulated_annealing_interactive(
    boxes,
    container,
    initial_sequence=None,
    T_start=500.0,
    T_end=5.0,
    cooling=0.97,
    iters_per_step=6,
    target_pct=80.0,
    progress_cb=None,
    user_ask_cb=None
):
    """
    Simulated Annealing - matches notebook behavior.
    """
    from app_engine import pack_sequence
    
    container_vol = container.length * container.width * container.height
    theoretical_max = sum(b.volume for b in boxes)
    theoretical_max_util = theoretical_max / container_vol * 100
    
    target_volume = theoretical_max * (target_pct / 100.0)
    target_util = target_volume / container_vol * 100
    
    # Start with initial sequence
    if initial_sequence is None:
        current_seq = sorted(boxes, key=lambda b: b.volume, reverse=True)
    else:
        current_seq = initial_sequence[:]
    
    random.shuffle(current_seq)
    
    # Initial evaluation
    #placed, current_score = pack_sequence(current_seq, container)
    placed, current_util_pct = pack_sequence(current_seq, container)
    current_score = current_util_pct 
    best_seq = current_seq[:]
    best_score = current_score
    
    print(f"\n{'='*60}")
    print(f"  Simulated Annealing Started")
    print(f"  Theoretical max: {theoretical_max_util:.1f}% of container")
    print(f"  Target: {target_pct:.0f}% of theoretical max = {target_util:.1f}% of container")
    print(f"  Initial: {best_score/container_vol*100:.1f}% of container")
    print(f"{'='*60}\n")
    
    no_improvement_count = 0
    T = T_start
    total_iterations = 0
    start_time = time.time()
    target_reached = False
    should_continue = True
    
    # Calculate max iterations
    max_iterations = int(math.log(T_end / T_start) / math.log(cooling)) * iters_per_step
    max_iterations = min(max_iterations, 5000)
    
    for iteration in range(max_iterations):
        if not should_continue:
            break
            
        total_iterations += 1
        current_container_util = current_score / container_vol * 100
        current_pct_of_max = current_score / theoretical_max * 100
        best_container_util = best_score / container_vol * 100
        
        # Check if target reached (using BEST score, not current)
        if not target_reached and best_score >= target_pct:
            target_reached = True
            best_pct_of_max = best_score / theoretical_max * 100 
            print(f"\nTarget reached at iteration {iteration}!")
            print(f"    {best_container_util:.1f}% of container ({best_pct_of_max:.1f}% of theoretical max)")
            
            if user_ask_cb:
                should_continue = user_ask_cb(best_container_util, best_pct_of_max, iteration)
                if not should_continue:
                    print("   Stopping as requested.")
                    break
        
        # Stop if no improvement for many iterations
        if no_improvement_count > 200:
            print(f"\n  No improvement for {no_improvement_count} iterations - stopping")
            break
        
        # Create neighbor by swapping (use current_seq, not best_seq)
        new_seq = current_seq[:]
        i, j = random.sample(range(len(new_seq)), 2)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        # Evaluate new sequence
        placed, new_score = pack_sequence(new_seq, container)
        
        delta = new_score - current_score  # Compare with current_score
        
        # Acceptance criterion
        if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
            current_seq = new_seq
            current_score = new_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Update best if improved
        if current_score > best_score:
            best_score = current_score
            best_seq = current_seq[:]
            no_improvement_count = 0
            best_pct_of_max = best_score / theoretical_max * 100
            
            # If target reached and improved, ask again
            if target_reached and user_ask_cb:
                print(f"\n📈 Improved to {best_pct_of_max:.1f}% of theoretical max!")
                should_continue = user_ask_cb(best_container_util, best_pct_of_max, iteration)
                if not should_continue:
                    print("   Stopping as requested.")
                    break
        
        # Progress output every 100 iterations
        if iteration % 100 == 0 and iteration > 0:
            current_util = current_score 
            best_util = best_score 
            print(f"  Iter {iteration}: T={T:.2f}, current={current_util:.1f}%, best={best_util:.1f}%")
            
            # Update GUI via callback
            if progress_cb and iteration % 100 == 0 and iteration > 0:
                progress_cb(T, best_score, iteration, placed)
        
        T *= cooling
    
    # Final packing with best sequence
    placed, _ = pack_sequence(best_seq, container)
    final_util_pct = best_score 
    execution_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  Simulated Annealing Finished")
    print(f"  Final utilization: {final_util_pct:.2f}%")
    print(f"  Boxes placed: {len(placed)}/{len(boxes)}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Time: {execution_time:.1f}s")
    print(f"{'='*60}\n")
    
    return placed, final_util_pct, execution_time

def validate_result(
    placed: List[dict],
    container: Container,
    verbose: bool = True,) -> dict:
    overlaps = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if _check_overlap(placed[i]["pos"], placed[i]["dim"],
                            placed[j]["pos"], placed[j]["dim"]):
                overlaps.append((placed[i]["id"], placed[j]["id"]))

    no_overlap = len(overlaps) == 0

    oob = []
    for pb in placed:
        x, y, z = pb["pos"]
        l, w, h = pb["dim"]
        if (x < 0 or y < 0 or z < 0 or
                x + l > container.length or
                y + w > container.width or
                z + h > container.height):
            oob.append(pb["id"])

    in_bounds = len(oob) == 0

    floating = []
    for pb in placed:
        x, y, z = pb["pos"]
        l, w, _ = pb["dim"]
        if z == 0:
            continue
        if not _is_supported(x, y, z, l, w, placed):
            floating.append(pb["id"])

    no_floating = len(floating) == 0

    if verbose:
        print("=" * 60)
        print("  VALIDATION REPORT")
        print("=" * 60)
        print(f"  No overlaps    : {'PASS' if no_overlap else f'FAIL ({len(overlaps)} pairs)'}")
        if not no_overlap:
            for a, b in overlaps[:5]:
                print(f"                   -> Box {a} <-> Box {b}")
            if len(overlaps) > 5:
                print(f"                   -> ... and {len(overlaps) - 5} more")

        print(f"  In bounds      : {'PASS' if in_bounds else f'FAIL ({len(oob)} boxes)'}")
        if not in_bounds:
            print(f"                   -> IDs: {oob[:10]}")

        print(f"  No floating    : {'PASS' if no_floating else f'FAIL ({len(floating)} boxes)'}")
        if not no_floating:
            print(f"                   -> IDs: {floating[:10]}")
        print("=" * 60)

    return {
        "no_overlap": no_overlap,
        "in_bounds": in_bounds,
        "no_floating": no_floating,
        "valid": no_overlap and in_bounds and no_floating,
        "overlaps": overlaps,
        "out_of_bounds": oob,
        "floating": floating,
    }

def trim_unfittable_boxes(boxes: List[Box], container: Container) -> List[Box]:
    """
    Remove boxes that cannot theoretically fit in the container in any orientation.
    """
    trimmed_boxes = []
    removed_ids = []
    
    for box in boxes:
        # Check if box can fit in any orientation
        fits = False
        for l, w, h in box.get_orientations():
            if l <= container.length and w <= container.width and h <= container.height:
                fits = True
                break
        
        if fits:
            trimmed_boxes.append(box)
        else:
            removed_ids.append(box.id)
    
    if removed_ids:
        print(f"⚠️ Removed {len(removed_ids)} box(es) that cannot fit: {removed_ids[:10]}...")
    
    return trimmed_boxes

all = [
    "Box",
    "Container",
    "PRESET_CONTAINERS",
    "pack_sequence",
    "pack_sequence_with_forced",
    "greedy_pack",
    "genetic_algorithm",
    "simulated_annealing",
    "draw_box_3d",
    "validate_result",
]

def trim_boxes_to_capacity(boxes: List[Box], container: Container):
    total_vol = 0.0
    capacity = container.length * container.width * container.height
    result = []
    for b in boxes:
        box_vol = b.length * b.width * b.height
        if total_vol + box_vol <= capacity:
            result.append(b)
            total_vol += box_vol
        else:
            break
    utilization = (total_vol / capacity * 100) if capacity > 0 else 0.0
    return result, utilization   # ← must return BOTH, as a tuple
