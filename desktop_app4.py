"""
3D Container Loading Optimizer — Desktop App
================================================
Features
  • Full zoom + rotate with toolbar & mouse scroll
  • Click placed box → flip orientation or reposition (non-fragile only)
  • Edit & validate predefined container sizes inline
  • White & Green premium UI
  • Algorithms linked to notebooks/NoteBook.ipynb via notebook_backend.py
"""

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

# ── Notebook-backed algorithms exposed through a stable app API ──────────────
from app_engine import (
    Box,
    Container,
    PRESET_CONTAINERS,
    pack_sequence,
    pack_sequence_with_forced,
    greedy_pack,
    genetic_algorithm,
    simulated_annealing,
    smart_greedy_pack,
    simulated_annealing_interactive,
    draw_box_3d,
)

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (White & Light Purple)
# ─────────────────────────────────────────────

C = {
    # Base backgrounds kept light/white
    "bg_main": "#ffffff",
    "bg_panel": "#f8fdf9",
    "bg_card": "#ffffff",
    "bg_light": "#f1faf3",

    # Text
    "fg_lavender": "#56427e",
    "fg_muted": "#6b7280",
    "fg_dark": "#392650",
    "fg_white": "#ffffff",

    # Main light-purple palette
    "primary": "#ab8cf5",
    "primary_dark": "#8f70db",
    "primary_soft": "#d3c2fb",

    # Compatibility keys used across the UI
    "pink_hot": "#ab8cf5",
    "pink_mid": "#bda0fa",
    "pink_soft": "#7f5eb8",
    "purple_dark": "#ddcffb",
    "purple_btn": "#9b7fe6",
    "purple_hi": "#6c4aa6",
    "green": "#7356ad",
    "amber": "#aa8fd3",
    "red": "#b85c5c",
}


# ─────────────────────────────────────────────
#  EDIT CONTAINER DIALOG
# ─────────────────────────────────────────────

class EditContainerDialog(tk.Toplevel):
    """Modal popup to edit a preset container's dimensions."""

    def __init__(self, parent, container: Container, on_save):
        super().__init__(parent)
        self.title(f"Edit — {container.name}")
        self.geometry("400x320")
        self.resizable(False, False)
        self.configure(bg=C["bg_panel"])
        self.grab_set()

        self.container = container
        self.on_save = on_save

        tk.Label(
            self,
            text=f"✏  Edit Container: {container.name}",
            font=("Helvetica", 12, "bold"),
            bg=C["pink_hot"], fg=C["fg_white"], pady=10,
        ).pack(fill="x")

        form = tk.Frame(self, bg=C["bg_panel"], padx=24, pady=12)
        form.pack(fill="both", expand=True)

        fields = [
            ("Name",   container.name,   "Display name"),
            ("Length", container.length, "Internal length (cm)"),
            ("Width",  container.width,  "Internal width  (cm)"),
            ("Height", container.height, "Internal height (cm)"),
        ]
        self._entries: dict = {}
        for i, (lbl, default, hint) in enumerate(fields):
            tk.Label(
                form, text=lbl + ":", bg=C["bg_panel"],
                fg=C["fg_lavender"], font=("Helvetica", 9, "bold"),
                width=9, anchor="w",
            ).grid(row=i, column=0, pady=6, sticky="w")
            e = tk.Entry(
                form, font=("Helvetica", 10), width=18,
                bg=C["bg_light"], fg=C["fg_dark"],
                insertbackground=C["purple_dark"], relief="flat",
            )
            e.insert(0, str(default))
            e.grid(row=i, column=1, padx=8, pady=6)
            tk.Label(
                form, text=hint, bg=C["bg_panel"],
                fg=C["purple_hi"], font=("Helvetica", 7),
            ).grid(row=i, column=2, sticky="w", padx=4)
            self._entries[lbl] = e

        self._msg = tk.Label(
            self, text="", fg=C["red"], bg=C["bg_panel"], font=("Helvetica", 9)
        )
        self._msg.pack()

        btn_row = tk.Frame(self, bg=C["bg_panel"], pady=10)
        btn_row.pack()
        for txt, cmd, bg in [
            (" Save", self._save, C["pink_hot"]),
            (" Cancel", self.destroy, C["purple_btn"]),
        ]:
            tk.Button(
                btn_row, text=txt, font=("Helvetica", 10, "bold"),
                bg=bg, fg=C["fg_white"], relief="flat",
                padx=14, pady=6, cursor="hand2", command=cmd,
            ).pack(side="left", padx=8)

    def _save(self):
        name = self._entries["Name"].get().strip()
        errors = []
        if not name:
            errors.append("Name cannot be empty.")
        nums = {}
        for d in ("Length", "Width", "Height"):
            try:
                v = float(self._entries[d].get())
                if v <= 0:
                    errors.append(f"{d} must be > 0.")
                nums[d] = v
            except ValueError:
                errors.append(f"{d} must be a number.")
                nums[d] = 0.0
        if errors:
            self._msg.config(text="\n".join(errors))
            return
        self.container.name   = name
        self.container.length = nums["Length"]
        self.container.width  = nums["Width"]
        self.container.height = nums["Height"]
        self.on_save()
        self.destroy()


# ─────────────────────────────────────────────
#  FLIP / REPOSITION BOX DIALOG
# ─────────────────────────────────────────────

class FlipBoxDialog(tk.Toplevel):
    """Popup to change orientation or position of a non-fragile placed box."""

    def __init__(self, parent, placed_box: dict, box_obj: Box,
                 container: Container, on_apply):
        super().__init__(parent)
        self.title(f"Edit Box #{placed_box['id']}")
        self.geometry("400x380")
        self.resizable(False, False)
        self.configure(bg=C["bg_panel"])
        self.grab_set()

        self.pb = placed_box
        self.box_obj = box_obj
        self.container = container
        self.on_apply = on_apply

        tk.Label(
            self,
            text=f"🔄  Box #{placed_box['id']}  —  Non-Fragile",
            font=("Helvetica", 11, "bold"),
            bg=C["pink_mid"], fg=C["fg_white"], pady=8,
        ).pack(fill="x")

        # Info strip
        info = tk.Frame(self, bg=C["bg_card"], pady=6, padx=12)
        info.pack(fill="x")
        cur_l, cur_w, cur_h = placed_box["dim"]
        cur_x, cur_y, cur_z = placed_box["pos"]
        for text in [
            f"Current orientation : {cur_l}×{cur_w}×{cur_h} cm",
            f"Current position    : x={cur_x}, y={cur_y}, z={cur_z}",
            f"Weight              : {placed_box['weight']} kg",
        ]:
            tk.Label(info, text=text, bg=C["bg_card"],
                     fg=C["fg_lavender"], font=("Helvetica", 9)).pack(anchor="w")

        # Orientation chooser
        ori_sec = tk.LabelFrame(
            self, text="  Choose orientation  ",
            bg=C["bg_panel"], fg=C["pink_soft"],
            font=("Helvetica", 9, "bold"), padx=12, pady=6,
        )
        ori_sec.pack(fill="x", padx=12, pady=(10, 0))

        self._ori_var = tk.StringVar()
        oris = box_obj.get_orientations()
        labels = box_obj.orientation_labels()
        cur_str = f"{cur_l}×{cur_w}×{cur_h} cm"
        for lbl, ori in zip(labels, oris):
            marker = "  ← current" if lbl == cur_str else ""
            tk.Radiobutton(
                ori_sec, text=lbl + marker,
                variable=self._ori_var, value=lbl,
                bg=C["bg_panel"], fg=C["fg_lavender"],
                selectcolor=C["bg_card"], font=("Helvetica", 9),
            ).pack(anchor="w")
        self._ori_var.set(cur_str)
        self._oris = dict(zip(labels, oris))

        # Position override
        pos_sec = tk.LabelFrame(
            self, text="  Override position (cm, optional)  ",
            bg=C["bg_panel"], fg=C["pink_soft"],
            font=("Helvetica", 9, "bold"), padx=12, pady=6,
        )
        pos_sec.pack(fill="x", padx=12, pady=(8, 0))
        row = tk.Frame(pos_sec, bg=C["bg_panel"])
        row.pack(fill="x")
        self._pos_entries: dict = {}
        for axis, val in [("X", cur_x), ("Y", cur_y), ("Z", cur_z)]:
            tk.Label(
                row, text=axis + ":", bg=C["bg_panel"],
                fg=C["fg_lavender"], font=("Helvetica", 9),
            ).pack(side="left", padx=(4, 0))
            e = tk.Entry(row, width=7, font=("Helvetica", 9),
                         bg=C["bg_light"], fg=C["fg_dark"], relief="flat")
            e.insert(0, str(val))
            e.pack(side="left", padx=(2, 6))
            self._pos_entries[axis] = e

        self._msg = tk.Label(
            self, text="", fg=C["red"], bg=C["bg_panel"], font=("Helvetica", 9)
        )
        self._msg.pack()

        btn_row = tk.Frame(self, bg=C["bg_panel"], pady=8)
        btn_row.pack()
        for txt, cmd, bg in [
            (" Apply", self._apply, C["pink_hot"]),
            (" Cancel", self.destroy, C["purple_btn"]),
        ]:
            tk.Button(
                btn_row, text=txt, font=("Helvetica", 10, "bold"),
                bg=bg, fg=C["fg_white"], relief="flat",
                padx=14, pady=6, cursor="hand2", command=cmd,
            ).pack(side="left", padx=8)

    def _apply(self):
        chosen_label = self._ori_var.get()
        new_ori = self._oris.get(chosen_label)
        if not new_ori:
            self._msg.config(text="Please select an orientation.")
            return
        try:
            nx = float(self._pos_entries["X"].get())
            ny = float(self._pos_entries["Y"].get())
            nz = float(self._pos_entries["Z"].get())
        except ValueError:
            self._msg.config(text="Position values must be numbers.")
            return
        bl, bw, bh = new_ori
        c = self.container
        if (nx < 0 or ny < 0 or nz < 0 or
                nx + bl > c.length or ny + bw > c.width or nz + bh > c.height):
            self._msg.config(
                text=f"Box doesn't fit at that position in the container.\n"
                     f"Container: {c.length}×{c.width}×{c.height} cm"
            )
            return
        self.on_apply(new_ori, (nx, ny, nz))
        self.destroy()


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D Container Loading Optimizer  v3")
        self.geometry("1300x880")
        self.configure(bg=C["bg_main"])
        self.resizable(True, True)

        self.boxes: list = []
        self.container: Container = PRESET_CONTAINERS[0]
        self.last_result  = None
        self.last_util    = 0.0
        self.last_algo    = ""
        self.selected_box_id = None
        self._forced_orientations: dict = {}

        self._apply_ttk_style()
        self._build_ui()

    # ── TTK STYLE ─────────────────────────────────────────────────────────────

    def _apply_ttk_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",          background=C["bg_panel"], borderwidth=0)
        style.configure("TNotebook.Tab",       background=C["bg_card"],
                        foreground=C["fg_lavender"], padding=[12, 5],
                        font=("Helvetica", 9, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", C["pink_hot"])],
                  foreground=[("selected", C["fg_white"])])
        style.configure("TProgressbar", troughcolor=C["bg_card"],
                        background=C["pink_hot"], borderwidth=0)
        style.configure("Treeview",       background=C["bg_card"],
                        foreground=C["fg_lavender"],
                        fieldbackground=C["bg_card"], rowheight=22)
        style.configure("Treeview.Heading", background=C["purple_dark"],
                        foreground=C["fg_white"], font=("Helvetica", 8, "bold"))
        style.map("Treeview", background=[("selected", C["pink_mid"])])

    # ── UI CONSTRUCTION ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["purple_dark"], pady=12)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="3D Container Loading Optimizer",
            font=("Helvetica", 18, "bold"), fg=C["pink_hot"], bg=C["purple_dark"],
        ).pack()
        tk.Label(
            hdr,
            text="Greedy, Genetic,  Simulated Annealing  with an  Interactive 3D visualization",
            font=("Helvetica", 9), fg=C["purple_hi"], bg=C["purple_dark"],
        ).pack()

        # ── Main area ────────────────────────────────────────────
        main = tk.Frame(self, bg=C["bg_main"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(main, bg=C["bg_panel"], width=440)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(main, bg=C["bg_card"],
                         highlightbackground=C["pink_hot"], highlightthickness=1)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        tabs = {}
        for name in ("Container", "Boxes", "Run", "Edit Boxes"):
            f = tk.Frame(nb, bg=C["bg_panel"])
            nb.add(f, text=f"  {name}  ")
            tabs[name] = f
        self._build_container_tab(tabs["Container"])
        self._build_boxes_tab(tabs["Boxes"])
        self._build_run_tab(tabs["Run"])
        self._build_edit_tab(tabs["Edit Boxes"])

    def _section(self, parent, title):
        f = tk.LabelFrame(
            parent, text=f"  {title}  ",
            font=("Helvetica", 9, "bold"),
            bg=C["bg_panel"], fg=C["pink_soft"],
            padx=8, pady=6, relief="groove", bd=1,
        )
        f.pack(fill="x", padx=8, pady=(8, 0))
        return f

    def _btn(self, parent, text, command, bg=None, **kw):
        bg = bg or C["pink_hot"]
        return tk.Button(
            parent, text=text, command=command,
            font=("Helvetica", 9, "bold"),
            bg=bg, fg=C["fg_white"], relief="flat",
            padx=10, pady=5, cursor="hand2", **kw,
        )

    # ── TAB 1: Container ──────────────────────────────────────────────────────

    def _build_container_tab(self, parent):
        sec = self._section(parent, "Preset containers  (click to edit)")
        self._container_var = tk.StringVar(value=PRESET_CONTAINERS[0].name)
        self._preset_rows = []

        for c in PRESET_CONTAINERS:
            row = tk.Frame(sec, bg=C["bg_panel"]); row.pack(fill="x", pady=1)
            rb_label = tk.StringVar(value=str(c))
            rb = tk.Radiobutton(
                row, textvariable=rb_label,
                variable=self._container_var, value=c.name,
                bg=C["bg_panel"], fg=C["fg_lavender"],
                selectcolor=C["bg_card"], font=("Helvetica", 9),
                command=self._on_preset_select,
            )
            rb.pack(side="left")
            tk.Button(
                row, text="✏", font=("Helvetica", 8),
                bg=C["purple_btn"], fg=C["fg_white"], relief="flat",
                cursor="hand2", padx=4,
                command=lambda cont=c, lv=rb_label: self._edit_preset(cont, lv),
            ).pack(side="right", padx=2)
            self._preset_rows.append((c, rb_label))

        sec2 = self._section(parent, "Custom container (cm)")
        grid = tk.Frame(sec2, bg=C["bg_panel"]); grid.pack(fill="x")
        self._custom_entries: dict = {}
        for i, (lbl, default) in enumerate([
            ("Name", "My Container"), ("Length", "520"),
            ("Width", "210"), ("Height", "210"),
        ]):
            tk.Label(grid, text=lbl + ":", bg=C["bg_panel"],
                     fg=C["fg_lavender"], font=("Helvetica", 9),
                     width=8, anchor="w").grid(row=i, column=0, pady=2, sticky="w")
            e = tk.Entry(grid, font=("Helvetica", 9), width=16,
                         bg=C["bg_light"], fg=C["fg_dark"], relief="flat")
            e.insert(0, default)
            e.grid(row=i, column=1, padx=6, pady=2)
            self._custom_entries[lbl] = e
        self._btn(sec2, "Use Custom Container",
                  self._apply_custom_container, bg=C["purple_btn"]).pack(pady=6)

        self._container_info = tk.Label(
            parent, text="", font=("Helvetica", 9),
            bg=C["bg_card"], fg=C["purple_hi"],
            relief="flat", pady=6, wraplength=400,
        )
        self._container_info.pack(fill="x", padx=8, pady=8)
        self._update_container_info()

    def _edit_preset(self, container, label_var):
        def on_save():
            label_var.set(str(container))
            if self.container is container:
                self._update_container_info()
        EditContainerDialog(self, container, on_save)

    def _on_preset_select(self):
        name = self._container_var.get()
        for c in PRESET_CONTAINERS:
            if c.name == name:
                self.container = c; break
        self._update_container_info()
        from app_engine import trim_unfittable_boxes
        original_count = len(self.boxes)
        self.boxes = trim_unfittable_boxes(self.boxes, self.container)
        trimmed_count = original_count - len(self.boxes)
        
        if trimmed_count > 0:
            messagebox.showwarning(
                "Boxes Trimmed",
                f" {trimmed_count} box(es) were removed because they cannot fit in the new container.\n\n"
                f"Remaining boxes: {len(self.boxes)}"
            )
            self._refresh_box_tree()

    def _apply_custom_container(self):
        errors = []
        name = self._custom_entries["Name"].get().strip() or "Custom"
        dims = {}
        for d in ("Length", "Width", "Height"):
            try:
                v = float(self._custom_entries[d].get())
                if v <= 0:
                    errors.append(f"{d} must be > 0.")
                dims[d] = v
            except ValueError:
                errors.append(f"{d} must be a number.")
                dims[d] = 0.0
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return
        
        self.container = Container(name, dims["Length"], dims["Width"], dims["Height"])
        self._container_var.set("__custom__")
        self._update_container_info()
        
        # Trim boxes when container changes
        from app_engine import trim_unfittable_boxes
        original_count = len(self.boxes)
        self.boxes = trim_unfittable_boxes(self.boxes, self.container)
        trimmed_count = original_count - len(self.boxes)
        
        if trimmed_count > 0:
            messagebox.showwarning(
                "Boxes Trimmed",
                f" {trimmed_count} box(es) were removed because they cannot fit in the custom container.\n\n"
                f"Container: {self.container.length}×{self.container.width}×{self.container.height} cm\n\n"
                f"Remaining boxes: {len(self.boxes)}"
            )
            self._refresh_box_tree()
        
        messagebox.showinfo("Container Set", f" Custom container set:\n{self.container}")
        
    def _update_container_info(self):
        c = self.container
        self._container_info.config(
            text=f"Selected: {c.name}\n"
                 f"Dimensions: {c.length} × {c.width} × {c.height} cm\n"
                 f"Volume: {c.volume/1e6:.3f} m³  ({c.volume:,.0f} cm³)"
        )

    # ── TAB 2: Boxes ──────────────────────────────────────────────────────────

    def _build_boxes_tab(self, parent):
        sec = self._section(parent, "Load from CSV")
        self._btn(sec, "Load CSV file", self._load_csv, bg=C["purple_btn"]).pack(
            fill="x", pady=2)

        sec2 = self._section(parent, "Add a box manually")
        self._box_entries: dict = {}
        row_f = tk.Frame(sec2, bg=C["bg_panel"]); row_f.pack(fill="x")
        for i, lbl in enumerate(("Length", "Width", "Height", "Weight(kg)")):
            tk.Label(row_f, text=lbl, bg=C["bg_panel"], fg=C["fg_lavender"],
                     font=("Helvetica", 8)).grid(row=0, column=i*2, padx=(4, 0))
            e = tk.Entry(row_f, width=7, font=("Helvetica", 9),
                         bg=C["bg_light"], fg=C["fg_dark"], relief="flat")
            e.grid(row=0, column=i*2+1, padx=(2, 4))
            self._box_entries[lbl] = e
        row2 = tk.Frame(sec2, bg=C["bg_panel"]); row2.pack(fill="x", pady=4)
        self._fragile_var = tk.BooleanVar()
        tk.Checkbutton(
            row2, text="Fragile", variable=self._fragile_var,
            bg=C["bg_panel"], fg=C["fg_lavender"],
            selectcolor=C["bg_card"], font=("Helvetica", 9),
        ).pack(side="left")
        self._btn(row2, "Add Box", self._add_box_manual, bg=C["pink_mid"]).pack(side="right")

        sec3 = self._section(parent, "Box list")
        cols = ("id", "L", "W", "H", "kg", "fragile")
        self._box_tree = ttk.Treeview(sec3, columns=cols, show="headings", height=8)
        for col in cols:
            self._box_tree.heading(col, text=col)
            self._box_tree.column(col, width=55, anchor="center")
        sc = ttk.Scrollbar(sec3, orient="vertical", command=self._box_tree.yview)
        self._box_tree.configure(yscrollcommand=sc.set)
        self._box_tree.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")

        row3 = tk.Frame(parent, bg=C["bg_panel"]); row3.pack(fill="x", padx=8, pady=4)
        self._btn(row3, "🗑 Remove", self._remove_box, bg=C["red"]).pack(side="left")
        self._btn(row3, "Clear All", self._clear_boxes, bg=C["purple_dark"]).pack(
            side="left", padx=6)
        self._boxes_label = tk.Label(
            parent, text="No boxes loaded.",
            font=("Helvetica", 9, "italic"), bg=C["bg_panel"], fg=C["purple_hi"],
        )
        self._boxes_label.pack(pady=2)

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Open boxes CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: 
            return
        try:
            df = pd.read_csv(path)
            df["fragile"] = (df["fragile"].astype(str).str.lower()
                            .map({"true": True, "false": False, "1": True, "0": False})
                            .fillna(False))
            self.boxes = [
                Box(id=int(r["id"]), length=float(r["length"]),
                    width=float(r["width"]), height=float(r["height"]),
                    weight_kg=float(r["weight_kg"]), fragile=bool(r["fragile"]))
                for _, r in df.iterrows()
            ]
            
            from app_engine import trim_unfittable_boxes, trim_boxes_to_capacity
            original_count = len(self.boxes)
            self.boxes = trim_unfittable_boxes(self.boxes, self.container)
            dim_removed = original_count - len(self.boxes)

            # Check volume
            total_volume = sum(b.volume for b in self.boxes)
            container_volume = self.container.volume

            if total_volume > container_volume:
                self.boxes, utilization = trim_boxes_to_capacity(self.boxes, self.container)
                self._refresh_box_tree()
                
                msg = f"Loaded {len(self.boxes)} boxes"
                if dim_removed > 0:
                    msg += f"\nRemoved {dim_removed} boxes that cannot fit dimensionally"
                msg += f"\nVolume utilization: {utilization:.1f}%"
                messagebox.showinfo("Loaded", msg)
            else:
                self._refresh_box_tree()
                msg = f"Loaded {len(self.boxes)} boxes"
                if dim_removed > 0:
                    msg += f"\nRemoved {dim_removed} boxes that cannot fit dimensionally"
                messagebox.showinfo("Loaded", msg)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load CSV:\n{e}")

    def _add_box_manual(self):
        try:
            l  = float(self._box_entries["Length"].get())
            w  = float(self._box_entries["Width"].get())
            h  = float(self._box_entries["Height"].get())
            kg = float(self._box_entries["Weight(kg)"].get())
            if l <= 0 or w <= 0 or h <= 0 or kg < 0: raise ValueError
            new_id = max((b.id for b in self.boxes), default=0) + 1
            self.boxes.append(Box(id=new_id, length=l, width=w, height=h,
                                  weight_kg=kg, fragile=self._fragile_var.get()))
            self._refresh_box_tree()
            for e in self._box_entries.values(): e.delete(0, "end")
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter valid positive numbers.")

    def _remove_box(self):
        for item in self._box_tree.selection():
            vid = self._box_tree.item(item)["values"][0]
            self.boxes = [b for b in self.boxes if b.id != vid]
        self._refresh_box_tree()

    def _clear_boxes(self):
        if messagebox.askyesno("Confirm", "Clear all boxes?"):
            self.boxes = []; self._refresh_box_tree()

    def _refresh_box_tree(self):
        self._box_tree.delete(*self._box_tree.get_children())
        for b in self.boxes:
            self._box_tree.insert("", "end", values=(
                b.id, b.length, b.width, b.height,
                b.weight_kg, "Yes" if b.fragile else "No"))
        n = len(self.boxes)
        self._boxes_label.config(
            text=f"{n} box{'es' if n != 1 else ''} loaded  |  "
                 f"{sum(1 for b in self.boxes if b.fragile)} fragile")

    # ── TAB 3: Run ────────────────────────────────────────────────────────────

    def _build_run_tab(self, parent):
        sec = self._section(parent, "Step 1 — Choose base algorithm")
        tk.Label(sec, text="Run Greedy, Smart Greedy or Genetic Algorithm first:",
                 bg=C["bg_panel"], fg=C["fg_lavender"],
                 font=("Helvetica", 9)).pack(anchor="w")

        bf = tk.Frame(sec, bg=C["bg_panel"]); bf.pack(fill="x", pady=6)
        self._btn_smart_greedy = tk.Button(
            bf, text="▶ Smart Greedy", font=("Helvetica", 10, "bold"),
            bg=C["purple_btn"], fg=C["fg_white"], relief="flat",
            padx=10, pady=8, cursor="hand2", width=16,
            command=self._run_smart_greedy)
        self._btn_smart_greedy.pack(side="left", padx=(0, 8))

        self._btn_greedy = tk.Button(
            bf, text="▶ Greedy", font=("Helvetica", 10, "bold"),
            bg=C["purple_btn"], fg=C["fg_white"], relief="flat",
            padx=10, pady=8, cursor="hand2", width=16,
            command=lambda: self._run_algo("greedy"))
        self._btn_greedy.pack(side="left", padx=(0, 8))

        self._btn_ga = tk.Button(
            bf, text="▶ Genetic", font=("Helvetica", 10, "bold"),
            bg=C["pink_hot"], fg=C["fg_white"], relief="flat",
            padx=10, pady=8, cursor="hand2",
            command=lambda: self._run_algo("ga"))
        self._btn_ga.pack(side="left")

        self._progress = ttk.Progressbar(sec, mode="indeterminate", length=400)
        self._progress.pack(fill="x", pady=4)
        self._result_label = tk.Label(
            sec, text="No results yet.",
            font=("Helvetica", 10), bg=C["bg_card"],
            fg=C["green"], relief="flat", pady=8, wraplength=400)
        self._result_label.pack(fill="x", pady=4)

        sec2 = self._section(parent, "Step 2 — Improve with Simulated Annealing")
        tk.Label(sec2, text="Apply SA to refine the baseline result:",
                bg=C["bg_panel"], fg=C["fg_lavender"],
                font=("Helvetica", 9)).pack(anchor="w")
        
        btn_frame = tk.Frame(sec2, bg=C["bg_panel"])
        btn_frame.pack(fill="x", pady=6)
        
        self._btn_sa = tk.Button(
            btn_frame, text=" Improve with SA",
            font=("Helvetica", 10, "bold"),
            bg=C["pink_mid"], fg=C["fg_white"], relief="flat",
            padx=10, pady=8, cursor="hand2",
            state="disabled", command=self._run_sa)
        self._btn_sa.pack(side="left", padx=(0, 8))
        
        self._btn_sa_interactive = tk.Button(
            btn_frame, text="  SA with User Input",
            font=("Helvetica", 10, "bold"),
            bg=C["purple_btn"], fg=C["fg_white"], relief="flat",
            padx=10, pady=8, cursor="hand2",
            state="disabled", command=self._run_sa_interactive)
        self._btn_sa_interactive.pack(side="left")
        
        self._sa_label = tk.Label(
            sec2, text="", font=("Helvetica", 10),
            bg=C["bg_card"], fg=C["amber"],
            relief="flat", pady=6, wraplength=400)
        self._sa_label.pack(fill="x")

        sec3 = self._section(parent, "SA Parameters")
        self._sa_params: dict = {}
        for lbl, default in [
            ("Start Temp", "150"), ("End Temp", "5"),
            ("Cooling Rate", "0.97"), ("Iters/Step", "6"),
            ("Target %", "80"),
        ]:
            row = tk.Frame(sec3, bg=C["bg_panel"]); row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl + ":", bg=C["bg_panel"],
                     fg=C["fg_lavender"], font=("Helvetica", 9),
                     width=14, anchor="w").pack(side="left")
            e = tk.Entry(row, width=10, font=("Helvetica", 9),
                         bg=C["bg_light"], fg=C["fg_dark"], relief="flat")
            e.insert(0, default); e.pack(side="left", padx=4)
            self._sa_params[lbl] = e

    # ── TAB 4: Edit Placed Boxes ──────────────────────────────────────────────

    def _build_edit_tab(self, parent):
        tk.Label(
            parent,
            text="After running an algorithm, select a placed box\n"
                 "to flip its orientation or change its position.\n"
                 "Only non-fragile boxes can be modified.",
            bg=C["bg_panel"], fg=C["purple_hi"],
            font=("Helvetica", 9), justify="left", pady=6,
        ).pack(padx=12, anchor="w")

        sec = self._section(parent, "Placed boxes")
        cols = ("id", "x", "y", "z", "L", "W", "H", "kg", "fragile")
        self._placed_tree = ttk.Treeview(sec, columns=cols, show="headings", height=10)
        widths = {"id": 40, "x": 60, "y": 60, "z": 60,
                  "L": 55, "W": 55, "H": 55, "kg": 55, "fragile": 55}
        for col in cols:
            self._placed_tree.heading(col, text=col)
            self._placed_tree.column(col, width=widths.get(col, 55), anchor="center")
        sc2 = ttk.Scrollbar(sec, orient="vertical", command=self._placed_tree.yview)
        self._placed_tree.configure(yscrollcommand=sc2.set)
        self._placed_tree.pack(side="left", fill="both", expand=True)
        sc2.pack(side="right", fill="y")
        self._placed_tree.bind("<<TreeviewSelect>>", self._on_placed_select)

        row = tk.Frame(parent, bg=C["bg_panel"]); row.pack(fill="x", padx=8, pady=6)
        self._btn_flip = tk.Button(
            row, text="✏  Flip / Reposition Selected Box",
            font=("Helvetica", 9, "bold"),
            bg=C["purple_btn"], fg=C["fg_white"], relief="flat",
            padx=10, pady=6, cursor="hand2", state="disabled",
            command=self._flip_box)
        self._btn_flip.pack(fill="x")
        self._edit_msg = tk.Label(
            parent, text="", font=("Helvetica", 9),
            bg=C["bg_panel"], fg=C["purple_hi"], wraplength=400)
        self._edit_msg.pack(padx=12, anchor="w")

    def _on_placed_select(self, event):
        sel = self._placed_tree.selection()
        if not sel:
            self._btn_flip.config(state="disabled"); return
        vals = self._placed_tree.item(sel[0])["values"]
        box_id, is_fragile = vals[0], vals[8] == "Yes"
        self.selected_box_id = box_id
        if self.last_result:
            self._render_3d(self.last_result,
                            f"{self.last_algo} — Box #{box_id} selected",
                            highlight_id=box_id)
        if is_fragile:
            self._btn_flip.config(state="disabled")
            self._edit_msg.config(text="Fragile boxes cannot be modified.")
        else:
            self._btn_flip.config(state="normal")
            self._edit_msg.config(text=f"Box #{box_id} selected. Click Flip/Reposition to edit.")

    def _flip_box(self):
        if not self.selected_box_id or not self.last_result: return
        pb = next((p for p in self.last_result if p["id"] == self.selected_box_id), None)
        box_obj = next((b for b in self.boxes if b.id == self.selected_box_id), None)
        if not pb or not box_obj: return

        def on_apply(new_ori, new_pos):
            self._forced_orientations[self.selected_box_id] = new_ori
            seq = [b for b in self.boxes if b.id in {p["id"] for p in self.last_result}]
            placed, util = pack_sequence_with_forced(seq, self.container, self._forced_orientations)
            self.last_result = placed; self.last_util = util
            self._refresh_placed_tree()
            self._render_3d(placed, f"{self.last_algo} (edited) — {util:.1f}%")
            self._edit_msg.config(
                text=f"Box #{self.selected_box_id} updated! Utilization: {util:.2f}%")

        FlipBoxDialog(self, pb, box_obj, self.container, on_apply)

    def _refresh_placed_tree(self):
        self._placed_tree.delete(*self._placed_tree.get_children())
        if not self.last_result: return
        for p in self.last_result:
            x, y, z = p["pos"];  l, w, h = p["dim"]
            self._placed_tree.insert("", "end", values=(
                p["id"], round(x, 1), round(y, 1), round(z, 1),
                l, w, h, p["weight"], "Yes" if p["fragile"] else "No"))

    # ── RIGHT PANEL: 3-D Visualisation ────────────────────────────────────────

    def _build_right(self, parent):
        top_row = tk.Frame(parent, bg=C["bg_card"]); top_row.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(
            top_row, text="3D Packing Visualization",
            font=("Helvetica", 12, "bold"),
            bg=C["bg_card"], fg=C["pink_soft"],
        ).pack(side="left")

        angle_frame = tk.Frame(top_row, bg=C["bg_card"]); angle_frame.pack(side="right")
        for label, elev, azim in [("Front", 0, 0), ("Side", 0, 90),
                                   ("Top", 90, 0), ("ISO", 25, 45)]:
            tk.Button(
                angle_frame, text=label, font=("Helvetica", 8),
                bg=C["purple_btn"], fg=C["fg_white"], relief="flat",
                padx=6, pady=2, cursor="hand2",
                command=lambda e=elev, a=azim: self._set_view_angle(e, a),
            ).pack(side="left", padx=2)

        # Matplotlib figure with a light card background
        self._fig = plt.Figure(figsize=(7.5, 6), dpi=90,
                               facecolor=C["bg_card"])
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_facecolor(C["bg_card"])
        self._ax.tick_params(colors=C["fg_lavender"], labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(C["purple_hi"])
        self._ax.set_title("Run an algorithm to see the result",
                           fontsize=10, color=C["purple_hi"])

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)

        toolbar_frame = tk.Frame(parent, bg=C["bg_card"]); toolbar_frame.pack(fill="x", padx=8)
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        self._stats_bar = tk.Label(
            parent, text="",
            font=("Helvetica", 10, "bold"),
            bg=C["purple_dark"], fg=C["pink_hot"],
            relief="flat", pady=6,
        )
        self._stats_bar.pack(fill="x", padx=8, pady=(2, 8))

        tk.Label(
            parent,
            text="🖱 Left-drag: rotate  |  Right-drag / scroll: zoom  |  Toolbar: save / pan",
            font=("Helvetica", 8), fg=C["purple_hi"], bg=C["bg_card"],
        ).pack(pady=(0, 4))

    def _set_view_angle(self, elev, azim):
        self._ax.view_init(elev=elev, azim=azim)
        self._canvas.draw()

    def _on_scroll(self, event):
        ax = self._ax
        factor = 0.9 if event.button == "up" else 1.1
        for get_lim, set_lim in [
            (ax.get_xlim, ax.set_xlim),
            (ax.get_ylim, ax.set_ylim),
            (ax.get_zlim, ax.set_zlim),
        ]:
            lo, hi = get_lim()
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * factor
            set_lim(mid - half, mid + half)
        self._canvas.draw_idle()

    # ── RUN / CALLBACKS ───────────────────────────────────────────────────────

    def _validate_ready(self) -> bool:
        if not self.boxes:
            messagebox.showwarning("No Boxes", "Please load or add boxes first.")
            return False
        if not self.container:
            messagebox.showwarning("No Container", "Please select a container first.")
            return False
        
        from app_engine import trim_unfittable_boxes, trim_boxes_to_capacity
        
        # removing dimensionally unfittable boxes
        original_count = len(self.boxes)
        self.boxes = trim_unfittable_boxes(self.boxes, self.container)
        dim_removed = original_count - len(self.boxes)
        
        if dim_removed > 0:
            print(f"Removed {dim_removed} dimensionally unfittable boxes")
        
        # trimming by volume 
        total_volume = sum(b.volume for b in self.boxes)
        container_volume = self.container.volume
        
        if total_volume > container_volume:
            response = messagebox.askyesno(
                "Volume Exceeds Capacity",
                f"Total box volume ({total_volume:,.0f} cm³) exceeds container capacity ({container_volume:,.0f} cm³).\n\n"
                f"Trim boxes to fit? (Yes = keep boxes in current order until fit, No = keep all boxes)"
            )
            
            if response:
                # keep boxes in order until capacity
                self.boxes, utilization = trim_boxes_to_capacity(self.boxes, self.container)
                self._refresh_box_tree()
                
                messagebox.showinfo(
                    "Boxes Trimmed",
                    f"Trimmed to {len(self.boxes)} boxes\n"
                    f"Volume: {sum(b.volume for b in self.boxes):,.0f} / {container_volume:,.0f} cm³\n"
                    f"Utilization: {utilization:.1f}%"
                )
        
        if not self.boxes:
            messagebox.showwarning("No Boxes", "No boxes remain after trimming. Please load different boxes.")
            return False
        
        return True

    def _run_algo(self, algo: str):
        if not self._validate_ready(): return
        self._progress.start()
        for btn in (self._btn_greedy, self._btn_ga, self._btn_sa):
            btn.config(state="disabled")
        self._result_label.config(text="Running…")
        self._forced_orientations = {}

        def task():
            t0 = time.time()
            if algo == "greedy":
                placed, util, _ = greedy_pack(self.boxes, self.container)
                name = "Greedy BFD"
            else:
                def prog(gen, total, best):
                    self._result_label.config(
                        text=f"GA: gen {gen}/{total}  best={best:.1f}%")
                placed, util, _, _ = genetic_algorithm(
                    self.boxes, self.container,
                    pop_size=30, generations=50, progress_cb=prog)
                name = "Genetic Algorithm"
            rt = time.time() - t0
            self.after(0, lambda: self._on_algo_done(placed, util, name, rt))

        threading.Thread(target=task, daemon=True).start()

    def _on_algo_done(self, placed, util, name, rt):
        self._progress.stop()
        for btn in (self._btn_greedy, self._btn_ga, self._btn_sa, self._btn_sa_interactive):
            btn.config(state="normal")
        self._result_label.config(
            text=f" {name} done!\n"
                f"Placed: {len(placed)}/{len(self.boxes)}  |  "
                f"Utilization: {util:.2f}%  |  Time: {rt:.1f}s")
        self.last_result = placed
        self.last_util = util
        self.last_algo = name
        self._render_3d(placed, f"{name} — {util:.1f}% utilization")
        self._refresh_placed_tree()

# ──── Smart Greedy ─────────────────────────────────────────
    def _run_smart_greedy(self):
        """Run Smart Greedy with multiple strategies."""
        if not self._validate_ready():
            return
        
        self._progress.start()
        for btn in (self._btn_greedy, self._btn_ga, self._btn_sa, self._btn_smart_greedy):
            btn.config(state="disabled")
        self._result_label.config(text="Smart Greedy running (trying multiple strategies)...")
        self._forced_orientations = {}
        
        def task():
            t0 = time.time()
            
            def prog(current, total, msg):
                self.after(0, lambda: self._result_label.config(
                    text=f"Smart Greedy: {msg} ({current+1}/{total})"))
            
            placed, util, strategy = smart_greedy_pack(
                self.boxes, self.container, progress_cb=prog
            )
            rt = time.time() - t0
            
            self.after(0, lambda: self._on_smart_greedy_done(placed, util, strategy, rt))
        
        threading.Thread(target=task, daemon=True).start()

    def _on_smart_greedy_done(self, placed, util, strategy, rt):
        self._progress.stop()
        for btn in (self._btn_greedy, self._btn_ga, self._btn_sa, self._btn_smart_greedy, self._btn_sa_interactive):
            btn.config(state="normal")
        self._result_label.config(
            text=f"Smart Greedy ({strategy}) done!\n"
                 f"Placed: {len(placed)}/{len(self.boxes)}  |  "
                 f"Utilization: {util:.2f}%  |  Time: {rt:.1f}s")
        self.last_result = placed
        self.last_util = util
        self.last_algo = f"Smart Greedy ({strategy})"
        self._render_3d(placed, f"Smart Greedy ({strategy}) — {util:.1f}% utilization")
        self._refresh_placed_tree()

    # ── SIMULATED ANNEALING (REGULAR) ─────────────────────────────────────────

    def _run_sa(self):
        """Run regular Simulated Annealing (non-interactive)."""
        if not self.last_result:
            messagebox.showwarning("No Baseline", "Run Greedy, Smart Greedy, or GA first.")
            return
        
        try:
            T_start = float(self._sa_params["Start Temp"].get())
            T_end   = float(self._sa_params["End Temp"].get())
            cooling = float(self._sa_params["Cooling Rate"].get())
            iters   = int(self._sa_params["Iters/Step"].get())
        except ValueError:
            messagebox.showerror("Invalid Params", "Check SA parameter values.")
            return

        self._progress.start()
        self._btn_sa.config(state="disabled")
        self._sa_label.config(text="SA running…")
        last_ids = [p["id"] for p in self.last_result]
        init_seq = sorted(
            self.boxes,
            key=lambda b: last_ids.index(b.id) if b.id in last_ids else 999)

        def task():
            t0 = time.time()
            def prog(T, T0, best):
                self.after(0, lambda: self._sa_label.config(
                    text=f"SA: T={T:.2f}  best={best:.2f}%"))
            
            placed, util, _ = simulated_annealing(
                self.boxes, self.container,
                initial_sequence=init_seq,
                T_start=T_start, T_end=T_end,
                cooling=cooling, iters_per_step=iters,
                progress_cb=prog)
            rt = time.time() - t0
            self.after(0, lambda: self._on_sa_done(placed, util, rt))

        threading.Thread(target=task, daemon=True).start()

    def _on_sa_done(self, placed, util, rt):
        self._progress.stop()
        self._btn_sa.config(state="normal")
        imp = util - self.last_util
        self._sa_label.config(
            text=f" SA done!  {util:.2f}%  (+{imp:.2f}% improvement)  {rt:.1f}s")
        self.last_result = placed
        self.last_util = util
        self.last_algo = f"SA Improved"
        self._render_3d(placed, f"SA Improved — {util:.1f}% utilization")
        self._refresh_placed_tree()

    # ── SIMULATED ANNEALING (INTERACTIVE) ─────────────────────────────────────
    def _run_sa_interactive(self):
        """Run Simulated Annealing with user interaction (asks to continue)."""
        if not self.last_result:
            messagebox.showwarning("No Baseline", "Run Greedy, Smart Greedy, or GA first.")
            return
        
        try:
            T_start = float(self._sa_params["Start Temp"].get())
            T_end   = float(self._sa_params["End Temp"].get())
            cooling = float(self._sa_params["Cooling Rate"].get())
            iters   = int(self._sa_params["Iters/Step"].get())
            target  = float(self._sa_params["Target %"].get())
        except ValueError:
            messagebox.showerror("Invalid Params", "Check SA parameter values.")
            return
        
        self._progress.start()
        self._btn_sa.config(state="disabled")
        self._btn_sa_interactive.config(state="disabled")
        self._btn_sa_interactive.config(text="SA Running...")
        self._sa_label.config(text="SA Interactive running...")
        
        # Get initial sequence from last result
        last_ids = [p["id"] for p in self.last_result]
        init_seq = sorted(
            self.boxes,
            key=lambda b: last_ids.index(b.id) if b.id in last_ids else 999)
    
        def user_ask_cb(current_util, pct_of_max, iteration):
            """Ask user via dialog whether to continue."""
            response = [None]
            
            def ask():
                response[0] = messagebox.askyesno(
                    "Target Reached!",
                    f"Reached {pct_of_max:.1f}% of theoretical max at iteration {iteration}!\n"
                    f"Current utilization: {current_util:.1f}%\n\n"
                    f"Continue searching for better solution?",
                    parent=self
                )
            
            self.after(0, ask)
            
            while response[0] is None:
                self.update()
                time.sleep(0.1)
            return response[0]
    
        def task():
            t0 = time.time()
            
            def prog(T, best, iteration):
                self.after(0, lambda: self._sa_label.config(
                    text=f"SA: T={T:.2f}  iter={iteration}  best={best:.1f}%"))
            
            try:
                placed, util, _ = simulated_annealing_interactive(
                    self.boxes, self.container,
                    initial_sequence=init_seq,
                    T_start=T_start, T_end=T_end,
                    cooling=cooling, iters_per_step=iters,
                    target_pct=target,
                    progress_cb=prog,
                    user_ask_cb=user_ask_cb
                )
                rt = time.time() - t0
                self.after(0, lambda: self._on_sa_interactive_done(placed, util, rt))
            except Exception as e:
                self.after(0, lambda: self._on_sa_interactive_error(str(e)))
        
        threading.Thread(target=task, daemon=True).start()


    def _on_sa_interactive_done(self, placed, util, rt):
        self._progress.stop()
        self._btn_sa.config(state="normal")
        self._btn_sa_interactive.config(state="normal")
        self._btn_sa_interactive.config(text="SA with User Input")
        imp = util - self.last_util
        self._sa_label.config(
            text=f"SA Interactive done!  {util:.2f}%  (+{imp:.2f}% improvement)  {rt:.1f}s")
        self.last_result = placed
        self.last_util = util
        self.last_algo = f"SA Interactive"
        self._render_3d(placed, f"SA Interactive — {util:.1f}% utilization")
        self._refresh_placed_tree()

    def _on_sa_interactive_error(self, error_msg):
        self._progress.stop()
        self._btn_sa.config(state="normal")
        self._btn_sa_interactive.config(state="normal")
        self._btn_sa_interactive.config(text=" SA with User Input")
        self._sa_label.config(text=f"Error: {error_msg[:50]}...")
        messagebox.showerror("SA Interactive Error", f"An error occurred:\n{error_msg}")

    # ── 3-D RENDER ────────────────────────────────────────────────────────────

    def _render_3d(self, placed, title, highlight_id=None):
        elev, azim = self._ax.elev, self._ax.azim
        self._ax.cla()
        self._ax.set_facecolor(C["bg_card"])
        c = self.container

        # Container wireframe in light purple
        verts = [
            (0, 0, 0), (c.length, 0, 0), (c.length, c.width, 0), (0, c.width, 0),
            (0, 0, c.height), (c.length, 0, c.height),
            (c.length, c.width, c.height), (0, c.width, c.height),
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            p1, p2 = verts[e[0]], verts[e[1]]
            self._ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                          color=C["purple_hi"], alpha=0.35, linewidth=0.8)

        # Light purple colour palette for boxes
        normal_cols = [
            "#efe8ff", "#ddcffb", "#cfbcfa", "#bda0fa",
            "#ab8cf5", "#9b7fe6", "#8664ca", "#6c4aa6",
        ]
        fragile_cols = ["#bda0fa", "#ab8cf5", "#8664ca"]
        ni = fi = 0

        for pb in placed:
            is_hl = (highlight_id is not None and pb["id"] == highlight_id)
            if pb["fragile"]:
                color = fragile_cols[fi % len(fragile_cols)]; fi += 1
            else:
                color = normal_cols[ni % len(normal_cols)]; ni += 1
            draw_box_3d(self._ax, pb["pos"], pb["dim"], color,
                        alpha=0.85 if is_hl else 0.65, highlight=is_hl)

        self._ax.view_init(elev=elev, azim=azim)
        self._ax.set_xlim(0, c.length); self._ax.set_ylim(0, c.width)
        self._ax.set_zlim(0, c.height)
        self._ax.set_xlabel("Length (cm)", fontsize=7, color=C["fg_lavender"])
        self._ax.set_ylabel("Width  (cm)", fontsize=7, color=C["fg_lavender"])
        self._ax.set_zlabel("Height (cm)", fontsize=7, color=C["fg_lavender"])
        self._ax.tick_params(colors=C["fg_lavender"], labelsize=6)
        self._ax.set_title(title, fontsize=9, fontweight="bold", color=C["pink_soft"])

        handles = [
            mpatches.Patch(color="#ab8cf5", label="Normal"),
            mpatches.Patch(color="#8664ca", label="Fragile"),
        ]
        if highlight_id:
            handles.append(mpatches.Patch(color="gold", label=f"Selected #{highlight_id}"))
        self._ax.legend(handles=handles, loc="upper left", fontsize=7,
                        facecolor=C["bg_card"], edgecolor=C["purple_hi"],
                        labelcolor=C["fg_lavender"])

        self._canvas.draw()

        vol_used = sum(p["dim"][0] * p["dim"][1] * p["dim"][2] for p in placed)
        util_pct = vol_used / c.volume * 100
        self._stats_bar.config(
            text=f"Container: {c.name}   |   "
                 f"Boxes: {len(placed)}/{len(self.boxes)}   |   "
                 f"{vol_used/1e6:.3f} m³ / {c.volume/1e6:.3f} m³   |   "
                 f"Utilization: {util_pct:.2f}%")



# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
