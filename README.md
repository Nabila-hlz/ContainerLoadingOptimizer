# The-3D-Container-Loading-Optimizer

## Project Summary

This repository contains a complete 3D container loading optimizer built as a desktop application. It is a 2nd-year AI project that demonstrates real-world packing optimization using ZR Express shipment data, notebook-backed algorithm execution, and a custom Tkinter GUI.

The system combines:
- real loading data from ZR Express,
- a powerful desktop visualization interface,
- greedy, genetic, and simulated annealing packing algorithms,
- notebook integration to reuse experimental notebook code,
- flexible container presets and custom packing workflows.

## Why This Project Matters

Efficient container loading is a critical logistics problem for shipping, warehousing, and transport planning. This project showcases how algorithmic optimization can improve space utilization, reduce wasted volume, and make packing decisions more predictable for real shipments.

## Data Source and Attribution

This repository includes a preprocessed dataset derived from ZR Express loading data. Most additional sample datasets were removed, leaving the current available dataset for app and experiment use.

Included dataset:
- `data/boxes_360_preprocessed.csv`

This file contains box dimensions, weights, fragility flags, and packing metadata used by the app and optimization engine.

## Application Architecture

The main application architecture is:

`app_modified.py` -> `app_engine.py` -> `notebooks/NoteBook.ipynb`

- `app_modified.py` is the desktop application entry point.
- `app_engine.py` contains the optimizer API, model classes, container definitions, and notebook integration logic.
- `notebooks/NoteBook.ipynb` contains the original packing algorithm implementations and experiment code.

### Notebook Integration

The app uses a notebook-backed runtime to import algorithm implementations without rewriting the notebook code.
- `app_engine.py` reads `notebooks/NoteBook.ipynb`.
- It extracts relevant classes and function definitions.
- The notebook code is executed in a runtime module.
- Wrapper functions such as `greedy_pack`, `genetic_algorithm`, and `simulated_annealing` are exposed to the GUI.

This lets the desktop app reuse notebook research code while keeping the GUI and engine separated from the raw notebook.



### Performance Note

The notebook-backed greedy implementation is still slower than a native `optimizer.py`-style backend because:
- notebook parsing and compilation add overhead,
- the notebook packing logic includes heavier free-space cleanup,
- the notebook path is optimized for research, not maximum GUI speed.

For production-quality use, the fastest backend remains the original optimized algorithm implementation.


## Features

- interactive 3D visualization of containers and placed boxes,
- zoom, rotate, and selection controls,
- box orientation and reposition editing for non-fragile items,
- editable container presets and custom dimensions,
- dataset-driven packing with real-world ZR Express box data,
- support for fragile items and placement stability checks,
- optimized notebook-backed algorithm integration,
- multiple optimization modes: greedy, genetic, and simulated annealing.

## Full Repository Structure

- `app_modified.py` — main desktop application entry point and GUI launcher.
- `app_engine.py` — core optimizer engine, packing API, notebook loader, and algorithm wrappers.
- `notebooks/NoteBook.ipynb` — original algorithm research notebook.
- `data/` — current available preprocessed dataset used for experiments and evaluation.
- `README.md` — project documentation.

## Setup and Run Instructions

### 1. Install dependencies

```powershell
python -m pip install pandas numpy matplotlib
```

On Windows, `tkinter` is typically included with standard Python. If not, install the Python Tcl/Tk support package.

### 2. Start the app

```powershell
python app_modified.py
```

The application launches the desktop GUI and loads the optimizer engine from `app_engine.py`.

## Usage Notes

- Use the GUI to select a container type or enter custom container dimensions.
- Load the available preprocessed dataset from the `data/` folder.
- Run greedy, genetic, or simulated annealing packing modes.
- Inspect the 3D placement view and adjust box orientation when needed.

## Recommended Workflow

1. Begin with greedy packing to get a fast baseline.
2. Use genetic algorithm for broader sequence improvement.
3. Use simulated annealing for fine-tuned packing quality when time allows.

## Important Project Notes

- The dataset is sourced from ZR Express and represents real logistics packing use cases.
- The notebook integration enables rapid research reuse without rewriting algorithm logic.
- The GUI is designed to show both algorithm behavior and packing quality clearly.
- This repository provides both a research-focused notebook and a practical desktop application.

## Contact and Credits

This project was developed as a 2nd-year AI project in container loading optimization. It highlights algorithm design, notebook integration, desktop UX, and real dataset experiments.

