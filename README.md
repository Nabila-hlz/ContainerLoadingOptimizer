# The-3D-Container-Loading-Optimizer
2nd Year AI project
<<<<<<< Updated upstream
Ines: I added the preprocessing part
Ines: I added the generated data to the local one we have 

## Notebook Link Summary

The desktop app can be linked to `notebooks/NoteBook.ipynb` through `notebook_backend.py`.

How it works:
- `app.py` imports from `notebook_backend.py`
- `notebook_backend.py` reads `notebooks/NoteBook.ipynb`
- it extracts the useful Python code cells
- it executes them in memory
- it exposes wrapper functions like `greedy_pack`, `genetic_algorithm`, and `simulated_annealing`
- it converts notebook results into the format expected by the GUI

So the connection is:

`app.py` -> `notebook_backend.py` -> `notebooks/NoteBook.ipynb`

## Greedy Fix Summary

When the app was first linked to the notebook, Greedy looked broken for two reasons:

1. There was a loader bug in `notebook_backend.py`
- the notebook code was executed with `exec()`
- the notebook dataclasses needed a real module in `sys.modules`
- without that, the first real algorithm call could fail at runtime

Fix:
- a runtime module was created with `types.ModuleType`
- it was registered in `sys.modules`
- then the notebook code was executed inside that module namespace

2. The notebook Greedy path is slower than the original app backend
- the first run has notebook parsing + compilation overhead
- the notebook packing logic is heavier than `optimizer.py`
- especially the free-space cleanup logic inside the notebook implementation

Result:
- the first Greedy run now works correctly
- but notebook-backed Greedy is still slower than the original `optimizer.py` approach

## Simulated Annealing Fix Summary

Simulated annealing was also too slow in the app.

Why:
- the notebook SA implementation is expensive for a GUI workflow
- the previous app defaults were very heavy:
  - `Start Temp = 1000`
  - `End Temp = 0.1`
  - `Cooling Rate = 0.995`
  - `Iters/Step = 30`
- those settings produce a very large number of evaluations, so the app can feel frozen

Fix:
- the app backend now routes `simulated_annealing()` through the faster implementation in `optimizer.py`
- the default SA values in `app.py` were reduced to faster GUI-friendly settings:
  - `Start Temp = 150`
  - `End Temp = 5`
  - `Cooling Rate = 0.97`
  - `Iters/Step = 6`
- the SA implementation in `optimizer.py` was also tightened:
  - sequence evaluations are cached
  - progress updates happen more often
  - the search stops early when it is no longer improving the greedy baseline

Result:
- SA is still available in the app
- it should respond much faster with the default settings
- if needed, the user can still manually increase the SA parameters for a deeper search

## Important Note

For experiments and demonstrations, using the notebook backend is fine.
For a fast and stable desktop app, `optimizer.py` is still the better backend.
=======
>>>>>>> Stashed changes
