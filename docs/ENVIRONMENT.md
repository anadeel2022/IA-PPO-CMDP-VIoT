# Recorded execution environment

The final run manifests record the following core environment:

- Python: **3.13.5**
- NumPy: **2.2.6**
- PyTorch: **2.7.1+cpu**

The source package uses only NumPy, pandas, Matplotlib, PyTorch, and pytest as required third-party packages for the manuscript workflow. The portable dependency specification remains in `requirements.txt` because exact PyTorch installation strings differ by platform and accelerator.

The final manuscript experiments were CPU-compatible. GPU acceleration is optional and does not change the frozen experiment definitions.
