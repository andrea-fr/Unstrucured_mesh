# 2D CFD Hybrid Mesh Generator for finite volume method

A Python tool for generating 2D hybrid meshes over rectangular domains, combining structured prism layers near the wall with an unstructured triangular core and defining all the connectivity data that a FVM requires.

- Grows **quadrilateral prism layers** from the bottom wall with geometric spacing
- Fills the remaining domain with **triangular cells** via Poisson disk sampling + Delaunay triangulation
- Applies **Laplacian smoothing** to improve tria mesh quality
- Computes face normals, centroids, and neighbor connectivity for each cell
- Exports mesh data to `.xlsx` and `.npz`
- Plots the mesh and quality metrics (aspect ratio, skewness)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage
Edit the parameters at the top of the script:
```python
lx = 2          # domain width
ly = 1          # domain height
r = 0.05        # minimum node spacing
n_layers = 22   # number of prism layers
h0 = 0.001      # first layer height
growth_rate = 1.2
```

Then run:
```bash
python mesh_generator.py
```

## Output
| File | Content |
|------|---------|
| `mesh_data.xlsx` | Cells, edges and faces in tabular form |
| `mesh_data.npz` | Full mesh arrays (nodes, cells, edges) |
