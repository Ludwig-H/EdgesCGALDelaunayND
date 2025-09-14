# EdgesCGALDelaunayND (n-D, .npy I/O)

Lit un nuage `N x d` depuis un `.npy` (float32/float64, C-order), calcule la **triangulation de Delaunay** en dimension `d` avec **CGAL**, et écrit les **arêtes** (indices triés) dans un `.npy` `(M,2)` en `int64`.

- **Entrée**: `.npy` `(N,d)` en float32 ou float64 (C-order). Fortran-order refusé.
- **Sortie**: `.npy` `(M,2)` en int64, lignes `i j` avec `i < j` (1‑squelette de la Delaunay).

## Dépendances (Ubuntu 22.04/24.04)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libcgal-dev libeigen3-dev
# Optionnel pour la voie 3D parallèle:
sudo apt-get install -y libtbb-dev libtbbmalloc2
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Utilisation

```bash
./build/EdgesCGALDelaunayND data/example_points.npy edges.npy
```

## Détails techniques

- **ND générique**: `CGAL::Epick_d<Dynamic_dimension_tag>` + `Delaunay_triangulation` dD.
- **3D fast‑path optionnel**: si `libtbb-dev` + `tbbmalloc` sont détectés à la configuration, on
  construit la Delaunay 3D via `Delaunay_triangulation_3` en **Parallel_tag** avec lock grid CGAL.
  Sinon on reste sur la voie ND (ou 3D séquentiel), sans planter à l’édition de liens.
- **Extraction**: en ND, on itère les **cellules finies** puis on déduplique les paires; en 3D, on parcourt `finite_edges_*`.
- **Parallélisme**: ND utilise **OpenMP** pour l’extraction; 3D utilise TBB si dispo.
- **.npy**: lecture v1.0/2.0 en C‑order (`<f4`/`<f8`), écriture `'<i8'`.

## Exemple Python

```python
import numpy as np
N, d = 2000, 4
pts = np.random.rand(N, d).astype('float64')
np.save('points.npy', pts)
# ./build/EdgesCGALDelaunayND points.npy edges.npy
edges = np.load('edges.npy')
print(edges.shape, edges[:10])
```
