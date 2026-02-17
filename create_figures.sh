#!/bin/bash
source .venv/bin/activate

for model in \
  models/vae-isot-mnist-umap-p20.00-e5.00000-s0.pt \
  models/vae-isot-mnist-umap-p20.00-e5.00000-s1.pt \
  models/vae-isot-mnist-umap-p20.00-e5.00000-s2.pt \
  models/vae-isot-kmnist-tsne-p20.00-e3.00000-s0.pt \
  models/vae-isot-kmnist-tsne-p20.00-e3.00000-s1.pt \
  models/vae-isot-kmnist-tsne-p20.00-e3.00000-s2.pt \
  models/vae-diag-mnist-umap-p20.00-e5.00000-s0.pt \
  models/vae-diag-mnist-umap-p20.00-e5.00000-s1.pt \
  models/vae-diag-mnist-umap-p20.00-e5.00000-s2.pt \
  models/vae-diag-kmnist-tsne-p20.00-e3.00000-s0.pt \
  models/vae-diag-kmnist-tsne-p20.00-e3.00000-s1.pt \
  models/vae-diag-kmnist-tsne-p20.00-e3.00000-s2.pt \
  models/vae-full-mnist-umap-p20.00-e5.00000-s0.pt \
  models/vae-full-mnist-umap-p20.00-e5.00000-s1.pt \
  models/vae-full-mnist-umap-p20.00-e5.00000-s2.pt \
  models/vae-full-kmnist-tsne-p20.00-e3.00000-s0.pt \
  models/vae-full-kmnist-tsne-p20.00-e3.00000-s1.pt \
  models/vae-full-kmnist-tsne-p20.00-e3.00000-s2.pt; do
  echo "=== $model ==="
  python3 visual.py --model "$model"
done
