#!/bin/bash
# Download KMNIST and FashionMNIST from alternative mirrors (bypasses flaky upstream servers)
set -e

KMNIST_DIR="datasets/KMNIST/raw"
FMNIST_DIR="datasets/FashionMNIST/raw"

# KMNIST: original server (codh.rois.ac.jp) is down, use Internet Archive
KMNIST_BASE="https://web.archive.org/web/2024/http://codh.rois.ac.jp/kmnist/dataset/kmnist"
# FashionMNIST: GitHub repo still hosts the files
FMNIST_BASE="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

download_dataset() {
    local dir=$1
    local base_url=$2

    mkdir -p "$dir"
    for f in "${FILES[@]}"; do
        if [ -f "$dir/$f" ]; then
            echo "  [skip] $f already exists"
        else
            echo "  [download] $f"
            curl -L --fail -o "$dir/$f" "$base_url/$f"
        fi
        # Extract (torchvision needs both .gz and extracted)
        local extracted="${f%.gz}"
        if [ ! -f "$dir/$extracted" ]; then
            echo "  [extract] $f"
            gunzip -k "$dir/$f"
        fi
    done
}

echo "=== Downloading KMNIST ==="
download_dataset "$KMNIST_DIR" "$KMNIST_BASE"

echo "=== Downloading FashionMNIST ==="
download_dataset "$FMNIST_DIR" "$FMNIST_BASE"

echo "Done."
