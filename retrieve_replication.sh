#!/bin/bash

set -e  # Exit on error

# Directories to manage
DIRS=("models" "records" "preprocessed")

# Handle --clear flag
if [[ "$1" == "--clear" ]]; then
    echo "Clearing directories: ${DIRS[*]}"
    for dir in "${DIRS[@]}"; do
        if [ -d "$dir" ]; then
            find "$dir" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
            echo "Cleared $dir"
        fi
    done
    exit 0
fi

# Create directories if they don't exist
mkdir -p temp "${DIRS[@]}"

# File URLs
declare -A files=(
    ["models.tar.xz.part-00"]="https://osf.io/download/ukazf/"
    ["models.tar.xz.part-01"]="https://osf.io/download/kahv9/"
    ["models.tar.xz.part-02"]="https://osf.io/download/2f8ye/"
    ["preprocessed.tar.xz"]="https://osf.io/download/fm6z5/"
    ["records.tar.xz"]="https://osf.io/download/cpvsb/"
)

# SHA256 checksums - populate with actual values from trusted source
declare -A checksums=(
    ["models.tar.xz.part-00"]="b8ccf6f17a09c1972cd7a6fe36e2689257a027c4b7f3364f6f263a7bc871c471"
    ["models.tar.xz.part-01"]="6513b0bdbe003671ecb628584dfd0630daf017cb9adbe662c1211e215edfd3b0"
    ["models.tar.xz.part-02"]="05d229a7bedb3c838aef8c24b9a87724336cd8136304541ee00c0329792fb6de"
    ["models.tar.xz"]="7f6049be9a08ba78b31dd30fb2bad8a506cc53d874ee03f4f0426169c3f1f5f9"
    ["preprocessed.tar.xz"]="c17d3fdd0ff408a134cb00f4a8a78bb6617f2782ae64a3c349b802ed4e118f5e"
    ["records.tar.xz"]="daa91fd3db9866a7359b65d979f8e8760e905434977932b60dc680270f662209"
)

# Verify checksum
verify_checksum() {
    local file="$1"
    local expected="${checksums[$2]}"

    if [ "$expected" == "CHANGEME" ]; then
        echo "WARNING: No checksum defined for $2. Skipping verification."
        return 0
    fi

    local actual=$(sha256sum "$file" | cut -d' ' -f1)
    if [ "$actual" != "$expected" ]; then
        echo "ERROR: Checksum mismatch for $2"
        echo "  Expected: $expected"
        echo "  Got:      $actual"
        return 1
    fi
    echo "Checksum verified for $2"
    return 0
}

# Download and verify all files
echo "Downloading files..."
for filename in "${!files[@]}"; do
    url="${files[$filename]}"
    output="temp/$filename"
    if [ ! -f "$output" ]; then
        echo "Downloading $filename..."
        curl --progress-bar -L "$url" -o "$output"
        verify_checksum "$output" "$filename" || exit 1
    else
        echo "$filename already exists. Verifying checksum..."
        verify_checksum "$output" "$filename" || exit 1
    fi
done

# Combine parts
echo "Combining model parts into models.tar.xz ..."
cat temp/models.tar.xz.part-0* > temp/models.tar.xz
verify_checksum "temp/models.tar.xz" "models.tar.xz" || exit 1

# Extract archives
echo "Extracting models.tar.xz ..."
tar -xJvf temp/models.tar.xz

echo "Extracting records.tar.xz ..."
tar -xJvf temp/records.tar.xz

echo "Extracting preprocessed.tar.xz ..."
tar -xJvf temp/preprocessed.tar.xz

echo "Deleting temp/ ..."
rm -r temp/

echo "Done."
