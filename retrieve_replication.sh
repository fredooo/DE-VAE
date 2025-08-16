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

# Download all files
echo "Downloading files..."
for filename in "${!files[@]}"; do
    url="${files[$filename]}"
    output="temp/$filename"
    if [ ! -f "$output" ]; then
        echo "Downloading $filename..."
        curl --progress-bar -L "$url" -o "$output"
    else
        echo "$filename already exists. Skipping download."
    fi
done

# Combine parts
echo "Combining model parts into models.tar.xz ..."
cat temp/models.tar.xz.part-0* > temp/models.tar.xz

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
