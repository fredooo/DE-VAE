#!/bin/bash

set -e  # Exit on error

# Create directories if they don't exist
mkdir -p temp models records preprocessed

# File URLs
declare -A files=(
    ["models.tar.gz.00"]="https://osf.io/download/2sujd/"
    ["models.tar.gz.01"]="https://osf.io/download/cgds5/"
    ["models.tar.gz.02"]="https://osf.io/download/3g4qa/"
    ["records.tar.gz"]="https://osf.io/download/rjax6/"
    ["preprocessed.tar.gz"]="https://osf.io/download/32p4q/"
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
echo "Combining model parts into models.tar.gz..."
cat temp/models.tar.gz.0* > temp/models.tar.gz

# Extract archives
echo "Extracting models.tar.gz to models/"
tar -xvzf temp/models.tar.gz -C models/

echo "Extracting records.tar.gz to records/"
tar -xvzf temp/records.tar.gz -C records/

echo "Extracting preprocessed.tar.gz to preprocessed/"
tar -xvzf temp/preprocessed.tar.gz -C preprocessed/

echo "Deleting temp/"
rm -r temp/

echo "Done."
