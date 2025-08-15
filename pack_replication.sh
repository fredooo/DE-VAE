#!/bin/bash

set -e  # Exit on error

# Directories to compress
DIRS=("models" "preprocessed" "records")

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        archive="${dir}.tar.xz"
        echo "Compressing $dir into $archive ..."
        tar --exclude="$dir/.gitkeep" -cJvf "$archive" "$dir"
        echo "$archive created."
    else
        echo "Directory $dir not found. Skipping."
    fi
done

if [ -e "models.tar.xz" ]; then
    echo "Splitting models.tar.xz ..."
    split -b 1G -d models.tar.xz models.tar.xz.part-
else
    echo "File models.tar.xz does not exist."
fi

echo "Done."