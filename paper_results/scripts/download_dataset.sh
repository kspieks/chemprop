#!/bin/bash

# Define directories and paths
script_dir=$(dirname "$(realpath "$0")")  # The dir of current script
paper_results_dir=$(dirname "$script_dir")  # The dir of paper_results
save_path="$paper_results_dir/Datasets.tar.bz2"  # The download path for the dataset tarball
dataset_dir="$paper_results_dir/datasets"  # The dir for storing the downloaded dataset

# Print information about the download
echo "Downloading datasets from Zenodo..."
echo "For more details, please visit https://zenodo.org/records/11493786"

# Download the file
wget -q --show-progress "https://zenodo.org/records/11493786/files/Datasets.tar.bz2?download=1" -O "$save_path"

# Check if the download was successful
if [[ ! -f "$save_path" ]]; then
    echo "Error: Failed to download the dataset tarball. Check your internet connection."
    exit 1
fi

echo "File downloaded and saved to: $save_path"

# Extract the downloaded tarball
echo "Extracting files from the downloaded tarball..."
# mkdir -p "$dataset_dir"  # Ensure dataset directory exists

if tar xvf "$save_path" -C "$paper_results_dir"; then
    echo "Extraction successful."

    # Move extracted folder to the dataset directory
    mv "$paper_results_dir/Datasets" "$dataset_dir" 2>/dev/null

    if [[ -d "$dataset_dir" ]]; then
        echo "Files moved to: $dataset_dir"
    else
        echo "Error: Failed to move extracted files to $dataset_dir."
        exit 1
    fi
else
    echo "Error: Failed to extract the tarball."
    exit 1
fi

# Clean up: remove the tarball after extraction
rm "$save_path"
echo "Cleanup: Removed the tarball from $save_path"

echo "Process completed successfully."
echo "You can access dataset files at $dataset_dir"
