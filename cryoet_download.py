# Core imports
import argparse
import json
import os
import subprocess
from pathlib import Path

# Data handling imports
import cryoet_data_portal as portal
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import requests
from PIL import Image
from sklearn.model_selection import train_test_split

# CryoET Data Portal Client
client = portal.Client()


# Data Portal Interaction Functions
def find_dataset_by_id(dataset_id):
    """Find a dataset by its ID."""
    datasets = portal.Dataset.find(client, [portal.Dataset.id == dataset_id])
    return datasets[0]


def get_dataset_to_runs_for_dataset_id(dataset_id):
    """Get mapping of dataset ID to its runs."""
    dataset = find_dataset_by_id(dataset_id)
    return {dataset.id: [run.name for run in dataset.runs]}


def get_run_to_tomograms_for_dataset_id(dataset_id):
    """Get mapping of run names to their tomograms."""
    dataset = find_dataset_by_id(dataset_id)
    return {run.name: run.tomograms for run in dataset.runs}


def get_annotations_for_tomogram(tomogram):
    """Get annotations for a specific tomogram."""
    return portal.Annotation.find(client, [portal.Tomogram.id == tomogram.id])


# File Processing Functions
def download_mrc_for_tomogram(dataset_id, tomogram, output_dir):
    """Download MRC file for a tomogram."""
    url = tomogram.https_mrc_file
    dir_name = os.path.join(
        output_dir, str(dataset_id), tomogram.run.name, str(tomogram.id)
    )
    os.makedirs(dir_name, exist_ok=True)
    local_file = os.path.join(dir_name, f"{tomogram.voxel_spacing}_downloaded.mrc")
    response = requests.get(url)
    with open(local_file, "wb") as f:
        f.write(response.content)
    return local_file


def visualize_slice_and_save(mrc_path, z_slice, tomogram_id, output_dir):
    """Visualize and save a specific slice from an MRC file."""
    with mrcfile.open(mrc_path) as mrc:
        slice = mrc.data[z_slice, :, :]
        plt.imshow(slice, cmap="gray")
        plt.colorbar()
        plt.title(f"Tomogram Slice {z_slice}")
        output_path = os.path.join(output_dir, f"{tomogram_id}_{z_slice}_slice.png")
        plt.savefig(output_path)
        plt.close()


def sync_annotations(dataset_to_runs, tomograms, output_dir, dataset_id):
    """Generate AWS sync commands for annotations."""
    commands = set()
    for dataset_id, run_names in dataset_to_runs.items():
        for run_name in run_names:
            tomograms_in_run = tomograms[run_name]
            for tomogram in tomograms_in_run:
                cmd = f"aws s3 --no-sign-request sync s3://cryoet-data-portal-public/{dataset_id}/{run_name}/Reconstructions/VoxelSpacing{tomogram.voxel_spacing}/Annotations {output_dir}/{dataset_id}/{run_name}/Annotations"
                commands.add(cmd)
    return commands


def process_and_save_all_mrc_layers(mrc_path, output_dir):
    """Process and save all layers from an MRC file as PNG images."""
    mrc_dir = os.path.dirname(mrc_path)
    voxel_spacing = os.path.basename(mrc_path).replace("_downloaded.mrc", "")
    with mrcfile.open(mrc_path) as mrc:
        num_layers = mrc.data.shape[0]
        for z in range(num_layers):
            slice = mrc.data[z, :, :]
            slice_norm = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
            slice_norm = (slice_norm * 255).astype(np.uint8)
            img = Image.fromarray(slice_norm)
            output_path = os.path.join(mrc_dir, f"{voxel_spacing}_{z}_slice.png")
            img.save(output_path)
            # print(f"Processed layer {z}/{num_layers-1}")


# COCO Dataset Functions
# args.output_dir, annotation_files
def create_coco_dataset(
    output_dir, dataset_id, dataset_to_runs, tomograms, all_annotation_files
):
    """Create COCO format dataset from images and annotation files."""
    # Initialize COCO format structure
    coco_format = {"images": [], "annotations": [], "categories": []}

    # Create categories
    for annotation_files in all_annotation_files.values():
        for cat_id, category in enumerate(annotation_files.keys(), 1):
            if category not in [cat["name"] for cat in coco_format["categories"]]:
                coco_format["categories"].append({"id": cat_id, "name": category})

    category_map = {cat["name"]: cat["id"] for cat in coco_format["categories"]}

    image_id = 0
    annotation_id = 0

    # Process each image

    run_names = dataset_to_runs[dataset_id]
    for run_name in run_names:
        image_dir = os.path.join(output_dir, str(dataset_id), run_name)
        tomogram_ids = [tomogram.id for tomogram in tomograms[run_name]]
        for tomogram_id in tomogram_ids:
            tomogram_slices = Path(image_dir) / f"{tomogram_id}"
            # import pdb; pdb.set_trace()
            for img_path in Path(tomogram_slices).glob("*_slice.png"):
                img = Image.open(img_path)
                width, height = img.size

                coco_format["images"].append(
                    {
                        "id": image_id,
                        "file_name": f"{tomogram_slices}/{img_path.name}",
                        "width": width,
                        "height": height,
                    }
                )

                z_index = int(img_path.stem.split("_")[-2])

                # Process annotations for each category
                annotation_files = all_annotation_files[run_name]
                for category, anno_file in annotation_files.items():
                    with open(anno_file) as f:
                        points = [json.loads(line) for line in f]

                    for point in points:
                        if abs(point["location"]["z"] - z_index) <= 0.5:
                            box_size = 30
                            bbox = [
                                int(point["location"]["x"]) - box_size // 2,
                                int(point["location"]["y"]) - box_size // 2,
                                box_size,
                                box_size,
                            ]

                            coco_format["annotations"].append(
                                {
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": category_map[category],
                                    "bbox": bbox,
                                    "area": box_size * box_size,
                                    "iscrowd": 0,
                                }
                            )
                            annotation_id += 1

                image_id += 1

    return coco_format


def create_annotation_mapping(output_dir, dataset_id, run_name):
    """
    Create mapping of class names to their annotation file paths.

    Args:
        output_dir (str): Base directory where annotations are stored

    Returns:
        dict: Mapping of class names to their .ndjson file paths
    """
    annotation_files = {}

    # Walk through the Annotations directory
    annotations_path = Path(output_dir) / str(dataset_id) / run_name / "Annotations"

    # Check all numbered directories (100, 101, etc.)
    for dir_path in sorted(annotations_path.glob("[0-9]*")):
        # Look for .ndjson files
        for file_path in dir_path.glob("*.ndjson"):
            # Get the class name from the filename (before the first hyphen)
            class_name = file_path.stem.split("-")[0]

            # Convert path to relative path string
            relative_path = str(file_path.relative_to(output_dir))

            annotation_files[class_name] = relative_path

    return annotation_files


def process_coco_split(data, prompt_text, clean=True):
    """Process a COCO dataset split."""
    # Clean up image IDs and annotations
    image_ids = set(img["id"] for img in data["images"])
    valid_annotations = [
        anno for anno in data["annotations"] if anno["image_id"] in image_ids
    ]

    # Remove text from images and add to annotations
    for img in data["images"]:
        if "text" in img:
            del img["text"]

    for ann in valid_annotations:
        ann["text"] = prompt_text

    data["annotations"] = valid_annotations

    return data


def main():
    """Main function to process CryoET data."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process CryoET Data")
    parser.add_argument(
        "--dataset_id", type=int, required=True, help="Dataset ID to fetch data for"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    args = parser.parse_args()

    # output_dir = "."
    # dataset_id = 10440
    args = argparse.Namespace(dataset_id=args.dataset_id, output_dir=args.output_dir)
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process dataset
    dataset_to_runs = get_dataset_to_runs_for_dataset_id(args.dataset_id)
    tomograms = get_run_to_tomograms_for_dataset_id(args.dataset_id)

    # Sync annotations and process MRC files
    sync_cmds = sync_annotations(
        dataset_to_runs, tomograms, args.output_dir, args.dataset_id
    )
    for cmd in sync_cmds:
        subprocess.run(cmd.split())

    for run_name, tomogram_list in tomograms.items():
        for tomogram in tomogram_list:
            mrc_path = download_mrc_for_tomogram(
                args.dataset_id, tomogram, args.output_dir
            )
            process_and_save_all_mrc_layers(mrc_path, args.output_dir)

    all_annotations = {}
    # Define annotation files
    for d, runs in dataset_to_runs.items():
        for run_name in runs:
            annotation_files = create_annotation_mapping(args.output_dir, d, run_name)
            all_annotations[run_name] = annotation_files

    # Create COCO datasets
    prompt_text = "Find ferritin complex, beta amylase, beta galactosidase, cytosolic ribosome, thyroglobulin, and virus"
    coco_data = create_coco_dataset(
        args.output_dir, args.dataset_id, dataset_to_runs, tomograms, all_annotations
    )

    # Split into train and val
    train_imgs, val_imgs = train_test_split(coco_data["images"], test_size=0.2)

    # Create train and val datasets
    train_data = coco_data.copy()
    train_data["images"] = train_imgs
    train_data = process_coco_split(train_data, prompt_text)

    val_data = coco_data.copy()
    val_data["images"] = val_imgs
    val_data = process_coco_split(val_data, prompt_text)

    # Save final datasets
    with open("train_coco.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("val_coco.json", "w") as f:
        json.dump(val_data, f, indent=2)

    # Print statistics
    print(f"Training images: {len(train_data['images'])}")
    print(f"Training annotations: {len(train_data['annotations'])}")
    print(f"Validation images: {len(val_data['images'])}")
    print(f"Validation annotations: {len(val_data['annotations'])}")
