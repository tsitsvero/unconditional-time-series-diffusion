# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import tarfile
from pathlib import Path
from urllib import request

from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path

default_dataset_path: Path = get_download_path() / "datasets"
wiki2k_download_link: str = "https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz"  # noqa: E501


def get_gts_dataset(dataset_name: str):
    """Load a GluonTS dataset"""
    if dataset_name == "wiki2000_nips":
        wiki_dataset_path = default_dataset_path / dataset_name
        Path(default_dataset_path).mkdir(parents=True, exist_ok=True)
        
        print(f"\nDataset storage locations:")
        print(f"Base directory: {default_dataset_path}")
        print(f"Dataset directory: {wiki_dataset_path}")
        print(f"Metadata file: {wiki_dataset_path / 'metadata'}")
        print(f"Train data: {wiki_dataset_path / 'train'}")
        print(f"Test data: {wiki_dataset_path / 'test'}\n")
        
        if not wiki_dataset_path.exists():
            tar_file_path = wiki_dataset_path.parent / f"{dataset_name}.tar.gz"
            request.urlretrieve(
                wiki2k_download_link,
                tar_file_path,
            )

            with tarfile.open(tar_file_path) as tar:
                tar.extractall(path=wiki_dataset_path.parent)

            os.remove(tar_file_path)
            
        return load_datasets(
            metadata=wiki_dataset_path / "metadata",
            train=wiki_dataset_path / "train",
            test=wiki_dataset_path / "test",
        )
    else:
        return get_dataset(dataset_name)
