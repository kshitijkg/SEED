# Adapted from https://huggingface.co/datasets/laion/laion_100m_vqgan_f8/blob/main/run_vqgan.py for vqgan_f16_16384

import os
import sys
import torch
import hydra
from omegaconf import OmegaConf
import pyrootutils
import argparse
import traceback
import braceexpand
import warnings
import numpy as np
import pandas as pd
import webdataset as wds
import torch.multiprocessing as mp

from tqdm import tqdm
from timeit import default_timer as timer


warnings.filterwarnings("ignore", category=UserWarning)
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

ALLOWED_DATASETS = ["laion", "mmc4"]

EXPECTED_CHUNK_SIZE = 10000


def remove_keys(sample):
    image, metadata = sample
    new_metadata = {}

    keys_to_keep = ['caption', 'similarity']

    for k, v in metadata.items():
        if k in keys_to_keep:
            new_metadata[k] = v
    return image, new_metadata

def get_dataset(dataset_type, path, s3):
    if s3:
        path = f"pipe:aws s3 cp {path} -"

    if dataset_type == "laion":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg", "json")
        )
        dataset = dataset.map(remove_keys)

        return dataset
    elif dataset_type == "mmc4":

        def resize_image(sample):
            keys = ["png", "jpg", "jpeg"]
            for key in keys:
                if key in sample:
                    image = np.array(sample[key].resize((256, 256))).astype(np.float32)
                    image = image.transpose(2, 0, 1) / 255.0
                    sample["image"] = torch.from_numpy(image)
            return sample

        dataset = (
            wds.WebDataset(path)
            .decode("pil")
            .map(resize_image)
            .to_tuple("image", "__key__")
        )
        return dataset


def process_chunk(
    rank,
    world_size,
    tokenizer_cfg_path,
    transform_cfg_path,
    paths,
    output_dir,
    num_workers,
    batch_size,
    s3,
    dataset_type,
):
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=rank, load_diffusion=False)

    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    num_paths_per_chunk = int(np.ceil(len(paths) / world_size))

    worker_paths = paths[
        rank * num_paths_per_chunk : min(len(paths), (rank + 1) * num_paths_per_chunk)
    ]
    print (f"Rank: {rank} processing {len(worker_paths)} shards")
    for path in worker_paths:
        basename = os.path.basename(path)
        output_path = os.path.join(
            output_dir, os.path.splitext(basename)[0] + ".parquet"
        )

        try:
            dataset = get_dataset(dataset_type, path, s3)
            dataloader = torch.utils.data.DataLoader(
                dataset, #.batched(batch_size),
                batch_size=batch_size,
                pin_memory=True,
                num_workers=num_workers,
            )
            rows = []
            embeddings = []
            for data, metas in tqdm(
                dataloader,
                total=int(np.ceil(EXPECTED_CHUNK_SIZE / batch_size)),
                desc=f"Rank : {rank}, Shard: {basename}",
                position=rank,
                leave=False,
            ):
                image_tensor = transform(data).to(rank)
                image_ids = tokenizer.encode_image(image_torch=image_tensor)

                rows.extend(metas)
                embeddings.append(image_ids)
            embeddings = torch.cat(embeddings, axis=0)

            df = pd.DataFrame(rows)
            embeddings_cpu = embeddings.cpu().numpy().reshape(len(df), -1)
            df["code"] = [item.tobytes() for item in embeddings_cpu]
            df.to_parquet(output_path, compression="brotli")
        except Exception:
            print(f"[-] Failed to process {basename}:", file=sys.stderr)
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        help="/path/to/images/{0000..1111}.tar",
    )
    parser.add_argument(
        "-s3",
        action="store_true",
        help="Pass this flag if using s3 bucket",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for *.parquet files with the code column",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        help="Number of workers per gpu for the dataloader",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per gpu for the dataloader",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="laion",
        help="Type of dataset used. Can be 'laion' or 'mmc4'",
    )
    args = parser.parse_args()


    pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

    tokenizer_cfg_path = '../configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    transform_cfg_path = '../configs/transform/clip_transform.yaml'

    paths = list(braceexpand.braceexpand(args.paths))

    start = timer()

    if args.dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"Dataset must be one of {ALLOWED_DATASETS}, got {args.dataset}"
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_gpus > 1:
        mp.spawn(
            process_chunk,
            args=(
                args.num_gpus,
                tokenizer_cfg_path,
                transform_cfg_path,
                paths,
                args.output_dir,
                args.num_workers,
                args.batch_size,
                args.s3,
                args.dataset,
            ),
            nprocs=args.num_gpus,
        )
    else:
        process_chunk(0, 1, tokenizer_cfg_path, transform_cfg_path, paths, args.output_dir, args.num_workers, args.batch_size, args.s3, args.dataset)

    print(f"Processing {len(paths)} shards took {timer() - start} seconds")


if __name__ == "__main__":
    main()
