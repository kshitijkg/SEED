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

from webdataset import ShardWriter

from tqdm import tqdm
from timeit import default_timer as timer


warnings.filterwarnings("ignore", category=UserWarning)
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

ALLOWED_DATASETS = ["laion", "mmc4"]

EXPECTED_CHUNK_SIZE = 10000

from torchvision import transforms

def transform_and_remove_keys(sample):
    image, metadata = sample

    # CLIP transform without resizing
    image = transforms.functional.resize(image, (224, 224))
    image = transforms.functional.normalize(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    
    new_dictionary = {}
    new_dictionary['key'] = metadata['key']
    new_dictionary['caption'] = metadata['caption']
    new_dictionary['uid'] = metadata['uid']
    return image, new_dictionary

def get_dataset(dataset_type, path, s3):
    if s3:
        path = f"pipe:aws s3 cp {path} -"

    if dataset_type == "laion":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg;png;webp", "json")
        )
        dataset = dataset.map(transform_and_remove_keys)

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
    write_freq = 5
    max_items_per_shard = batch_size * write_freq

    dataset = get_dataset(dataset_type, paths, s3)
    sink = ShardWriter(os.path.join(output_dir, f"%05d.tar"), maxcount=max_items_per_shard)
    dataloader = torch.utils.data.DataLoader(
        dataset, #.batched(batch_size),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    num_chunks = len(list(braceexpand.braceexpand(paths)))
    rows = {}
    embeddings = []
    write_count = 0
    for data, metas in tqdm(dataloader, total=np.ceil((EXPECTED_CHUNK_SIZE * num_chunks) / batch_size), desc=f"Rank : {rank}", position=rank, leave=False):
        image_tensor = data.to(rank)
        image_ids = tokenizer.encode_image(image_torch=image_tensor)
        # metas["seed_tokens"] = image_ids
        
        # for i in range(len(image_ids)):
        #     sample = {}
        #     for key, val in metas.items():
        #         sample[key] = val[i]
        #     sink.write(sample)


        if len(rows.keys()) == 0:
            for k, v in metas.items():
                if type(v) == torch.Tensor:
                    v = v.cpu().numpy().tolist()
                rows[k] = v
        else:
            for k, v in metas.items():
                if type(v) == torch.Tensor:
                    v = v.cpu().numpy().tolist()
                rows[k].extend(v)

        embeddings.append(image_ids)
        
        if (write_count + 1) % write_freq == 0:
            rows['embeddings'] = embeddings

            for i in range(len(rows['embeddings'])):
                sample = {}
                for key, val in rows.items():
                    sample[key] = val[i]
                sink.write(sample)

            rows = {}
            embeddings = []

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

    # paths = list(braceexpand.braceexpand(args.paths))
    paths = args.paths

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
