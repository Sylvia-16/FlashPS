from pathlib import Path
import sys
import torch
from PIL import Image
import sys
sys.path.append('/app/')
from ootd.run.utils_ootd import get_mask_location
import time

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from ootd.preprocess.openpose.run_openpose import OpenPose
from ootd.preprocess.humanparsing.run_parsing import Parsing
from ootd.ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.ootd.inference_ootd_dc import OOTDiffusionDC
from torch.profiler import profile, record_function, ProfilerActivity
import yaml
from configs.edit_config import EditConfig
import json
import argparse
import os


def load_cache_from_one_folder(edit_config, cached_o_folder, cached_o_files):
    for file in cached_o_files:

        tmp_key = file.split(".")[0]
        if tmp_key not in edit_config.cached_o or edit_config.cached_o[tmp_key] is None:
            edit_config.cached_o[tmp_key] = []
        if edit_config.async_copy:
            # if async_copy, copy to cpu first
            edit_config.cached_o[tmp_key].append(
                torch.load(
                    os.path.join(cached_o_folder, file),
                    map_location=torch.device("cpu"),
                ).contiguous().pin_memory()
            )
        else:
            # if not async_copy, copy to gpu directly
            edit_config.cached_o[tmp_key].append(
                torch.load(os.path.join(cached_o_folder, file))
            )


def load_cache_o(edit_config):
    assert (
        edit_config.save_o is False
    ), f"save_o: {edit_config.save_o}; use_cached_o: {edit_config.use_cached_o}"
    assert (
        edit_config.cached_o_folder != ""
    ), "cached_o_folder must be provided if use_cached_o is True"
    edit_config.cached_o = {}

    if isinstance(edit_config.cached_o_folder, list):
        for folder in edit_config.cached_o_folder:
            cached_o_files = [
                item for item in os.listdir(folder) if item.endswith(".pt")
            ]
            load_cache_from_one_folder(edit_config, folder, cached_o_files)
    else:
        cached_o_files = [
            item
            for item in os.listdir(edit_config.cached_o_folder)
            if item.endswith(".pt")
        ]
        load_cache_from_one_folder(
            edit_config, edit_config.cached_o_folder, cached_o_files
        )


def get_cloth_model_mask(edit_config):
    if edit_config.batch_size > 1:
        edit_config.cloth_path = [edit_config.cloth_path for _ in range(edit_config.batch_size)]
        edit_config.model_path = [edit_config.model_path for _ in range(edit_config.batch_size)]
        edit_config.masked_vton_img_path = [edit_config.masked_vton_img_path for _ in range(edit_config.batch_size)]
        edit_config.ootd_mask_path = [edit_config.ootd_mask_path for _ in range(edit_config.batch_size)]
    if isinstance(edit_config.cloth_path, list):
        cloth_img = [
            Image.open(path).resize((768, 1024)) for path in edit_config.cloth_path
        ]
    else:
        cloth_img = Image.open(edit_config.cloth_path).resize((768, 1024))
    if isinstance(edit_config.model_path, list):
        model_img = [
            Image.open(path).resize((768, 1024)) for path in edit_config.model_path
        ]
    else:
        model_img = Image.open(edit_config.model_path).resize((768, 1024))
    if isinstance(edit_config.model_path, list):

        masked_vton_img = []
        mask_list = []
        if edit_config.masked_vton_img_path is not None and edit_config.masked_vton_img_path != "":
                if isinstance(edit_config.masked_vton_img_path, list):
                    masked_vton_img = [Image.open(path).resize((768, 1024)) for path in edit_config.masked_vton_img_path]
                else:
                    masked_vton_img = Image.open(edit_config.masked_vton_img_path).resize((768, 1024))

                if isinstance(edit_config.ootd_mask_path, list):
                    mask_list = [Image.open(path).resize((768, 1024)) for path in edit_config.ootd_mask_path]
                else:
                    mask_list = Image.open(edit_config.ootd_mask_path).resize((768, 1024))
        else:
            for i in range(len(model_img)):
                keypoints = openpose_model(model_img[i].resize((384, 512)))
                model_parse, _ = parsing_model(model_img[i].resize((384, 512)))
                mask, mask_gray = get_mask_location(
                    model_type, category_dict_utils[category], model_parse, keypoints
                )
                mask = mask.resize((768, 1024), Image.NEAREST)
                mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
                mask_list.append(mask)
                masked_vton_img.append(Image.composite(mask_gray, model_img[i], mask))
        mask = mask_list
    else:
        if edit_config.masked_vton_img_path is not None and edit_config.masked_vton_img_path != "":
                masked_vton_img = Image.open(edit_config.masked_vton_img_path).resize((768, 1024))
                mask = Image.open(edit_config.ootd_mask_path).resize((768, 1024))
        else:
            keypoints = openpose_model(model_img.resize((384, 512)))
            model_parse, _ = parsing_model(model_img.resize((384, 512)))

            mask, mask_gray = get_mask_location(
                model_type, category_dict_utils[category], model_parse, keypoints
            )
            mask = mask.resize((768, 1024), Image.NEAREST)
            mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

            masked_vton_img = Image.composite(mask_gray, model_img, mask)
    # save Image
    return model_img, cloth_img, masked_vton_img, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_config_path", type=str, required=True)
    parser.add_argument("--log_folder", type=str, default="./")
    args = parser.parse_args()
    # load config file
    with open(args.edit_config_path, "r") as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict), "Config load failed"
    edit_config = EditConfig(config)

    openpose_model = OpenPose(edit_config.device_num)
    parsing_model = Parsing(edit_config.device_num)

    category_dict = ["upperbody", "lowerbody", "dress"]
    category_dict_utils = ["upper_body", "lower_body", "dresses"]

    model_type = edit_config.model_type  # "hd" or "dc"
    category = edit_config.category  # 0:upperbody; 1:lowerbody; 2:dress
    cloth_path = edit_config.cloth_path
    model_path = edit_config.model_path

    image_scale = edit_config.image_scale
    n_steps = edit_config.num_inference_steps
    n_samples = edit_config.n_samples
    seed = edit_config.seed

    if model_type == "hd":
        model = OOTDiffusionHD(edit_config.device_num)
    elif model_type == "dc":
        model = OOTDiffusionDC(edit_config.device_num)
    else:
        raise ValueError("model_type must be 'hd' or 'dc'!")
    print("Config loaded")
    print("Edit Config", json.dumps(edit_config.__dict__, indent=4))

    # cloth_img = [cloth_img for _ in range(4)]
    t0 = time.time()
    model_img, cloth_img, masked_vton_img, mask = get_cloth_model_mask(edit_config)
    t1 = time.time()

    if edit_config.use_cached_o:
        load_cache_o(edit_config)
    if edit_config.async_copy:
        edit_config.load_stream = torch.cuda.Stream(edit_config.device_num)
        edit_config.compute_stream = torch.cuda.Stream(edit_config.device_num)
    edit_config.batch_size = edit_config.batch_size * 2 # cfg

    if edit_config.profile:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        images = model(
            model_type=model_type,
            category=category_dict[category],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
            edit_config=edit_config,
        )
        with profile(
            activities=activities,
            with_stack=True,
        ) as prof:
            images = model(
                model_type=model_type,
                category=category_dict[category],
                image_garm=cloth_img,
                image_vton=masked_vton_img,
                mask=mask,
                num_steps=5,
                image_ori=model_img,
                num_samples=n_samples,
                image_scale=image_scale,
                seed=seed,
                edit_config=edit_config,
            )
        prof.export_chrome_trace(args.log_folder + "/trace.json")
        exit(0)
    else:
        # warm up
        images = model(
            model_type=model_type,
            category=category_dict[category],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
            edit_config=edit_config,
        )
    total_time = 0
    for i in range(5):
        t0 = time.time()
        torch.cuda.synchronize()
        images = model(
            model_type=model_type,
            category=category_dict[category],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
            edit_config=edit_config,
        )
        torch.cuda.synchronize()
        t1 = time.time()
        total_time += (t1 - t0)
    print(f"time: {total_time/5}")
    image_idx = 0
    for image in images:
        if edit_config.save_image_path != "":
            image.save(edit_config.save_image_path)
        else:   
            image.save(
                args.log_folder + "/out_" + model_type + "_" + str(image_idx) + ".png"
            )
            image_idx += 1
