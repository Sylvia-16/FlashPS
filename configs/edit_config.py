import json
import os

import yaml


class EditConfig:
    def __init__(self, config_file):

        # launch script
        self.launch_script = config_file["launch_script"]

        # general config
        self.num_inference_steps = config_file["num_inference_steps"]

        # edit related config
        self.mask_filepath = config_file.get("mask_filepath", "")
        self.use_cached_o = config_file.get("use_cached_o", False)
        self.use_cached_kv = config_file.get("use_cached_kv", False)

        self.save_o = config_file.get("save_o", False)
        self.save_kv = config_file.get("save_kv", False)
        self.save_latents = config_file.get("save_latents", False)
        self.cached_o_folder = config_file.get("cached_o_folder", "")
        self.cached_kv_folder = config_file.get("cached_kv_folder", "")
        self.cached_latents_folder = config_file.get("cached_latents_folder", "")
        self.use_cached_ff = config_file.get("use_cached_ff", False)
        self.save_ff = config_file.get("save_ff", False)
        self.cached_ff_folder = config_file.get("cached_ff_folder", "")
        # prompt related
        self.prompt = config_file.get("prompt", "")
        if self.prompt == "":
            print("=== [Warning] Prompt is empty ===")
        self.prompt_3 = config_file.get("prompt_3", "")

        # runtime configs, which are set in runtime
        self.block_name = "unknown"

        # a list of masks, each for one denoising step
        self.masks = []
        self.cached_o = {}
        self.cached_kv = {}
        self.cached_latents = {}
        self.cached_ff = {}

        self.mask = None  # for FluxAttnProcessor2_0
        self.async_copy = config_file.get("async_copy", False)
        self.load_stream = None
        self.compute_stream = None
        # load mask and image
        self.mask_path = config_file.get("mask_path", "")
        self.image_path = config_file.get("image_path", "")

        self.profile = config_file.get("profile", False)
        self.profile_path = config_file.get("profile_path", "")

        self.use_flash_attn_rope = config_file.get("use_flash_attn_rope", True)
        self.use_test_batch = config_file.get("use_test_batch", False)
        self.batch_size = config_file.get("batch_size", 1)
        # self.test_batch_image_size = config_file.get("test_batch_image_size", 1024)
        # self.test_batch_mask_height = config_file.get("test_batch_mask_height",1024)
        # self.test_varlen = config_file.get("test_varlen",False)
        self.device_num = config_file.get("device_num", 0)
        self.device = None
        self.test_varlen = config_file.get("test_varlen", False)
        self.real_varlen = config_file.get("real_varlen", False)
        self.compare_diff = config_file.get("compare_diff", False)
        self.latents_repeats = None
        self.text_repeats = None
        self.repeats = None
        self.indices = None
        self.latents_indices = None
        self.text_indices = None
        self.test_rope = config_file.get("test_rope", False)
        self.test_seqlen = config_file.get("test_seqlen", False)
        self.generated_seqlen = config_file.get("generated_seqlen", None)

        self.model_path = config_file.get("model_path", "")
        self.cloth_path = config_file.get("cloth_path", "")
        self.image_scale = config_file.get("image_scale", 2.0)
        self.n_samples = config_file.get("n_samples", 4)
        self.seed = config_file.get("seed", 1449)
        self.category = config_file.get("category", 0)
        self.model_type = config_file.get(
            "model_type", "hd"
        )  # hd or dc hd: half-body, dc: full-body
        self.max_batch_size = config_file.get("max_batch_size", 1)
        self.masked_vton_img_path = config_file.get("masked_vton_img_path", "")
        self.ootd_mask_path = config_file.get("ootd_mask_path", "")
        self.save_image_path = config_file.get("save_image_path", "")
        self.mask_pt_path = config_file.get("mask_pt_path", "")
        self.random_seed_path = config_file.get("random_seed_path", "")
    @classmethod
    def from_file(cls, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

if __name__ == "__main__":

    # check all configs are valid
    config_files = [item for item in os.listdir("./") if item.endswith(".yml")]
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict), "Config load failed"
        edit_config = EditConfig(config)
        print("Config loaded", config_file)
        print(json.dumps(edit_config.__dict__, indent=4))
        print("===")
