import os
import shutil
import json
from os.path import exists
from shared import state
from shared import path_manager

DEFAULT_CNSETTINGS_FILE = "settings/powerup.default"
CNSETTINGS_FILE = "settings/powerup.json"
NEWCN = "Custom..."

# https://huggingface.co/stabilityai/control-lora/tree/main/control-LoRAs-rank128
controlnet_models = {
    "canny": "control-lora-canny-rank128.safetensors",
    "depth": "control-lora-depth-rank128.safetensors",
    "recolour": "control-lora-recolor-rank128.safetensors",
    "sketch": "control-lora-sketch-rank128-metadata.safetensors",
    "img2img": None,
    "upscale": None,
    "rembg": None,
    "faceswap": None,
    "pixelize": None,
    "palettize": None,
    "quantize": None,
    "kcentroid": None,
    "bitize": None,
}


def load_cnsettings():
    settings = {}
    if not os.path.isfile(CNSETTINGS_FILE):
        shutil.copy(DEFAULT_CNSETTINGS_FILE, CNSETTINGS_FILE)

    if os.path.exists(CNSETTINGS_FILE):
        with open(CNSETTINGS_FILE) as f:
            settings.update(json.load(f))

    with open(DEFAULT_CNSETTINGS_FILE) as f:
        default_settings = json.load(f)

    # Update settings with any missing keys from default_settings
    settings_updated = False
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
            settings_updated = True

    # If settings were updated, write them back to the file
    if settings_updated:
        with open(CNSETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)

    # Ignore unknown settings
    for key in list(settings.keys()):
        if settings[key]["type"] not in controlnet_models:
            del settings[key]

    return settings


def save_cnsettings(cn_save_options):
    global controlnet_settings, cn_options
    with open(CNSETTINGS_FILE, "w") as f:
        json.dump(cn_save_options, f, indent=2)
    controlnet_settings = cn_save_options
    cn_options = {f"{k}": v for k, v in controlnet_settings.items()}


def modes():
    return controlnet_settings.keys()


def get_model(type):
    #return controlnet_models[type] if type in controlnet_models else None
    return path_manager.get_file_path(f"cn_{type}", default=None)


def get_settings(gen_data):
    if "cn_selection" not in gen_data:
        return {}
    if gen_data["cn_selection"] == NEWCN:
        return {
            "type": gen_data["cn_type"].lower(),
            "edge_low": gen_data["cn_edge_low"],
            "edge_high": gen_data["cn_edge_high"],
            "start": gen_data["cn_start"],
            "stop": gen_data["cn_stop"],
            "strength": gen_data["cn_strength"],
            "upscaler": gen_data["cn_upscale"],
            "downsample_factor": gen_data.get("cn_downsample_factor", 4),
            "upsample": gen_data.get("cn_upsample", True),
            "palette_file": gen_data.get("cn_palette_file", "None"),
            "extract_palette": gen_data.get("cn_extract_palette", False),
            "palette_name": gen_data.get("cn_palette_name", ""),
            "palette_colors": gen_data.get("cn_palette_colors", 256),
            "dither": gen_data.get("cn_dither", False),
            "prequantize": gen_data.get("cn_prequantize", False),
            "prequantize_colors": gen_data.get("cn_prequantize_colors", 256),
            "quantize_mode": gen_data.get("cn_quantize_mode", "fast_octree"),
            "colors": gen_data.get("cn_quantize_colors", 64),
            "method": gen_data.get("cn_quantize_method", "Median Cut"),
            "kmeans": gen_data.get("cn_quantize_kmeans", 0),
            "downscale_factor": gen_data.get("cn_kcentroid_downscale", 8),
            "color_clusters": gen_data.get("cn_kcentroid_clusters", 24),
            "dither_strength": gen_data.get("cn_kcentroid_dither_strength", 0),
        }
    else:
        return (
            controlnet_settings[gen_data["cn_selection"]]
            if gen_data["cn_selection"] in controlnet_settings
            else {}
        )


controlnet_settings = load_cnsettings()
cn_options = {f"{k}": v for k, v in controlnet_settings.items()}
