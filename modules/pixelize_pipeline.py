import modules.async_worker as worker
from PIL import Image
import numpy as np
import torch

class pipeline:
    pipeline_type = ["pixelize"]

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # No model needed for pixelize
        return

    def load_keywords(self, lora):
        return ""

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Pixelizing ...", None)
        )

        input_image = gen_data["input_image"]
        if input_image is None:
            print("ERROR: No input image provided for pixelize")
            return ["html/error.png"]

        # Get parameters from gen_data (from controlnet settings)
        import modules.controlnet as controlnet
        cn_settings = controlnet.get_settings(gen_data)
        
        downsample_factor = int(cn_settings.get("downsample_factor", 4))
        upsample = cn_settings.get("upsample", True)

        # Convert to PIL if needed
        if not isinstance(input_image, Image.Image):
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().numpy()
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray((input_image * 255).astype(np.uint8))

        # Get original dimensions
        width, height = input_image.size

        # Convert to RGB if needed
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Downsample with BOX filter (blocky pixelization)
        downsampled_width = max(1, width // downsample_factor)
        downsampled_height = max(1, height // downsample_factor)
        
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Downsampling to {downsampled_width}x{downsampled_height} ...", None)
        )
        
        pixelized_image = input_image.resize(
            (downsampled_width, downsampled_height), 
            Image.BOX
        )

        # Upsample back to original size with NEAREST (keeps pixels blocky)
        if upsample:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Upsampling back to {width}x{height} ...", None)
            )
            pixelized_image = pixelized_image.resize(
                (width, height), 
                Image.NEAREST
            )

        return [pixelized_image]
