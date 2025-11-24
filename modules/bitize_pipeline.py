import modules.async_worker as worker
from PIL import Image
import numpy as np
import torch

class pipeline:
    pipeline_type = ["bitize"]

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # No model needed for bitize
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
            (-1, f"Bitizing to 1-bit ...", None)
        )

        input_image = gen_data["input_image"]
        if input_image is None:
            print("ERROR: No input image provided for bitize")
            return ["html/error.png"]

        # Get parameters from gen_data (from controlnet settings)
        import modules.controlnet as controlnet
        cn_settings = controlnet.get_settings(gen_data)
        
        dither = cn_settings.get("dither", True)

        # Convert to PIL if needed
        if not isinstance(input_image, Image.Image):
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().numpy()
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray((input_image * 255).astype(np.uint8))

        # Convert to RGB if needed
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Converting to 1-bit with {'dithering' if dither else 'no dithering'} ...", None)
        )

        # Convert to 1-bit (black and white) with optional dithering
        dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
        bitized_image = input_image.convert("1", dither=dither_mode).convert("RGB")

        return [bitized_image]
