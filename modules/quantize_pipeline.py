import modules.async_worker as worker
from PIL import Image
import numpy as np
import torch

class pipeline:
    pipeline_type = ["quantize"]

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # No model needed for quantize
        return

    def load_keywords(self, lora):
        return ""

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def get_palette(self, image):
        """
        Extract unique colors from an image and return as palette image.
        Based on InvokeAI retro_helpers.py
        
        Args:
            image: PIL Image in RGB mode
            
        Returns:
            PIL Image in P (indexed) mode containing unique colors
        """
        # Get palette from the image
        palette = image.convert("P", palette=Image.ADAPTIVE, colors=256).getpalette()

        # Create a set to store unique colors
        unique_colors = set()
        new_palette = []

        for i in range(0, len(palette), 3):
            r = palette[i]
            g = palette[i+1]
            b = palette[i+2]
            color = (r, g, b)

            # Add unique colors to set
            if color not in unique_colors:
                unique_colors.add(color)
                new_palette.extend(color)

        num_colors = len(new_palette) // 3

        # Create 1-pixel high image with all unique colors
        palette_image = Image.new("P", (num_colors, 1))

        for i in range(num_colors):
            r = new_palette[i*3]
            g = new_palette[i*3+1]
            b = new_palette[i*3+2]
            palette_image.putpixel((i, 0), (r, g, b))

        return palette_image

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Quantizing image ...", None)
        )

        input_image = gen_data["input_image"]
        if input_image is None:
            print("ERROR: No input image provided for quantize")
            return ["html/error.png"]

        # Get parameters from gen_data (from controlnet settings)
        import modules.controlnet as controlnet
        cn_settings = controlnet.get_settings(gen_data)
        
        colors = int(cn_settings.get("colors", 64))
        method_name = cn_settings.get("method", "Median Cut")
        kmeans = int(cn_settings.get("kmeans", 0))
        dither = cn_settings.get("dither", True)

        # Map method names to PIL constants
        method_map = {
            "Median Cut": Image.Quantize.MEDIANCUT,
            "Max Coverage": Image.Quantize.MAXCOVERAGE,
            "Fast Octree": Image.Quantize.FASTOCTREE
        }
        method = method_map.get(method_name, Image.Quantize.MEDIANCUT)

        # Clamp colors to valid range (1-256)
        colors = max(1, min(256, colors))

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
            (-1, f"Quantizing to {colors} colors using {method_name} ...", None)
        )

        # Quantize the image based on InvokeAI retro_quantize.py logic
        if dither:
            # Get palette from pre-quantized image for better dithering
            palette = self.get_palette(
                input_image.quantize(colors, method=method).convert('RGB')
            )
            quantized_image = input_image.quantize(
                colors=colors,
                palette=palette,
                method=method,
                kmeans=kmeans,
                dither=Image.Dither.FLOYDSTEINBERG
            ).convert('RGB')
        else:
            # Quantize without dithering
            quantized_image = input_image.quantize(
                colors=colors,
                method=method,
                kmeans=kmeans,
                dither=Image.Dither.NONE
            ).convert('RGB')

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Quantization complete", None)
        )

        return [quantized_image]
