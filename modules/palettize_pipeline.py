import modules.async_worker as worker
from PIL import Image
import numpy as np
import torch
from pathlib import Path
from shared import path_manager

class pipeline:
    pipeline_type = ["palettize"]

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # No model needed for palettize
        return

    def load_keywords(self, lora):
        return ""

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def get_palette(self, image, max_colors=256):
        """
        Extract unique colors from an image and return as palette image.
        Based on InvokeAI retro_helpers.py
        
        Args:
            image: PIL Image in RGB mode
            max_colors: Maximum number of colors to extract (2-256)
            
        Returns:
            PIL Image in P (indexed) mode containing unique colors
        """
        # Clamp max_colors to valid range
        max_colors = max(2, min(256, int(max_colors)))
        
        # Get palette from the image with specified color limit
        palette = image.convert("P", palette=Image.ADAPTIVE, colors=max_colors).getpalette()

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

    def save_palette(self, palette_image, name, task_id=None):
        """
        Save a palette image to the palettes folder.
        
        Args:
            palette_image: PIL Image in P mode
            name: Filename for the palette (will add .png if missing)
            task_id: Optional task ID for worker notifications
            
        Returns:
            Path to saved palette file or None if already exists
        """
        palette_path = Path(path_manager.model_paths["palette_path"])
        palette_path.mkdir(parents=True, exist_ok=True)

        # Ensure .png extension
        if not name.lower().endswith('.png'):
            name = name + '.png'

        save_path = palette_path / name

        # Check if file already exists
        if save_path.exists():
            if task_id:
                worker.add_result(
                    task_id,
                    "preview",
                    (-1, f"Palette '{name}' already exists, using existing file", None)
                )
            print(f"Palette '{name}' already exists, skipping save")
            return None

        # Save the palette
        palette_image.save(save_path)
        
        if task_id:
            worker.add_result(
                task_id,
                "preview",
                (-1, f"Saved palette to '{name}'", None)
            )
        print(f"Saved palette to '{save_path}'")
        
        # Refresh palette list
        path_manager.palette_filenames = path_manager.get_palette_filenames(
            path_manager.model_paths["palette_path"]
        )
        
        return save_path

    def palettize(self, image, palette_image, prequantize, prequantize_colors, method, dither):
        """
        Palettize an image using a palette image.
        
        Args:
            image: PIL Image to palettize (RGB)
            palette_image: PIL Image to use as palette (should be in P mode)
            prequantize: Apply quantization first before palettizing
            prequantize_colors: Number of colors to use for prequantization (2-256)
            method: PIL quantize method (0=MEDIANCUT, 1=MAXCOVERAGE, 2=FASTOCTREE)
            dither: Apply Floyd-Steinberg dithering
        """
        palettized = image

        if prequantize:
            # Clamp prequantize_colors to valid range
            prequantize_colors = max(2, min(256, int(prequantize_colors)))
            palettized = palettized.quantize(
                colors=prequantize_colors, 
                method=method, 
                dither=Image.Dither.NONE
            ).convert('RGB')

        # If palette is not in indexed mode, we need to extract its palette first
        if palette_image.mode != 'P':
            # Extract palette from RGB image
            palette_image = self.get_palette(palette_image)

        return palettized.quantize(
            palette=palette_image, 
            method=method, 
            dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
        )

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Palettizing ...", None)
        )

        input_image = gen_data["input_image"]
        if input_image is None:
            print("ERROR: No input image provided for palettize")
            return ["html/error.png"]

        # Get parameters from gen_data (from controlnet settings)
        import modules.controlnet as controlnet
        cn_settings = controlnet.get_settings(gen_data)
        
        palette_file = cn_settings.get("palette_file", "None")
        extract_palette = cn_settings.get("extract_palette", False)
        palette_name = cn_settings.get("palette_name", "")
        palette_colors = cn_settings.get("palette_colors", 256)
        dither = cn_settings.get("dither", False)
        prequantize = cn_settings.get("prequantize", False)
        prequantize_colors = cn_settings.get("prequantize_colors", 256)
        quantize_mode = cn_settings.get("quantize_mode", "fast_octree")
        
        # Map quantize mode string to PIL constant
        quantize_map = {
            "median_cut": Image.Quantize.MEDIANCUT,
            "max_coverage": Image.Quantize.MAXCOVERAGE,
            "fast_octree": Image.Quantize.FASTOCTREE
        }
        method = quantize_map.get(quantize_mode.lower(), Image.Quantize.FASTOCTREE)

        # Convert to PIL if needed
        if not isinstance(input_image, Image.Image):
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().numpy()
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray((input_image * 255).astype(np.uint8))

        # Convert to RGB if needed
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Handle palette extraction if requested
        if extract_palette:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Extracting {palette_colors} colors from input image ...", None)
            )
            
            extracted_palette = self.get_palette(input_image, palette_colors)
            
            # Use provided name or default to "palette"
            save_name = palette_name if palette_name else "palette"
            self.save_palette(extracted_palette, save_name, gen_data["task_id"])

        # Determine which palette to use
        palette_image = None
        
        if palette_file and palette_file != "None":
            # Load palette from file
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Loading palette '{palette_file}' ...", None)
            )
            
            palette_path = Path(path_manager.model_paths["palette_path"]) / palette_file
            if palette_path.exists():
                palette_image = Image.open(palette_path)
                # Keep palette in P mode to preserve the exact color palette
                if palette_image.mode != "P":
                    # If not already indexed, convert to RGB first for processing
                    if palette_image.mode != "RGB":
                        palette_image = palette_image.convert("RGB")
            else:
                print(f"WARNING: Palette file '{palette_file}' not found, using input image")
                palette_image = input_image
        else:
            # Use input image as palette
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Using input image as palette ...", None)
            )
            palette_image = input_image

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Applying palette (mode: {quantize_mode}, dither: {dither}) ...", None)
        )

        # Palettize the image
        palettized_image = self.palettize(
            input_image, 
            palette_image, 
            prequantize, 
            prequantize_colors,
            method, 
            dither
        )

        # Convert back to RGB for output
        palettized_image = palettized_image.convert('RGB')

        return [palettized_image]
