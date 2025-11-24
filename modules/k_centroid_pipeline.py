import modules.async_worker as worker
from PIL import Image
import numpy as np
import torch
from pathlib import Path
from shared import path_manager
from itertools import product

try:
    import hitherdither
    HITHERDITHER_AVAILABLE = True
except ImportError:
    HITHERDITHER_AVAILABLE = False
    print("WARNING: hitherdither not available - dithering disabled for k-centroid")

class pipeline:
    pipeline_type = ["kcentroid"]

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # No model needed for k-centroid
        return

    def load_keywords(self, lora):
        return ""

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def adjust_gamma(self, image, gamma=1.0):
        """
        Apply gamma correction to an image.
        Based on palettize.py implementation.
        
        Args:
            image: PIL Image
            gamma: Gamma value (1.0 = no change, <1 = darker, >1 = brighter)
            
        Returns:
            Gamma-corrected PIL Image
        """
        # Create a lookup table for the gamma function
        gamma_map = [255 * ((i / 255.0) ** (1.0 / gamma)) for i in range(256)]
        gamma_table = bytes([(int(x / 255.0 * 65535.0) >> 8) for x in gamma_map] * 3)

        # Apply the gamma correction using the lookup table
        return image.point(gamma_table)

    def kCentroid(self, image, width, height, centroids):
        """
        K-Centroid downscaling algorithm.
        Divides image into tiles and quantizes each tile to dominant colors.
        Based on palettize.py implementation.
        
        Args:
            image: PIL Image in RGB mode
            width: Target width in pixels
            height: Target height in pixels
            centroids: Number of color clusters per tile (k-means k value)
            
        Returns:
            PIL Image downscaled with k-centroid algorithm
        """
        image = image.convert("RGB")
        downscaled = np.zeros((height, width, 3), dtype=np.uint8)
        wFactor = image.width / width
        hFactor = image.height / height
        
        for x, y in product(range(width), range(height)):
            # Extract tile from original image
            tile = image.crop((
                x * wFactor, 
                y * hFactor, 
                (x * wFactor) + wFactor, 
                (y * hFactor) + hFactor
            ))
            
            # Quantize tile to centroids colors using k-means
            tile_quantized = tile.quantize(
                colors=centroids, 
                method=1,  # MAXCOVERAGE 
                kmeans=centroids
            ).convert("RGB")
            
            # Find most common color in quantized tile
            color_counts = tile_quantized.getcolors()
            most_common_color = max(color_counts, key=lambda x: x[0])[1]
            downscaled[y, x, :] = most_common_color
            
        return Image.fromarray(downscaled, mode='RGB')

    def apply_palette_quantization(self, image, palette_image, method=1):
        """
        Quantize image to use only colors from the palette.
        
        Args:
            image: PIL Image to quantize
            palette_image: PIL Image containing the palette colors
            method: PIL quantize method (0=MEDIANCUT, 1=MAXCOVERAGE, 2=FASTOCTREE)
            
        Returns:
            PIL Image quantized to palette colors
        """
        if palette_image is None:
            return image
            
        # Convert palette to indexed mode if needed
        if palette_image.mode != 'P':
            # Extract unique colors from palette image
            if palette_image.mode != 'RGB':
                palette_image = palette_image.convert('RGB')
            
            # Get all unique colors from the palette
            colors = palette_image.getcolors(16777216)  # Max possible colors
            if not colors:
                return image
                
            # Create a simple indexed palette image
            num_colors = len(colors)
            palette_img = Image.new('P', (num_colors, 1))
            pal_data = []
            for count, color in colors:
                pal_data.extend(color)
            # Pad palette to 256 colors
            pal_data.extend([0] * (768 - len(pal_data)))
            palette_img.putpalette(pal_data)
            palette_image = palette_img
        
        # Quantize the image to the palette
        quantized = image.quantize(palette=palette_image, method=method, dither=Image.Dither.NONE)
        return quantized.convert('RGB')

    def apply_palette_dithering(self, image, palette_image, dither_strength, dither_order):
        """
        Apply Bayer dithering with a palette.
        Based on palettize.py implementation.
        
        Args:
            image: PIL Image to dither
            palette_image: PIL Image or list of RGB tuples to use as palette
            dither_strength: Strength value (0-10)
            dither_order: Bayer matrix order (2, 4, or 8)
            
        Returns:
            Dithered PIL Image
        """
        if not HITHERDITHER_AVAILABLE:
            return image
            
        # Apply gamma adjustment for better dithering results
        gamma_adjusted = self.adjust_gamma(image, 1.0 - (0.02 * dither_strength))
        
        # Build palette from palette image or list
        palette = []
        if isinstance(palette_image, Image.Image):
            # Convert palette image to RGB if needed
            if palette_image.mode != "RGB":
                palette_image = palette_image.convert("RGB")
            # Extract colors
            colors = palette_image.getcolors(16777216)  # Max possible colors
            if colors:
                for count, color in colors:
                    palette.append(color)
        elif isinstance(palette_image, list):
            palette = palette_image
        else:
            # No palette, extract from image itself
            image_indexed = image.quantize(colors=256, method=1, kmeans=0, dither=0).convert('RGB')
            colors = image_indexed.getcolors(16777216)
            if colors:
                for count, color in colors:
                    palette.append(color)
        
        if not palette:
            return image
            
        # Create hitherdither palette
        hd_palette = hitherdither.palette.Palette(palette)
        
        # Calculate threshold based on strength
        threshold = (16 * dither_strength) / 4
        
        # Apply Bayer dithering
        dithered = hitherdither.ordered.bayer.bayer_dithering(
            gamma_adjusted, 
            hd_palette, 
            [threshold, threshold, threshold], 
            order=dither_order
        ).convert('RGB')
        
        return dithered

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"K-Centroid processing ...", None)
        )

        input_image = gen_data["input_image"]
        if input_image is None:
            print("ERROR: No input image provided for k-centroid")
            return ["html/error.png"]

        # Get parameters from gen_data (from controlnet settings)
        import modules.controlnet as controlnet
        cn_settings = controlnet.get_settings(gen_data)
        
        downscale_factor = int(cn_settings.get("downscale_factor", 8))
        color_clusters = int(cn_settings.get("color_clusters", 24))
        dither = cn_settings.get("dither", False)
        dither_strength = int(cn_settings.get("dither_strength", 0))
        palette_file = cn_settings.get("palette_file", "None")
        upsample = cn_settings.get("upsample", True)

        # Convert to PIL if needed
        if not isinstance(input_image, Image.Image):
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().numpy()
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray((input_image * 255).astype(np.uint8))

        # Get original dimensions
        original_width, original_height = input_image.size

        # Convert to RGB if needed
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Calculate target dimensions
        target_width = max(1, original_width // downscale_factor)
        target_height = max(1, original_height // downscale_factor)

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"K-Centroid downscaling to {target_width}x{target_height} with {color_clusters} clusters ...", None)
        )

        # Apply k-centroid algorithm
        processed_image = self.kCentroid(
            input_image, 
            target_width, 
            target_height, 
            color_clusters
        )

        # Load palette if specified
        palette_image = None
        if palette_file and palette_file != "None":
            palette_path = Path(path_manager.model_paths["palette_path"]) / palette_file
            if palette_path.exists():
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Loading palette '{palette_file}' ...", None)
                )
                palette_image = Image.open(palette_path).convert("RGB")

        # Apply palette quantization if palette is specified
        if palette_image is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Quantizing to palette colors ...", None)
            )
            processed_image = self.apply_palette_quantization(
                processed_image, 
                palette_image, 
                method=1  # MAXCOVERAGE
            )

        # Apply dithering if requested
        if dither and dither_strength > 0:
            if HITHERDITHER_AVAILABLE:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Applying Bayer dithering (strength: {dither_strength}) ...", None)
                )
                
                # Determine dither order from strength
                if dither_strength <= 3:
                    dither_order = 2  # 2x2 Bayer
                elif dither_strength <= 6:
                    dither_order = 4  # 4x4 Bayer
                else:
                    dither_order = 8  # 8x8 Bayer
                
                # Use palette if available, otherwise extract from processed image
                dither_palette = palette_image if palette_image else processed_image
                processed_image = self.apply_palette_dithering(
                    processed_image, 
                    dither_palette, 
                    dither_strength, 
                    dither_order
                )
            else:
                worker.add_result(
                    gen_data["task_id"],
                    "preview",
                    (-1, f"Dithering unavailable (hitherdither not installed)", None)
                )

        # Upsample back to original size with NEAREST (keeps pixels blocky)
        if upsample:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Upsampling back to {original_width}x{original_height} ...", None)
            )
            processed_image = processed_image.resize(
                (original_width, original_height), 
                Image.NEAREST
            )

        return [processed_image]
