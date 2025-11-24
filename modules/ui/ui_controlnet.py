import modules.controlnet as controlnet
from modules.controlnet import (
    cn_options,
    load_cnsettings,
    save_cnsettings,
    NEWCN,
)
import gradio as gr
from shared import add_ctrl, path_manager, translate
import modules.ui.ui_evolve as ui_evolve
import modules.ui.ui_llama as ui_llama
from PIL import Image

t = translate

def add_controlnet_tab(main_view, inpaint_view, prompt, image_number, run_event):
    with gr.Tab(label=t("PowerUp")):
        with gr.Row():
            cn_selection = gr.Dropdown(
                label=t("Cheat Code"),
                choices=["None"] + list(cn_options.keys()) + [NEWCN],
                value="None",
            )
            add_ctrl("cn_selection", cn_selection)

        cn_name = gr.Textbox(
            show_label=False,
            placeholder=t("Name"),
            interactive=True,
            visible='hidden',
        )
        cn_save_btn = gr.Button(
            value=t("Save"),
            visible='hidden',
        )

        type_choices=list(map(lambda x: x.capitalize(), controlnet.controlnet_models.keys()))
        cn_type = gr.Dropdown(
            label=t("Type"),
            choices=type_choices,
            value=type_choices[0],
            visible='hidden',
        )
        add_ctrl("cn_type", cn_type)

        cn_edge_low = gr.Slider(
            label=t("Edge (low)"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.2,
            visible='hidden',
        )
        add_ctrl("cn_edge_low", cn_edge_low)

        cn_edge_high = gr.Slider(
            label=t("Edge (high)"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.8,
            visible='hidden',
        )
        add_ctrl("cn_edge_high", cn_edge_high)

        cn_start = gr.Slider(
            label=t("Start"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            visible='hidden',
        )
        add_ctrl("cn_start", cn_start)

        cn_stop = gr.Slider(
            label=t("Stop"),
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=1.0,
            visible='hidden',
        )
        add_ctrl("cn_stop", cn_stop)

        cn_strength = gr.Slider(
            label=t("Strength"),
            minimum=0.0,
            maximum=2.0,
            step=0.01,
            value=1.0,
            visible='hidden',
        )
        add_ctrl("cn_strength", cn_strength)

        cn_upscaler = gr.Dropdown(
            label=t("Upscaler"),
            show_label=False,
            choices=["None"] + path_manager.upscaler_filenames,
            value="None",
            visible='hidden',
        )
        add_ctrl("cn_upscale", cn_upscaler)

        cn_downsample_factor = gr.Slider(
            label=t("Downsample Factor"),
            minimum=1,
            maximum=30,
            step=1,
            value=4,
            visible='hidden',
        )
        add_ctrl("cn_downsample_factor", cn_downsample_factor)

        cn_upsample = gr.Checkbox(
            label=t("Upsample to Original"),
            value=True,
            visible='hidden',
        )
        add_ctrl("cn_upsample", cn_upsample)

        cn_palette_file = gr.Dropdown(
            label=t("Palette File"),
            choices=["None"] + path_manager.palette_filenames,
            value="None",
            visible='hidden',
        )
        add_ctrl("cn_palette_file", cn_palette_file)

        cn_palette_reload = gr.Button(
            value="🔄 Reload Palettes",
            visible='hidden',
        )

        @cn_palette_reload.click(
            show_api=False,
            outputs=[cn_palette_file]
        )
        def reload_palette_list():
            # Refresh the palette filenames from disk
            path_manager.palette_filenames = path_manager.get_palette_filenames(
                path_manager.model_paths["palette_path"]
            )
            return gr.update(choices=["None"] + path_manager.palette_filenames)

        cn_extract_palette = gr.Checkbox(
            label=t("Extract & Save Palette"),
            value=False,
            visible='hidden',
        )
        add_ctrl("cn_extract_palette", cn_extract_palette)

        cn_palette_name = gr.Textbox(
            label=t("Palette Name"),
            placeholder=t("Enter palette name (optional)"),
            value="",
            visible='hidden',
        )
        add_ctrl("cn_palette_name", cn_palette_name)

        cn_palette_colors = gr.Slider(
            label=t("Palette Colors to Extract"),
            minimum=2,
            maximum=256,
            step=1,
            value=256,
            visible='hidden',
        )
        add_ctrl("cn_palette_colors", cn_palette_colors)

        @cn_extract_palette.change(
            show_api=False,
            inputs=[cn_extract_palette],
            outputs=[cn_palette_name, cn_palette_colors]
        )
        def extract_palette_changed(checked):
            if checked:
                return [gr.update(visible=True), gr.update(visible=True)]
            else:
                return [gr.update(visible='hidden'), gr.update(visible='hidden')]

        cn_dither = gr.Checkbox(
            label=t("Dither"),
            value=False,
            visible='hidden',
        )
        add_ctrl("cn_dither", cn_dither)

        cn_prequantize = gr.Checkbox(
            label=t("Prequantize"),
            value=False,
            visible='hidden',
        )
        add_ctrl("cn_prequantize", cn_prequantize)

        cn_prequantize_colors = gr.Slider(
            label=t("Prequantize Colors"),
            minimum=2,
            maximum=256,
            step=1,
            value=256,
            visible='hidden',
        )
        add_ctrl("cn_prequantize_colors", cn_prequantize_colors)

        @cn_prequantize.change(
            show_api=False,
            inputs=[cn_prequantize],
            outputs=[cn_prequantize_colors]
        )
        def prequantize_changed(checked):
            if checked:
                return gr.update(visible=True)
            else:
                return gr.update(visible='hidden')

        cn_quantize_mode = gr.Dropdown(
            label=t("Quantize Mode"),
            choices=["median_cut", "max_coverage", "fast_octree"],
            value="fast_octree",
            visible='hidden',
        )
        add_ctrl("cn_quantize_mode", cn_quantize_mode)

        # Quantize-specific controls
        cn_quantize_colors = gr.Slider(
            label=t("Colors"),
            minimum=1,
            maximum=256,
            step=1,
            value=64,
            visible='hidden',
        )
        add_ctrl("cn_quantize_colors", cn_quantize_colors)

        cn_quantize_method = gr.Dropdown(
            label=t("Quantize Method"),
            choices=["Median Cut", "Max Coverage", "Fast Octree"],
            value="Median Cut",
            visible='hidden',
        )
        add_ctrl("cn_quantize_method", cn_quantize_method)

        cn_quantize_kmeans = gr.Slider(
            label=t("K-means"),
            minimum=0,
            maximum=10,
            step=1,
            value=0,
            visible='hidden',
        )
        add_ctrl("cn_quantize_kmeans", cn_quantize_kmeans)

        cn_quantize_dither = gr.Checkbox(
            label=t("Dither (Quantize)"),
            value=True,
            visible='hidden',
        )
        add_ctrl("cn_quantize_dither", cn_quantize_dither)

        # K-Centroid specific controls
        cn_kcentroid_downscale = gr.Slider(
            label=t("Downscale Factor"),
            minimum=2,
            maximum=32,
            step=1,
            value=8,
            visible='hidden',
        )
        add_ctrl("cn_kcentroid_downscale", cn_kcentroid_downscale)

        cn_kcentroid_clusters = gr.Slider(
            label=t("Color Clusters"),
            minimum=2,
            maximum=128,
            step=1,
            value=24,
            visible='hidden',
        )
        add_ctrl("cn_kcentroid_clusters", cn_kcentroid_clusters)

        cn_kcentroid_dither_strength = gr.Slider(
            label=t("Dither Strength"),
            minimum=0,
            maximum=10,
            step=1,
            value=0,
            visible='hidden',
        )
        add_ctrl("cn_kcentroid_dither_strength", cn_kcentroid_dither_strength)

        cn_outputs = [
            cn_name,
            cn_save_btn,
            cn_type,
        ]
        cn_sliders = [
            cn_start,
            cn_stop,
            cn_strength,
            cn_edge_low,
            cn_edge_high,
            cn_upscaler,
            cn_downsample_factor,
            cn_upsample,
            cn_palette_file,
            cn_palette_reload,
            cn_extract_palette,
            cn_palette_name,
            cn_palette_colors,
            cn_dither,
            cn_prequantize,
            cn_prequantize_colors,
            cn_quantize_mode,
            cn_quantize_colors,
            cn_quantize_method,
            cn_quantize_kmeans,
            cn_quantize_dither,
            cn_kcentroid_downscale,
            cn_kcentroid_clusters,
            cn_kcentroid_dither_strength,
        ]

        @cn_selection.change(
            show_api=False,
            inputs=[cn_selection, cn_type],
            outputs=[cn_name] + cn_outputs + cn_sliders
        )
        def cn_changed(selection, current_type):
            if selection != NEWCN:
                return [gr.update(visible='hidden')] + [gr.update(visible='hidden')] * len(
                    cn_outputs + cn_sliders
                )
            else:
                # Show name input, save button, and type selector
                base_visible = [gr.update(value="")] + [gr.update(visible=True)] * len(cn_outputs)
                
                # Get visibility state for current type
                slider_states = {
                    "canny": [True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                    "img2img": [False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                    "default": [True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                    "upscale": [False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                    "faceswap": [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                    "pixelize": [False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                    "palettize": [False, False, False, False, False, False, False, False, True, True, True, False, False, True, True, False, True, False, False, False, False, False, False, False],
                    "quantize": [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False],
                    "kcentroid": [False, False, False, False, False, False, False, True, True, True, False, False, False, True, False, False, False, False, False, False, False, True, True, True],
                }
                
                type_key = current_type.lower() if current_type else "canny"
                show = slider_states.get(type_key, slider_states["default"])
                
                slider_updates = []
                for vis in show:
                    slider_updates += [gr.update(visible=True if vis else 'hidden')]
                
                return base_visible + slider_updates

        @cn_type.change(
            show_api=False,
            inputs=[cn_type],
            outputs=cn_sliders,
        )
        def cn_type_changed(selection):
            # cn_start,cn_stop,cn_strength,cn_edge_low,cn_edge_high, cn_upscaler, cn_downsample_factor, cn_upsample, cn_palette_file, cn_palette_reload, cn_extract_palette, cn_palette_name, cn_palette_colors, cn_dither, cn_prequantize, cn_prequantize_colors, cn_quantize_mode, cn_quantize_colors, cn_quantize_method, cn_quantize_kmeans, cn_quantize_dither, cn_kcentroid_downscale, cn_kcentroid_clusters, cn_kcentroid_dither_strength
            slider_states = {
                "canny": [True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                "img2img": [False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                "default": [True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                "upscale": [False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                "faceswap": [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                "pixelize": [False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                "palettize": [False, False, False, False, False, False, False, False, True, True, True, False, False, True, True, False, True, False, False, False, False, False, False, False],
                "quantize": [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False],
                "kcentroid": [False, False, False, False, False, False, False, True, True, True, False, False, False, True, False, False, False, False, False, False, False, True, True, True],
            }
            if selection.lower() in slider_states:
                show = slider_states[selection.lower()]
            else:
                show = slider_states["default"]

            result = []
            for vis in show:
                result += [gr.update(visible=True if vis else 'hidden')]

            return result

        @cn_save_btn.click(
            show_api=False,
            inputs=cn_outputs + cn_sliders,
            outputs=[cn_selection],
        )
        def cn_save(
            cn_name,
            cn_save_btn,
            cn_type,
            cn_start,
            cn_stop,
            cn_strength,
            cn_edge_low,
            cn_edge_high,
            upscale_model,
            downsample_factor,
            upsample,
            palette_file,
            palette_reload_btn,  # Ignored, just the button component
            extract_palette,
            palette_name,
            palette_colors,
            dither,
            prequantize,
            prequantize_colors,
            quantize_mode,
            quantize_colors,
            quantize_method,
            quantize_kmeans,
            quantize_dither,
            kcentroid_downscale,
            kcentroid_clusters,
            kcentroid_dither_strength,
        ):
            if cn_name != "":
                cn_options = load_cnsettings()
                opts = {
                    "type": cn_type.lower(),
                    "start": cn_start,
                    "stop": cn_stop,
                    "strength": cn_strength,
                    "upscaler": upscale_model,
                }
                if cn_type.lower() == "canny":
                    opts.update(
                        {
                            "edge_low": cn_edge_low,
                            "edge_high": cn_edge_high,
                        }
                    )
                elif cn_type.lower() == "pixelize":
                    opts = {
                        "type": "pixelize",
                        "downsample_factor": int(downsample_factor),
                        "upsample": bool(upsample),
                    }
                elif cn_type.lower() == "palettize":
                    opts = {
                        "type": "palettize",
                        "palette_file": str(palette_file),
                        "dither": bool(dither),
                        "prequantize": bool(prequantize),
                        "prequantize_colors": int(prequantize_colors),
                        "quantize_mode": str(quantize_mode),
                    }
                elif cn_type.lower() == "quantize":
                    opts = {
                        "type": "quantize",
                        "colors": int(quantize_colors),
                        "method": str(quantize_method),
                        "kmeans": int(quantize_kmeans),
                        "dither": bool(quantize_dither),
                    }
                elif cn_type.lower() == "kcentroid":
                    opts = {
                        "type": "kcentroid",
                        "downscale_factor": int(kcentroid_downscale),
                        "color_clusters": int(kcentroid_clusters),
                        "dither": bool(kcentroid_dither_strength > 0),
                        "dither_strength": int(kcentroid_dither_strength),
                        "palette_file": str(palette_file),
                        "upsample": bool(upsample),
                    }
                cn_options[cn_name] = opts
                save_cnsettings(cn_options)
                choices = list(cn_options.keys()) + [NEWCN]
                return gr.update(choices=choices, value=cn_name)
            else:
                return gr.update()

        input_image = gr.Image(
            label=t("Input image"),
            type="pil",
            visible=True,
        )
        add_ctrl("input_image", input_image)
        inpaint_toggle = gr.Checkbox(label=t("Inpainting"), value=False)

        add_ctrl("inpaint_toggle", inpaint_toggle)

        @inpaint_toggle.change(
            show_api=False,
            inputs=[inpaint_toggle, main_view],
            outputs=[main_view, inpaint_view]
        )
        def inpaint_checked(r, image):
            if r:
                base_height = 600
                # Handle both PIL Image objects and file paths
                if isinstance(image, Image.Image):
                    img = image
                else:
                    img = Image.open(image)
                scale = (base_height / float(img.size[1]))
                width = int((float(img.size[0]) * float(scale)))
                img = img.resize((width, base_height), Image.Resampling.LANCZOS)

                return {
                    main_view: gr.update(visible='hidden'),
                    inpaint_view: gr.update(
                        visible=True,
                        interactive=True,
                        value={
                            'background': img,
                            'layers': [Image.new("RGBA", (width, base_height))],
                            'composite': None,
                        },
                    )
                }
            else:
                return {
                    main_view: gr.update(visible=True),
                    inpaint_view: gr.update(
                        visible='hidden',
                        interactive=False,
                    ),
                }

        ui_evolve.add_evolve_tab(prompt, image_number, run_event)

        ui_llama.add_llama_tab(prompt)

    return inpaint_toggle

