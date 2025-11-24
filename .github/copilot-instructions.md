# RuinedFooocus AI Coding Agent Instructions

## Project Overview
RuinedFooocus is an enhanced Stable Diffusion XL (SDXL) image generation UI built on ComfyUI backend, featuring multiple AI pipelines for image/video generation, LLM integration, and extensive customization options.

## Architecture

### Pipeline System (`modules/pipelines.py`)
- **Dynamic pipeline selection**: Routing based on prompt prefix, controlnet type, or model architecture
  - `#!` prefix → `hashbang_pipeline` (custom commands like save/merge)
  - `search:` prefix → `search_pipeline` (image search)
  - `ruinedfooocuslogo` → `template_pipeline` (built-in templates)
  - Model-based: Hunyuan Video, WAN Video, LTX Video → respective video pipelines
  - Default → `sdxl_pipeline` (main image generation)
- **Pipeline interface**: All pipelines implement `load_base_model()`, `load_loras()`, `process()`, and `pipeline_type` attribute
- ComfyUI nodes wrapped in pipeline classes for model loading, sampling, VAE encoding/decoding

### Model Management (`modules/model_handler.py`)
- Automatic CivitAI integration: downloads thumbnails and LoRA trigger words from metadata
- Supports subdirectories in `models/checkpoints/` and `models/loras/`
- Model detection via hash or architecture inspection (SDXL, Flux, SD3, etc.)

### Async Processing (`modules/async_worker.py`)
- Background worker thread processes generation queue from `buffer`
- `process(gen_data)` → pipeline selection → image generation → results stored in `outputs`
- `gen_data` dict contains all generation parameters (prompt, seed, loras, controlnet, etc.)

### Settings System
- JSON-based configuration in `settings/` folder with `--settings` arg for multiple profiles
- `shared.py` centralizes state: `settings`, `path_manager`, `performance_settings`, `resolution_settings`, `models`
- Custom paths via `settings/paths.json`, custom resolutions via `settings/resolutions.json`
- Styles defined in `settings/styles.csv`

## Key Workflows

### Adding a New Pipeline
1. Create `modules/new_pipeline.py` with `class pipeline:` and `pipeline_type = ["newtype"]`
2. Implement: `load_base_model()`, `load_loras()`, `refresh_controlnet()`, `process(gen_data, callback)`
3. Add import and routing logic to `modules/pipelines.py` in `update()` function
4. Use ComfyUI nodes from `nodes` and `comfy_extras` modules

### Prompt Processing (`modules/prompt_processing.py`)
- JSON metadata injection: entire gen_data can be passed as JSON in prompt
- Wildcards: `__filename__` replaced with random line from `wildcards/filename.txt`
- In-prompt LoRAs: `<lora:name:weight>` auto-loaded
- Multi-subject: `---` separator for batch prompts
- Style injection: `<style:stylename>` applies from `settings/styles.csv`

### Model Loading Pattern (SDXL pipeline)
```python
from comfy.sd import load_checkpoint_guess_config
# Load checkpoint
self.model, self.clip, self.vae = load_checkpoint_guess_config(path)
# Apply LoRAs
for lora in loras:
    self.model, self.clip = comfy.sd.load_lora_for_models(self.model, self.clip, lora_path, strength, strength)
```

### ControlNet/PowerUp System (`modules/controlnet.py`)
- Settings stored in `settings/powerup.json` with named presets
- Types: canny, depth, img2img, upscale, rembg, faceswap, inpainting
- `get_settings(gen_data)` extracts current controlnet config from gen_data

## Project-Specific Conventions

### Import Order
1. ComfyUI core: `comfy.`, `nodes`, `comfy_extras`
2. Project modules: `modules.`, `shared`
3. Always import `shared` for access to managers and state

### Path Resolution
- Use `path_manager.model_paths["modelfile_path"]` for model locations
- Paths can be lists (multiple search locations) or single strings
- Always handle relative paths starting with `../` (parent directory is working root)

### Error Handling
- Pipelines fail gracefully: return None or empty results rather than crash
- Use try/except around model operations with console logging
- Preview images sent via `worker.add_result(task_id, "preview", (step, desc, path))`

### UI Components (`webui.py`)
- Gradio-based with custom CSS in `modules/html.py`
- State managed via `shared.state["ctrls_obj"]` for dynamic UI updates
- `generate_clicked()` converts UI args to `gen_data` dict and adds to queue
- Multi-tab interface: Main (generation), Image browser, Chat bots, Settings

## Testing & Debugging
- Run with `python launch.py` (auto-installs dependencies via torchruntime)
- `--nobrowser` prevents auto-launch
- `--listen 0.0.0.0` for network access
- Check console for pipeline selection: "Using default pipeline" indicates SDXL
- Test mode: Set `image_number` to 0 for infinite generation

## Dependencies
- ComfyUI backend in `repositories/ComfyUI/` (git submodule)
- GGUF support via `repositories/calcuis_gguf/`
- OneButtonPrompt in `random_prompt/` for dynamic prompt generation
- Translation support: `language/en.json`, `language/sv.json`

## Common Gotchas
- Pipeline switching requires checking `state["pipeline"]` type before reassigning
- LoRA weights default to 0.5, stored as tuples: `(name, weight)`
- Video models need `EmptyHunyuanLatentVideo` instead of `EmptyLatentImage`
- Always check model architecture before applying model-specific nodes (ModelSamplingSD3, ModelSamplingAuraFlow, etc.)
