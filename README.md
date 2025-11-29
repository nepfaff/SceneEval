# SceneEval

### SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis

[Hou In Ivan Tam](https://iv-t.github.io/), [Hou In Derek Pun](https://houip.github.io/), [Austin T. Wang](https://atwang16.github.io/), [Angel X. Chang](https://angelxuanchang.github.io/), [Manolis Savva](https://msavva.github.io/)

<!-- <img src="docs/static/images/teaser.webp" alt="teaser" style="width:100%"/> -->

[Page](https://3dlg-hcvc.github.io/SceneEval/) | [Paper](https://arxiv.org/abs/2503.14756) | [Data](https://github.com/3dlg-hcvc/SceneEval/releases)



## News
- 2025-11-28: Added [SceneAgent](https://github.com/your-org/scene-agent) support with Drake physics metrics using pre-computed SDFs!
- 2025-11-26: Added [SceneWeaver](https://github.com/princeton-vl/SceneWeaver) support with proper Infinigen object decomposition!
- 2025-11-04: Released scripts for converting Holodeck output into SceneEval-compatible format!
- 2025-10-27: Release v1.1 with a new metric *Opening Clearance*, support for [LayoutVLM](https://github.com/sunfanyunn/LayoutVLM) and [HSM](https://github.com/3dlg-hcvc/hsm), bug fixes, and more! The environment setup is now simplified and the demo is easier to run! Give it a try!
- 2025-06-27: Codebase release v1.0!
- 2025-06-10: Released SceneEval-500 dataset and v0.9 of the SceneEval codebase!



## Todo List
- [x] Add documentation for the scene state format
- [x] Provide script for downloading and processing Holodeck's assets
- [x] Create guide for extending SceneEval with new methods and metrics
- [ ] Replace custom VLM interface with Pydantic AI



## Quick Start

### 1. Install dependencies with uv

```bash
uv sync
source .venv/bin/activate
```

### 2. Download data

Run the setup script to download all available datasets:

```bash
./setup.sh
```

This will:
- Download SceneEval-500 annotations
- Download Objathor assets (for Holodeck) - ~50GB
- Download HSSD assets (for HSM) - ~80GB (requires `gltf-transform` and `ktx`)

**Manual downloads required** (due to licensing):
- **3D-FUTURE** (for ATISS, DiffuScene, LayoutGPT, InstructScene): [Download here](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) → extract to `_data/3D-FUTURE-model/`
- **LayoutVLM assets**: [Download here](https://github.com/sunfanyunn/LayoutVLM#data-preprocessing) → extract to `_data/layoutvlm-objathor/`

### 3. (Optional) Setup OpenAI API Key

For VLM-dependent metrics (*Object Count*, *Object Attribute*, *Object-Object Relationship*, *Object-Architecture Relationship*, *Object Support*, *Object Accessibility*), create a `.env` file:
```
OPENAI_API_KEY=<your_key>
```

Metrics that do NOT require a VLM: *Collision*, *Navigability*, *Out of Bounds*, *Opening Clearance*, *Drake Collision*, *Static Equilibrium*, *Welded Equilibrium*

---

<details>
<summary><strong>Alternative: Install with conda</strong></summary>

```bash
conda env create -f environment.yaml
conda activate scene_eval
```

</details>

<details>
<summary><strong>HSSD prerequisites (for HSM support)</strong></summary>

To download HSSD assets, you need:
1. [Agree to the HSSD dataset license on HuggingFace](https://huggingface.co/datasets/hssd/hssd-models)
2. Install tools:
   - [gltf-transform](https://www.npmjs.com/package/@gltf-transform/cli): `npm install -g @gltf-transform/cli`
   - [ktx](https://github.com/KhronosGroup/KTX-Software/releases) from KhronosGroup

</details>

<details>
<summary><strong>Expected <code>_data/</code> directory structure</strong></summary>

```
_data
├── 3D-FUTURE-model
│   ├── model_info.json
│   ├── 0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e
│   |   ├── raw_model.obj
│   |   ├── model.mtl
│   |   ├── texture.png
│   |   ├── ...
│   ├── ...
├── objathor-assets
│   ├── annotations.json
│   ├── 0a0a8274693445a6b533dce7f97f747c
│   |   ├── 0a0a8274693445a6b533dce7f97f747c.glb
│   |   ├── ...
├── layoutvlm-objathor
│   ├── 0a3dc72fb1bb41439005cac4a3ebb765
│   |   ├── 0a3dc72fb1bb41439005cac4a3ebb765.glb
│   |   ├── data.json
│   |   ├── ...
│   ├── ...
└── hssd
    ├── fpmodels.csv
    ├── glb
     │   ├── 0
     │   |   ├── 0a0b9f607345b6cee099d681f28e19e1f0a215c8.glb
     │   |   ├── ...
     │   ├── ...
     └── decomposed
         ├── 00a2b0f3886ccb5ffddac704f8eeec324a5e14c6
         |   ├── 00a2b0f3886ccb5ffddac704f8eeec324a5e14c6_part_1.glb
         |   ├── 00a2b0f3886ccb5ffddac704f8eeec324a5e14c6_part_2.glb
         |   ├── ...
         ├── ...
```

</details>


## Object Matching Behavior

VLM-based metrics require matching scene objects to expected categories from the annotation. The `ObjMatching` metric uses GPT-4o to classify each object.

### How It Works

1. Target categories are extracted from the annotation's `ObjCount` field (e.g., `"eq,1,sofa; eq,1,coffee_table"` → `["sofa", "coffee_table"]`)
2. Each object is rendered from the front view
3. GPT-4o is asked to match the object to one of the target categories
4. **Object dimensions are provided** to help distinguish between similar objects (e.g., coaster vs coffee table)

### Matching Rules

The matching uses **strict rules** to avoid false matches:
- Only exact category matches are accepted (a side table is NOT a coffee table)
- Size is considered: a 10cm coaster cannot match a 100cm coffee table
- When uncertain, objects are left unmatched rather than incorrectly categorized

### Reporting Metrics

When reporting VLM-based metrics, be aware:
- `obj_matching_result.json` shows `per_category` (matched) and `not_matched_objs` (rejected)
- Scene generation methods may create objects not in the expected categories
- Match rate depends on both generation quality AND annotation specificity

**Example**: If annotation expects `coffee_table` but generation creates `side_table` + `coaster`, neither will match despite being table-like objects.



## Quick Start Demo

Try SceneEval with our provided example scenes. You do *not* need an OpenAI API key for this demo.

### 1. Download the LayoutVLM Assets
Follow the instructions in the [*For LayoutVLM*](#layoutvlm) section above to download the LayoutVLM assets.

### 2. Run the Demo
```bash
# Copy provided example scenes to input directory
cp -r input_example/* input/

# Run SceneEval
python main.py
```
This will run the evaluation on five example scenes generated by LayoutVLM using the `no_llm_plan` evaluation plan, which runs the following metrics: *Collision*, *Navigability*, *Out of Bounds*, and *Opening Clearance*.

Results will be saved to `./output_eval`.



## Extending SceneEval to a New Method or Dataset

SceneEval is built to be extensible! You can easily add new scene generation methods, evaluation metrics, and assets.

**[Follow this step-by-step guide to see how to add a new method that uses a new 3D asset source.](./GUIDE.md)**



## Contributing to SceneEval

Found a bug or want to contribute a new method or metric? We'd love your help! Please open an issue or submit a pull request. 



## Citation
If you find SceneEval helpful in your research, please cite our work:
```
@article{tam2025sceneeval,
    title = {{SceneEval}: Evaluating Semantic Coherence in Text-Conditioned {3D} Indoor Scene Synthesis},
    author = {Tam, Hou In Ivan and Pun, Hou In Derek and Wang, Austin T. and Chang, Angel X. and Savva, Manolis},
    year = {2025},
    eprint = {2503.14756},
    archivePrefix = {arXiv}
}
```

**Note:** When using SceneEval in your work, we encourage you to specify which version of the code and dataset you are using (e.g., commit hash, release tag, or dataset version) to ensure reproducibility and proper attribution.

## SceneWeaver Support

SceneEval supports evaluating scenes generated by [SceneWeaver](https://github.com/princeton-vl/SceneWeaver). The conversion script properly handles Infinigen's procedural object factories, exporting each object individually (including bed components like mattress, pillows, comforter, etc.) rather than bundling them.

### Converting SceneWeaver Output

Use Blender to run the conversion script:

```bash
blender --background --python conversion/sceneweaver/convert_SceneEval.py -- \
    --input_dir /path/to/SceneWeaver/Pipeline/output/Design_me_a_messy_kids_bedroom_0 \
    --output_dir ./input/SceneWeaver \
    --scene_id 0
```

**Arguments:**
- `--input_dir`: Path to SceneWeaver output directory (contains `record_files/`, `record_scene/`, `roominfo.json`)
- `--output_dir`: Path to SceneEval input directory (default: `./input/SceneWeaver`)
- `--scene_id`: Scene ID for output filename (default: 0)
- `--wall_height`: Wall height in meters (default: 2.8)

### Running Evaluation

After conversion, run SceneEval:

```bash
python main.py \
    evaluation_plan.input_cfg.scene_methods=[SceneWeaver] \
    evaluation_plan.input_cfg.scene_mode=range \
    evaluation_plan.input_cfg.scene_range=[0,1]
```

Results will be saved to `./output_eval/SceneWeaver/`.

### How It Works

The conversion script:
1. Opens the SceneWeaver Blender file (`.blend`)
2. Finds all objects matching the `*.spawn_asset(*` pattern (Infinigen factory objects)
3. Exports each object as an individual GLB file with world-space transforms
4. Creates a SceneEval scene state JSON with proper architecture (floor, walls)

This approach matches SceneWeaver's own object counting methodology (`Nobj_unique` in `metric_*.json`), ensuring fair evaluation.


## LayoutVLM Support

SceneEval supports evaluating scenes generated by [LayoutVLM](https://github.com/sunfanyunn/LayoutVLM). LayoutVLM uses Objaverse assets and outputs scenes with separate files for assets and layout.

### Converting LayoutVLM Output

Use the conversion script to convert LayoutVLM output to SceneEval format:

```bash
python conversion/layoutvlm/convert_SceneEval.py \
    /path/to/LayoutVLM/results \
    input/LayoutVLM \
    --mapping '{"0": 106, "1": 56, "2": 39, "3": 74, "4": 94}'
```

**Arguments:**
- First positional: Path to LayoutVLM results directory (contains `scene_000/`, `scene_001/`, etc.)
- Second positional: Path to SceneEval input directory (e.g., `input/LayoutVLM`)
- `--mapping`: Optional JSON mapping from source scene index to target scene ID. If not provided, scene IDs are preserved.

**LayoutVLM Output Structure:**
```
results/
├── scene_000/
│   ├── scene.json                    # Asset IDs and room boundary
│   ├── layout.json                   # Object positions and rotations
│   └── complete_sandbox_program.py   # Asset metadata (optional)
├── scene_001/
└── ...
```

**Mapping Example:**
If you generated scenes from SceneEval-500 prompts but LayoutVLM named them sequentially:
- `scene_000` → scene ID 106
- `scene_001` → scene ID 56
- `scene_002` → scene ID 39

Use: `--mapping '{"0": 106, "1": 56, "2": 39}'`

### Running Evaluation

After conversion, run SceneEval:

```bash
# Evaluate all LayoutVLM scenes
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]' \
    'evaluation_plan.input_cfg.scene_mode=range' \
    'evaluation_plan.input_cfg.scene_range=[0,5]'

# Or evaluate specific scene IDs (e.g., from SceneEval-500), 3 workers
./scripts/run_parallel_list.sh "39,56,74,94,106" 3 \
    'evaluation_plan=eval_plan' \
    'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]'
```

Results will be saved to `./output_eval/LayoutVLM/`.

Note that SceneAgent needs to use `sceneagent_plan.yaml` for its
specific Drake metrics.


## SceneAgent Support

SceneEval supports evaluating scenes generated by [scene-agent](https://github.com/your-org/scene-agent). Scene-agent exports `sceneeval_state.json` with pre-computed physics properties (inertia tensors, CoACD collision geometry, friction coefficients).

### Converting SceneAgent Output

```bash
python conversion/scene_agent/convert_SceneEval.py \
    ~/scene-agent/outputs/2025-11-27/21-25-16 \
    input/SceneAgent
```

This copies:
- `scene_states/final_scene/sceneeval_state.json` → `scene_X.json`
- `generated_assets/` → `scene_X/assets/`
- `floor_plan.sdf` → `scene_X/floor_plan.sdf` (for Drake metrics)

### Running Evaluation

Use the dedicated evaluation plan for SceneAgent:

```bash
# Run all SceneAgent metrics (uses sceneagent_plan.yaml)
python main.py evaluation_plan=sceneagent_plan

# With scene range override
python main.py evaluation_plan=sceneagent_plan \
    'evaluation_plan.input_cfg.scene_range=[0,5]'
```

Or run specific metrics:

```bash
# Non-Drake metrics only
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[SceneAgent]' \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,NavigabilityMetric,OutOfBoundMetric,OpeningClearanceMetric]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true'

# SceneAgent Drake metrics
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[SceneAgent]' \
    'evaluation_plan.evaluation_cfg.metrics=[DrakeCollisionMetricSceneAgent,StaticEquilibriumMetricSceneAgent,WeldedEquilibriumMetricSceneAgent]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true'
```

Results will be saved to `./output_eval/SceneAgent/`.

### SceneAgent-Specific Drake Metrics

SceneAgent has dedicated Drake metrics that use its pre-computed SDFs with CoACD collision geometry:

| SceneAgent Metric | Equivalent Standard Metric | Difference |
|-------------------|---------------------------|------------|
| `DrakeCollisionMetricSceneAgent` | `DrakeCollisionMetricCoACD` | Uses pre-computed SDFs from scene-agent |
| `StaticEquilibriumMetricSceneAgent` | `StaticEquilibriumMetricCoACD` | Uses pre-computed SDFs from scene-agent |
| `WeldedEquilibriumMetricSceneAgent` | `WeldedEquilibriumMetricCoACD` | Uses pre-computed SDFs from scene-agent |

**Why separate metrics?** The physics simulation and evaluation logic is identical. The only difference is how the Drake plant is constructed:
- **Standard CoACD metrics**: Build SDFs from scratch using trimesh + CoACD decomposition
- **SceneAgent metrics**: Use scene-agent's pre-computed SDFs directly (SDF pass-through)

This preserves the exact collision geometry that scene-agent used during placement, avoiding redundant computation and ensuring physics consistency.

**Note**: VHACD metrics are not available for SceneAgent since scene-agent only computes CoACD decomposition. The SceneAgent metrics are functionally equivalent to the CoACD variants.


## Drake Physics Metrics

SceneEval includes physics-based metrics using [Drake](https://drake.mit.edu/) for more accurate collision detection and stability analysis. Each metric is available with two convex decomposition methods:

- **CoACD** ([Convex Approximate Convex Decomposition](https://github.com/SarahWeiii/CoACD)) - Adaptive decomposition, typically more accurate but slower
- **VHACD** (Volumetric Hierarchical Approximate Convex Decomposition) - Fixed 64 convex pieces, faster but may miss fine details

### Available Metrics

| Metric | Description |
|--------|-------------|
| **DrakeCollisionMetricCoACD** / **DrakeCollisionMetricVHACD** | Detects collisions using convex decomposition for physics-accurate collision geometry. |
| **StaticEquilibriumMetricCoACD** / **StaticEquilibriumMetricVHACD** | Runs physics simulation and measures object displacement. Lower displacement = better stability. |
| **WeldedEquilibriumMetricCoACD** / **WeldedEquilibriumMetricVHACD** | Detects penetrating objects, welds them to the world, then simulates. Isolates stability issues from penetration-induced movement. |

### Why Drake-based Collision Detection?

The original `CollisionMetric` uses trimesh intersection which checks exact mesh geometry. However, physics simulators like Drake use **convex collision geometry** (via CoACD/VHACD decomposition), which can slightly inflate meshes. Drake metrics detect collisions that would actually occur in physics simulation.

### Output Statistics

`DrakeCollisionMetric*` reports:
- `max_penetration_depth` - Maximum penetration depth across all collision pairs
- `min_penetration_depth` - Minimum penetration depth
- `mean_penetration_depth` - Average penetration depth
- `median_penetration_depth` - Median penetration depth
- `num_collision_pairs` - Number of unique colliding object pairs
- `num_obj_in_collision` - Number of objects involved in collisions

`StaticEquilibriumMetric*` and `WeldedEquilibriumMetric*` report:
- `scene_stable` - Whether all objects are stable (displacement < threshold)
- `mean_displacement` - Average object displacement after simulation
- `max_displacement` - Maximum object displacement
- `per_object_results` - Displacement and rotation per object

### HTML Visualization for Debugging

The equilibrium metrics can export interactive Meshcat visualizations as HTML files. This is useful for debugging physics issues like objects flying away unexpectedly.

```bash
# Enable HTML visualization for a specific metric
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]' \
    'evaluation_plan.evaluation_cfg.metrics=[WeldedEquilibriumMetricVHACD]' \
    '+metrics.WeldedEquilibriumMetricVHACD.save_simulation_html=true'
```

This saves interactive HTML files to metric-specific folders:
- `static_equilibrium_vhacd/simulation/simulation.html` - StaticEquilibrium simulation
- `welded_equilibrium_vhacd/simulation/simulation.html` - WeldedEquilibrium simulation

Each metric has its own output folder:
```
output_eval/<Method>/<scene>/
├── drake_collision_coacd/          # DrakeCollisionMetricCoACD
├── drake_collision_vhacd/          # DrakeCollisionMetricVHACD
├── static_equilibrium_coacd/       # StaticEquilibriumMetricCoACD
│   └── simulation/
│       └── simulation.html
├── static_equilibrium_vhacd/       # StaticEquilibriumMetricVHACD
│   └── simulation/
│       └── simulation.html
├── welded_equilibrium_coacd/       # WeldedEquilibriumMetricCoACD
│   ├── penetration_detection/      # Static scene for detecting collisions
│   └── simulation/                 # Dynamic simulation with colliding objects welded
│       └── simulation.html
└── welded_equilibrium_vhacd/       # WeldedEquilibriumMetricVHACD
    ├── penetration_detection/
    └── simulation/
        └── simulation.html
```

Open the HTML in a browser to:
1. View the scene with convex collision geometry
2. Play back the simulation timeline
3. Identify objects that move unexpectedly

### Usage

```bash
# Run all Drake physics metrics (both CoACD and VHACD)
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]' \
    'evaluation_plan.evaluation_cfg.metrics=[DrakeCollisionMetricCoACD,DrakeCollisionMetricVHACD,StaticEquilibriumMetricCoACD,StaticEquilibriumMetricVHACD,WeldedEquilibriumMetricCoACD,WeldedEquilibriumMetricVHACD]'

# Run only VHACD variants (faster)
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]' \
    'evaluation_plan.evaluation_cfg.metrics=[DrakeCollisionMetricVHACD,StaticEquilibriumMetricVHACD,WeldedEquilibriumMetricVHACD]'

# Enable HTML visualization for all equilibrium metrics
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]' \
    'evaluation_plan.evaluation_cfg.metrics=[StaticEquilibriumMetricVHACD,WeldedEquilibriumMetricVHACD]' \
    '+metrics.StaticEquilibriumMetricVHACD.save_simulation_html=true' \
    '+metrics.WeldedEquilibriumMetricVHACD.save_simulation_html=true'
```


## Parallel Evaluation

For large-scale evaluation, use the bash scripts to run multiple independent processes. Three scripts are available depending on your use case:

### 1. Range Mode (`run_parallel.sh`)

For evaluating a contiguous range of scenes (0 to N-1):

```bash
# Evaluate scenes 0-99 with 4 parallel workers
./scripts/run_parallel.sh 100 4 \
    'evaluation_plan.input_cfg.scene_methods=[SceneWeaver]' \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,StaticEquilibriumMetricCoACD]'
```

**Arguments:**
- First: Total number of scenes (evaluates 0 to N-1)
- Second: Number of parallel workers
- Remaining: Passed to `main.py`

### 2. List Mode (`run_parallel_list.sh`)

For evaluating specific non-contiguous scene IDs:

```bash
# Evaluate specific scenes with 3 workers
./scripts/run_parallel_list.sh "39,56,74,94,106" 3 \
    'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]'

# With metrics
./scripts/run_parallel_list.sh "0,5,10,15,20" 5 \
    'evaluation_plan.input_cfg.scene_methods=[SceneWeaver]' \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,StaticEquilibriumMetricCoACD]'
```

**Arguments:**
- First: Comma-separated list of scene IDs
- Second: Number of parallel workers
- Remaining: Passed to `main.py`

### 3. All Mode (`run_parallel_all.sh`)

Automatically discovers and evaluates all scenes in an input directory:

```bash
# Evaluate ALL LayoutVLM scenes with 4 workers
./scripts/run_parallel_all.sh LayoutVLM 4

# With metrics
./scripts/run_parallel_all.sh SceneWeaver 8 \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,StaticEquilibriumMetricCoACD]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=True'
```

**Arguments:**
- First: Method name (directory under `input/`)
- Second: Number of parallel workers
- Remaining: Passed to `main.py`

This script scans `input/<method>/` for all `scene_*.json` files, extracts their IDs, and splits them evenly across workers.

### Output

All scripts produce:
- Each scene creates `eval_result.json` and `eval.log` in its output directory
- Worker logs are saved to `logs/worker_*.log`, `logs/worker_list_*.log`, or `logs/worker_all_*.log`

This approach runs completely independent Python processes, avoiding any issues with Blender's `bpy` module or Drake's Meshcat in multi-threaded environments.


## Acknowledgements
This work was funded in part by the Sony Research Award Program, a CIFAR AI Chair, a Canada Research Chair, NSERC Discovery Grants, and enabled by support from the [Digital Research Alliance of Canada](https://alliancecan.ca/).
We thank Nao Yamato, Yotaro Shimose, and other members on the Sony team for their feedback.
We also thank Qirui Wu, Xiaohao Sun, and Han-Hung Lee for helpful discussions.
