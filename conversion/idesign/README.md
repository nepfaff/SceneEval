# IDesign Conversion for SceneEval

This directory contains tools for converting IDesign output to SceneEval input format.

## Overview

[IDesign](https://atcelen.github.io/I-Design/) is a personalized LLM interior designer that generates room layouts from text prompts. It outputs:
- `scene_graph.json`: Scene definition with object positions, rotations, sizes, and metadata
- `Assets/`: GLB files retrieved from Objaverse
- `scene.blend`: Blender scene file (optional)
- `render.png`: Rendered image (optional)

SceneEval can evaluate IDesign scenes using:
- **VLM metrics**: ObjCountMetric, ObjAttributeMetric, ObjObjRelationshipMetric, ObjArchRelationshipMetric, SupportMetric, AccessibilityMetric
- **Non-VLM metrics**: CollisionMetric, NavigabilityMetric, OutOfBoundMetric, OpeningClearanceMetric
- **Drake physics metrics**: DrakeCollisionMetricCoACD/VHACD, StaticEquilibriumMetricCoACD/VHACD, WeldedEquilibriumMetricCoACD/VHACD, ArchitecturalWeldedEquilibriumMetricCoACD/VHACD

Drake metrics use on-the-fly SDF generation (same as LayoutVLM/SceneWeaver/HSM).

## Conversion

### Usage

```bash
# Convert all scenes from IDesign scenes_batch directory
python conversion/idesign/convert_SceneEval.py \
    /home/ubuntu/IDesign/data/scenes_batch \
    input/IDesign

# With custom ID mapping
python conversion/idesign/convert_SceneEval.py \
    /home/ubuntu/IDesign/data/scenes_batch \
    input/IDesign \
    --mapping '{"0": 106, "1": 56}'
```

### Input Structure (IDesign output)

```
IDesign/data/scenes_batch/
├── scene_000/
│   ├── scene_graph.json      # Scene definition (flat JSON array)
│   ├── Assets/               # GLB files per object
│   │   ├── twin_bed_1.glb
│   │   ├── nightstand_1.glb
│   │   └── ...
│   ├── scene.blend           # Blender scene (optional)
│   └── render.png            # Rendered image (optional)
├── scene_001/
│   └── ...
```

### Output Structure (SceneEval input)

```
input/IDesign/
├── scene_0.json              # Scene state in SceneEval format
├── scene_0/
│   └── assets/
│       ├── twin_bed_1.glb
│       ├── nightstand_1.glb
│       ├── original_idesign.blend  # Copied from scene.blend
│       └── ...
├── scene_1.json
├── scene_1/
│   └── ...
```

## Transform Handling

The conversion follows IDesign's `place_in_blender.py` logic:

1. **Rotation**: Adds 180° to `z_angle` (IDesign convention)
2. **Scale**: Calculates scale factors from GLB bounding box vs `size_in_meters`
3. **Position**: Uses position directly from `scene_graph.json`
4. **Front Vector**: Sets `objectFrontVector: [0, 1, 0]` (same as SceneAgent)

## Running Evaluation

### Non-VLM Metrics Only

```bash
python main.py evaluation_plan=eval_plan \
    'evaluation_plan.input_cfg.scene_methods=[IDesign]' \
    'evaluation_plan.input_cfg.scene_range=[0,5]' \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,NavigabilityMetric,OutOfBoundMetric,OpeningClearanceMetric]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true' \
    'evaluation_plan.render_cfg.normal_render_tasks=[]'
```

### Drake Physics Metrics

```bash
python main.py evaluation_plan=eval_plan \
    'evaluation_plan.input_cfg.scene_methods=[IDesign]' \
    'evaluation_plan.input_cfg.scene_range=[0,5]' \
    'evaluation_plan.evaluation_cfg.metrics=[DrakeCollisionMetricCoACD,StaticEquilibriumMetricCoACD,WeldedEquilibriumMetricCoACD]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true' \
    'evaluation_plan.render_cfg.normal_render_tasks=[]' \
    '+metrics.StaticEquilibriumMetricCoACD.save_simulation_html=true' \
    '+metrics.WeldedEquilibriumMetricCoACD.save_simulation_html=true'
```

### Full Evaluation with Rendering

```bash
python main.py evaluation_plan=eval_plan \
    'evaluation_plan.input_cfg.scene_methods=[IDesign]' \
    'evaluation_plan.input_cfg.scene_range=[0,5]'
```

### All Non-VLM Metrics

```bash
python main.py evaluation_plan=eval_plan \
    'evaluation_plan.input_cfg.scene_methods=[IDesign]' \
    'evaluation_plan.input_cfg.scene_range=[0,5]' \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,NavigabilityMetric,OutOfBoundMetric,OpeningClearanceMetric,DrakeCollisionMetricCoACD,StaticEquilibriumMetricCoACD,WeldedEquilibriumMetricCoACD,DrakeCollisionMetricVHACD,StaticEquilibriumMetricVHACD,WeldedEquilibriumMetricVHACD,ArchitecturalWeldedEquilibriumMetricCoACD,ArchitecturalWeldedEquilibriumMetricVHACD]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true' \
    'evaluation_plan.render_cfg.normal_render_tasks=[]'
```

## IDesign Scene Format

### scene_graph.json Structure

```json
[
  {
    "new_object_id": "twin_bed_1",
    "style": "Modern",
    "material": "Wood",
    "size_in_meters": {"length": 1.9, "width": 1.0, "height": 0.6},
    "is_on_the_floor": true,
    "facing": "south_wall",
    "placement": {
      "room_layout_elements": [{"layout_element_id": "north_wall", "preposition": "on"}],
      "objects_in_room": []
    },
    "rotation": {"z_angle": 180.0},
    "position": {"x": 1.06, "y": 3.5, "z": 0.3}
  },
  {
    "new_object_id": "south_wall",
    "itemType": "wall",
    "position": {"x": 1.75, "y": 0, "z": 1.25},
    "size_in_meters": {"length": 3.5, "width": 0.0, "height": 2.5},
    "rotation": {"z_angle": 0.0}
  },
  ...
]
```

### Coordinate System

- **Origin**: Southwest corner of room
- **X-axis**: West to East
- **Y-axis**: South to North
- **Z-axis**: Floor to Ceiling (up)
- **Units**: Meters

## Notes

- IDesign uses Objaverse assets retrieved via OpenShape embeddings
- `use_empty_matching_result=true` is recommended unless you have annotation files
- Set `render_cfg.normal_render_tasks=[]` to skip rendering for faster evaluation
- Drake physics uses on-the-fly CoACD/VHACD decomposition (not pre-computed like SceneAgent)
- The conversion script requires `trimesh` for GLB bounding box calculation (falls back to identity scale if not available)
