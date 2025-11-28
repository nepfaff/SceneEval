# SceneAgent Conversion for SceneEval

This directory contains tools for converting scene-agent output to SceneEval input format.

## Overview

Scene-agent exports scenes with:
- Pre-computed CoACD collision geometry
- Full inertia tensors (mass, CoM, ixx, iyy, izz, ixy, ixz, iyz)
- SDF files for each object
- Floor plan SDF with architecture

SceneEval can evaluate these scenes using both:
- **Non-Drake metrics**: CollisionMetric, NavigabilityMetric, OutOfBoundMetric, OpeningClearanceMetric
- **Drake metrics**: DrakeCollisionMetricSceneAgent, StaticEquilibriumMetricSceneAgent, WeldedEquilibriumMetricSceneAgent

The Drake metrics use scene-agent's pre-computed SDFs directly, avoiding the need to regenerate collision geometry.

## Conversion

### Usage

```bash
python conversion/scene_agent/convert_SceneEval.py \
    ~/scene-agent/outputs/2025-11-27/21-25-16 \
    input/SceneAgent
```

### Input Structure (scene-agent output)

```
scene-agent/outputs/YYYY-MM-DD/HH-MM-SS/
├── scene_000/
│   ├── scene_states/
│   │   └── final_scene/
│   │       └── sceneeval_state.json    # Scene state in SceneEval format
│   ├── generated_assets/
│   │   ├── furniture/
│   │   │   └── workstation_desk/
│   │   │       ├── workstation_desk.sdf
│   │   │       ├── workstation_desk.obj
│   │   │       └── workstation_desk_collision_*.obj
│   │   └── manipulands/
│   │       └── coffee_mug/
│   │           └── ...
│   └── floor_plan.sdf
├── scene_001/
│   └── ...
```

### Output Structure (SceneEval input)

```
input/SceneAgent/
├── scene_0.json                # Scene state (modelId updated)
├── scene_0/
│   ├── assets/
│   │   ├── furniture/
│   │   │   └── workstation_desk/
│   │   │       └── ...
│   │   └── manipulands/
│   │       └── coffee_mug/
│   │           └── ...
│   └── floor_plan.sdf          # Architecture SDF
├── scene_1.json
├── scene_1/
│   └── ...
```

## Running Evaluation

### Non-Drake Metrics

Standard geometry-based metrics work out of the box:

```bash
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[SceneAgent]' \
    'evaluation_plan.input_cfg.scene_mode=range' \
    'evaluation_plan.input_cfg.scene_range=[0,3]' \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,NavigabilityMetric,OutOfBoundMetric,OpeningClearanceMetric]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true' \
    'evaluation_plan.render_cfg.normal_render_tasks=[]'
```

### Drake Metrics (Physics Simulation)

SceneAgent has specialized Drake metrics that use pre-computed SDFs:

```bash
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[SceneAgent]' \
    'evaluation_plan.input_cfg.scene_mode=range' \
    'evaluation_plan.input_cfg.scene_range=[0,3]' \
    'evaluation_plan.evaluation_cfg.metrics=[DrakeCollisionMetricSceneAgent,StaticEquilibriumMetricSceneAgent,WeldedEquilibriumMetricSceneAgent]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true' \
    'evaluation_plan.render_cfg.normal_render_tasks=[]' \
    '+metrics.StaticEquilibriumMetricSceneAgent.save_simulation_html=true' \
    '+metrics.WeldedEquilibriumMetricSceneAgent.save_simulation_html=true'
```

### All Metrics Combined

```bash
python main.py \
    'evaluation_plan.input_cfg.scene_methods=[SceneAgent]' \
    'evaluation_plan.input_cfg.scene_mode=range' \
    'evaluation_plan.input_cfg.scene_range=[0,3]' \
    'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,NavigabilityMetric,OutOfBoundMetric,OpeningClearanceMetric,DrakeCollisionMetricSceneAgent,StaticEquilibriumMetricSceneAgent,WeldedEquilibriumMetricSceneAgent]' \
    'evaluation_plan.evaluation_cfg.use_empty_matching_result=true' \
    'evaluation_plan.render_cfg.normal_render_tasks=[]'
```

## SceneAgent Drake Metrics

| Metric | Description |
|--------|-------------|
| `DrakeCollisionMetricSceneAgent` | Detects penetrations using pre-computed CoACD collision geometry |
| `StaticEquilibriumMetricSceneAgent` | Runs physics simulation to measure stability |
| `WeldedEquilibriumMetricSceneAgent` | Welds penetrating objects, then simulates stability |

These metrics differ from the standard Drake metrics (e.g., `DrakeCollisionMetricCoACD`) in that they:
1. Use scene-agent's pre-computed SDFs directly (faster)
2. Apply transforms from scene_X.json instead of using world-frame meshes
3. Include floor_plan.sdf for architecture geometry

## Notes

- Only CoACD metrics are supported (scene-agent uses CoACD, not VHACD)
- `use_empty_matching_result=true` is required since SceneAgent doesn't have reference annotations
- Set `render_cfg.normal_render_tasks=[]` to skip rendering (optional)
- Simulation HTML files are saved when `save_simulation_html=true`
