#!/usr/bin/env python3
"""Debug mask generation for scene_1."""

import cv2
import trimesh
import numpy as np
from matplotlib import pyplot as plt
import pathlib

# Config values from metrics.yaml
image_resolution = 256
half_image_resolution = image_resolution // 2
scale_margin = 0.0  # From config
floor_color = [255, 0, 0]
obj_color = [0, 255, 0]

# Load the trimesh scene
glb_path = pathlib.Path("output_eval/IDesign/scene_1/trimesh_scene.glb")
t_scene = trimesh.load(str(glb_path))

# Get floor
floor = t_scene.geometry['floor_0']
print(f"Floor bounds: {floor.bounds}")
print(f"Floor extents: {floor.extents}")
print(f"Floor vertices shape: {floor.vertices.shape}")
print(f"Floor faces shape: {floor.faces.shape}")

# Calculate floor center and scale (matching accessibility.py)
t_floor_center = floor.bounds[0] + floor.extents / 2
print(f"\nFloor center: {t_floor_center}")

floor_vertices = floor.vertices - t_floor_center
denormed_floor_vertices = floor_vertices[:, :2]
scale = np.max(np.abs(denormed_floor_vertices)) + scale_margin
print(f"Scale: {scale}")
print(f"Denormed floor vertices range X: [{denormed_floor_vertices[:, 0].min():.3f}, {denormed_floor_vertices[:, 0].max():.3f}]")
print(f"Denormed floor vertices range Y: [{denormed_floor_vertices[:, 1].min():.3f}, {denormed_floor_vertices[:, 1].max():.3f}]")

# Scene to image coordinate conversion
def scene_to_image_coordinates(scene_x, scene_y):
    scene_y = -scene_y
    x_image = int(scene_x / scale * half_image_resolution) + half_image_resolution
    y_image = int(scene_y / scale * half_image_resolution) + half_image_resolution
    return x_image, y_image

# Create mask
mask = np.zeros((image_resolution, image_resolution, 3), dtype=np.uint8)

# Center floor vertices at origin
floor_vertices_2d = floor_vertices[:, :2]

print(f"\nDrawing {floor.faces.shape[0]} floor faces...")
for i, face in enumerate(floor.faces):
    face_vertices = floor_vertices_2d[face]
    face_vertices_image = [scene_to_image_coordinates(x, y) for (x, y) in face_vertices]
    
    if i < 3:  # Show first 3 faces
        print(f"  Face {i}:")
        print(f"    Scene coords: {face_vertices}")
        print(f"    Image coords: {face_vertices_image}")
    
    pts = np.array(face_vertices_image, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], floor_color)

# Count red pixels
red_pixels = np.sum(np.all(mask == floor_color, axis=-1))
total_pixels = image_resolution * image_resolution
print(f"\nRed (floor) pixels: {red_pixels} / {total_pixels} ({100 * red_pixels / total_pixels:.1f}%)")

if red_pixels == 0:
    print("\n❌ BUG: No floor pixels drawn!")
else:
    print("\n✓ Floor drawn successfully")

# Save debug mask
plt.figure(figsize=(8, 8))
plt.title("Debug Floor Mask")
plt.imshow(mask[:, :, ::-1])
plt.savefig("debug_scene1_floor_mask.png")
print(f"\nSaved: debug_scene1_floor_mask.png")
