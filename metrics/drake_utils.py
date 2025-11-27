"""Drake utilities for physics simulation in SceneEval.

This module provides utilities for:
- CoACD and VHACD convex decomposition for collision geometry
- SDF file generation from trimesh objects
- Drake MultibodyPlant creation from SceneEval scenes
- Penetration detection and displacement measurement
"""

import logging
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import coacd
import numpy as np
import trimesh

# Register Drake namespace for SDF files.
ET.register_namespace("drake", "drake.mit.edu")

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    DiscreteContactApproximation,
    LoadModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    ProcessModelDirectives,
    Rgba,
    Role,
    SceneGraph,
    Simulator,
    StartMeshcat,
)

from scenes import Scene

console_logger = logging.getLogger(__name__)


def generate_collision_geometry(
    mesh: trimesh.Trimesh, threshold: float = 0.05, **kwargs
) -> list[trimesh.Trimesh]:
    """Generate convex decomposition collision geometry using CoACD.

    Args:
        mesh: Input mesh to decompose.
        threshold: CoACD approximation threshold (0.01-0.1 typical range).
            Lower = more pieces, higher fidelity. Higher = fewer pieces, simpler.
        **kwargs: Additional kwargs passed to coacd.run_coacd().

    Returns:
        List of convex mesh pieces from the decomposition.
        Falls back to convex hull if CoACD fails.
    """
    console_logger.info(
        f"Generating collision geometry with CoACD (threshold={threshold})"
    )

    try:
        start_time = time.time()

        # Set log level to reduce noise.
        coacd.set_log_level("error")

        # Create CoACD mesh from trimesh.
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

        # Run convex decomposition.
        coacd_result = coacd.run_coacd(coacd_mesh, threshold=threshold, **kwargs)

        # Convert CoACD result to trimesh objects.
        convex_pieces = []
        for vertices, faces in coacd_result:
            piece = trimesh.Trimesh(vertices=vertices, faces=faces)
            convex_pieces.append(piece)

        console_logger.info(
            f"CoACD decomposition complete: {len(convex_pieces)} convex pieces "
            f"generated from {len(mesh.vertices)} vertices, {len(mesh.faces)} faces "
            f"in {time.time() - start_time:.2f} seconds"
        )

        return convex_pieces

    except Exception as e:
        console_logger.warning(
            f"CoACD decomposition failed ({e}), falling back to original mesh. "
            f"Drake will auto-compute convex hull if needed."
        )
        # Fallback to original mesh - Drake handles non-convex meshes by
        # automatically computing convex hull when needed.
        return [mesh.copy()]


def generate_collision_geometry_vhacd(
    mesh: trimesh.Trimesh, **kwargs
) -> list[trimesh.Trimesh]:
    """Generate convex decomposition collision geometry using VHACD.

    Args:
        mesh: Input mesh to decompose.
        **kwargs: Additional kwargs passed to trimesh.convex_decomposition().

    Returns:
        List of convex mesh pieces from the decomposition.
        Falls back to convex hull if VHACD fails.
    """
    console_logger.info("Generating collision geometry with VHACD")

    try:
        start_time = time.time()

        # Use trimesh's convex_decomposition which uses vhacdx.
        convex_pieces = mesh.convex_decomposition(**kwargs)

        # Ensure result is a list.
        if not isinstance(convex_pieces, list):
            convex_pieces = [convex_pieces]

        console_logger.info(
            f"VHACD decomposition complete: {len(convex_pieces)} convex pieces "
            f"generated from {len(mesh.vertices)} vertices, {len(mesh.faces)} faces "
            f"in {time.time() - start_time:.2f} seconds"
        )

        return convex_pieces

    except Exception as e:
        console_logger.warning(
            f"VHACD decomposition failed ({e}), falling back to original mesh. "
            f"Drake will auto-compute convex hull if needed."
        )
        # Fallback to original mesh - Drake handles non-convex meshes by
        # automatically computing convex hull when needed.
        return [mesh.copy()]


def is_mesh_valid_for_hydroelastic(
    mesh: trimesh.Trimesh,
    min_volume: float = 1e-10,
) -> bool:
    """Check if a mesh is valid for hydroelastic contact (heuristic check).

    Filters out obviously degenerate convex pieces by volume.
    Same approach as mesh-to-sim-asset.

    Args:
        mesh: The mesh to validate.
        min_volume: Minimum volume threshold. Pieces smaller than this
            are considered degenerate and will be skipped.

    Returns:
        True if the mesh passes volume check.
    """
    if len(mesh.faces) == 0:
        return False

    # Filter by volume - same approach as mesh-to-sim-asset
    if mesh.volume < min_volume:
        return False

    return True


def test_object_hydroelastic_in_drake(
    mesh_paths: list[Path],
    hydroelastic_modulus: float = 1e6,
) -> bool:
    """Test if an object (all its convex pieces) works with hydroelastic in Drake.

    Actually loads all mesh pieces into Drake with hydroelastic properties and
    tests if a simple simulation step succeeds. This catches degenerate
    tetrahedra that heuristic checks miss, including piece-to-piece interactions.

    Args:
        mesh_paths: List of paths to all OBJ mesh files for this object.
        hydroelastic_modulus: Hydroelastic modulus to use for testing.

    Returns:
        True if the object works with hydroelastic in Drake.
    """
    import shutil

    # Build collision elements for all pieces.
    collision_elements = []
    for i, mesh_path in enumerate(mesh_paths):
        collision_elements.append(f"""
      <collision name="collision_{i}">
        <geometry>
          <mesh>
            <uri>{mesh_path.name}</uri>
            <drake:declare_convex/>
          </mesh>
        </geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>{hydroelastic_modulus:.3e}</drake:hydroelastic_modulus>
        </drake:proximity_properties>
      </collision>""")

    # Create a minimal SDF with all pieces.
    sdf_content = f"""<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="test_model">
    <link name="test_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
{"".join(collision_elements)}
    </link>
  </model>
</sdf>
"""
    # Also need a ground plane with rigid hydroelastic for contact.
    ground_sdf = """<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="ground">
    <link name="ground_link">
      <collision name="collision">
        <geometry>
          <box><size>10 10 0.1</size></box>
        </geometry>
        <pose>0 0 -0.05 0 0 0</pose>
        <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
        </drake:proximity_properties>
      </collision>
    </link>
  </model>
</sdf>
"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write the test SDF.
        test_sdf_path = tmp_path / "test_model.sdf"
        test_sdf_path.write_text(sdf_content)

        # Write the ground SDF.
        ground_sdf_path = tmp_path / "ground.sdf"
        ground_sdf_path.write_text(ground_sdf)

        # Copy all mesh files to the temp directory.
        for mesh_path in mesh_paths:
            shutil.copy(mesh_path, tmp_path / mesh_path.name)

        # Create Drake directives.
        directives_content = f"""
directives:
  - add_model:
      name: ground
      file: file://{ground_sdf_path}
  - add_weld:
      parent: world
      child: ground::ground_link
  - add_model:
      name: test_model
      file: file://{test_sdf_path}
  - add_frame:
      name: test_frame
      X_PF:
        base_frame: world
        translation: [0, 0, 0.5]
  - add_weld:
      parent: test_frame
      child: test_model::test_link
"""
        directives_path = tmp_path / "directives.yaml"
        directives_path.write_text(directives_content)

        try:
            # Build Drake plant.
            builder = DiagramBuilder()
            plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-2)
            plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)

            directives = LoadModelDirectives(str(directives_path))
            ProcessModelDirectives(directives, plant, parser=None)
            plant.Finalize()

            diagram = builder.Build()
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()

            # Try a tiny simulation step - this triggers contact computation.
            simulator.AdvanceTo(0.01)
            return True

        except RuntimeError as e:
            if "Cannot instantiate plane from normal" in str(e):
                return False
            # Re-raise other errors
            raise
        except Exception:
            # Any other error means the object doesn't work
            return False


def add_compliant_proximity_properties_element(
    collision_item: ET.Element,
    hydroelastic_modulus: float,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
) -> ET.Element:
    """Add compliant hydroelastic proximity properties to a collision element.

    Args:
        collision_item: The collision XML element to add properties to.
        hydroelastic_modulus: The hydroelastic modulus (Pa). Higher values = stiffer.
        hunt_crossley_dissipation: Optional Hunt-Crossley dissipation (s/m).
        mu_dynamic: Optional dynamic friction coefficient.
        mu_static: Optional static friction coefficient.

    Returns:
        The proximity properties XML element.
    """
    proximity_item = ET.SubElement(
        collision_item, "{drake.mit.edu}proximity_properties"
    )
    ET.SubElement(proximity_item, "{drake.mit.edu}compliant_hydroelastic")
    modulus_item = ET.SubElement(
        proximity_item, "{drake.mit.edu}hydroelastic_modulus"
    )
    modulus_item.text = f"{hydroelastic_modulus:.3e}"
    if hunt_crossley_dissipation is not None:
        dissipation_item = ET.SubElement(
            proximity_item, "{drake.mit.edu}hunt_crossley_dissipation"
        )
        dissipation_item.text = f"{hunt_crossley_dissipation:.3f}"
    if mu_dynamic is not None:
        mu_dyn_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_dynamic")
        mu_dyn_item.text = f"{mu_dynamic:.3f}"
    if mu_static is not None:
        mu_stat_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_static")
        mu_stat_item.text = f"{mu_static:.3f}"
    return proximity_item


def generate_sdf_from_trimesh(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    name: str,
    use_trimesh_inertia: bool = False,
    density: float = 1000.0,
    coacd_threshold: float = 0.05,
    decomposition_method: Literal["coacd", "vhacd"] = "coacd",
    hydroelastic_modulus: float | None = None,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
) -> Path:
    """Generate Drake SDF file from trimesh mesh.

    Creates an SDF file with convex decomposition collision geometry pieces
    used for both visual and collision geometry. This simplifies the pipeline
    since visual fidelity doesn't matter for physics simulation.

    Args:
        mesh: Input trimesh mesh in local coordinates.
        output_dir: Directory to write SDF and OBJ files.
        name: Name for the asset (used in filenames and model name).
        use_trimesh_inertia: If True, compute mass from volume*density and
            inertia from mesh geometry. If False, use mass=1kg and omit
            inertia (Drake uses defaults).
        density: Density in kg/m³ (only used if use_trimesh_inertia=True).
        coacd_threshold: CoACD approximation threshold (only used for coacd).
        decomposition_method: Convex decomposition method ("coacd" or "vhacd").
        hydroelastic_modulus: If set, adds compliant hydroelastic properties
            with this modulus (Pa). If None, no hydroelastic properties are added.
        hunt_crossley_dissipation: Optional Hunt-Crossley dissipation (s/m).
        mu_dynamic: Optional dynamic friction coefficient.
        mu_static: Optional static friction coefficient.

    Returns:
        Path to the generated SDF file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate collision geometry using specified method.
    if decomposition_method == "vhacd":
        collision_pieces = generate_collision_geometry_vhacd(mesh)
    else:
        collision_pieces = generate_collision_geometry(mesh, threshold=coacd_threshold)

    # Create SDF XML structure.
    sdf = ET.Element("sdf", version="1.7")
    model = ET.SubElement(sdf, "model", name=name)

    # Add single link (simple rigid body).
    link = ET.SubElement(model, "link", name="base_link")

    # Inertial properties.
    inertial = ET.SubElement(link, "inertial")

    if use_trimesh_inertia:
        # Compute mass from volume and density.
        volume = mesh.volume
        if volume <= 0:
            console_logger.warning(
                f"Mesh '{name}' has invalid volume ({volume}). Using default mass=1kg."
            )
            mass = 1.0
            inertia_tensor = None
        else:
            mass = volume * density

            # Get inertia tensor from trimesh (assumes uniform density).
            # trimesh.moment_inertia returns moment of inertia; scale by density.
            inertia_tensor = mesh.moment_inertia * density

            # Validate inertia tensor has positive eigenvalues.
            eigenvalues = np.linalg.eigvals(inertia_tensor)
            if np.any(eigenvalues <= 0):
                console_logger.warning(
                    f"Computed inertia tensor for '{name}' has non-positive eigenvalues "
                    f"[{eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}, {eigenvalues[2]:.3f}]. "
                    f"Using mass={mass:.3f}kg but omitting inertia tensor."
                )
                inertia_tensor = None
    else:
        # Use default mass of 1kg.
        mass = 1.0
        inertia_tensor = None

    mass_elem = ET.SubElement(inertial, "mass")
    mass_elem.text = f"{mass:.6f}"

    # Center of mass pose.
    center_of_mass = mesh.center_mass
    com_pose = ET.SubElement(inertial, "pose")
    com_pose.text = (
        f"{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} "
        f"{center_of_mass[2]:.6f} 0 0 0"
    )

    # Inertia tensor (only include if valid).
    if inertia_tensor is not None:
        inertia = ET.SubElement(inertial, "inertia")
        ET.SubElement(inertia, "ixx").text = f"{inertia_tensor[0, 0]:.6f}"
        ET.SubElement(inertia, "iyy").text = f"{inertia_tensor[1, 1]:.6f}"
        ET.SubElement(inertia, "izz").text = f"{inertia_tensor[2, 2]:.6f}"
        ET.SubElement(inertia, "ixy").text = f"{inertia_tensor[0, 1]:.6f}"
        ET.SubElement(inertia, "ixz").text = f"{inertia_tensor[0, 2]:.6f}"
        ET.SubElement(inertia, "iyz").text = f"{inertia_tensor[1, 2]:.6f}"

    # Visual and collision geometry (using convex decomposition pieces).
    # First save all pieces, then test the whole object in Drake if hydroelastic is requested.
    # If the object fails the test, fall back to point contact for all pieces.
    valid_pieces: list[tuple[int, trimesh.Trimesh]] = []
    skipped_piece_count = 0

    for i, piece in enumerate(collision_pieces):
        # Only filter pieces when hydroelastic is enabled - small/thin pieces can
        # cause degenerate tetrahedra in Drake's hydroelastic pressure field.
        # For point contact, all pieces are valid.
        if hydroelastic_modulus is not None and not is_mesh_valid_for_hydroelastic(piece):
            console_logger.warning(
                f"Skipping piece {i} of '{name}' entirely "
                f"(failed heuristic check: too small or too thin)"
            )
            skipped_piece_count += 1
            continue
        valid_pieces.append((i, piece))

    # Save all valid pieces as OBJ files.
    piece_paths: list[Path] = []
    for piece_idx, (orig_idx, piece) in enumerate(valid_pieces):
        piece_filename = f"{name}_piece_{piece_idx}.obj"
        piece_path = output_dir / piece_filename
        piece.export(piece_path)
        piece_paths.append(piece_path)

    # Determine if the whole object can use hydroelastic by testing in Drake.
    use_hydroelastic_for_object = False
    if hydroelastic_modulus is not None and len(piece_paths) > 0:
        if test_object_hydroelastic_in_drake(piece_paths, hydroelastic_modulus):
            use_hydroelastic_for_object = True
            console_logger.info(
                f"'{name}': {len(piece_paths)} pieces with hydroelastic "
                f"({skipped_piece_count} skipped)"
            )
        else:
            console_logger.warning(
                f"'{name}' failed Drake hydroelastic test, "
                f"using point contact for all {len(piece_paths)} pieces"
            )
    elif hydroelastic_modulus is None and len(piece_paths) > 0:
        console_logger.info(
            f"'{name}': {len(piece_paths)} pieces with point contact "
            f"({skipped_piece_count} skipped)"
        )

    # Add visual and collision elements for each piece.
    for piece_idx, piece_path in enumerate(piece_paths):
        piece_filename = piece_path.name

        # Visual geometry.
        visual = ET.SubElement(link, "visual", name=f"visual_{piece_idx}")
        visual_geom = ET.SubElement(visual, "geometry")
        visual_mesh_elem = ET.SubElement(visual_geom, "mesh")
        visual_uri = ET.SubElement(visual_mesh_elem, "uri")
        visual_uri.text = piece_filename

        # Collision geometry.
        collision = ET.SubElement(link, "collision", name=f"collision_{piece_idx}")
        collision_geom = ET.SubElement(collision, "geometry")
        collision_mesh_elem = ET.SubElement(collision_geom, "mesh")
        collision_uri = ET.SubElement(collision_mesh_elem, "uri")
        collision_uri.text = piece_filename
        # Declare mesh as convex for collision detection.
        ET.SubElement(collision_mesh_elem, "{drake.mit.edu}declare_convex")

        # Add compliant hydroelastic proximity properties only if object passed test.
        if use_hydroelastic_for_object:
            add_compliant_proximity_properties_element(
                collision_item=collision,
                hydroelastic_modulus=hydroelastic_modulus,
                hunt_crossley_dissipation=hunt_crossley_dissipation,
                mu_dynamic=mu_dynamic,
                mu_static=mu_static,
            )

    # Format XML with indentation.
    ET.indent(sdf, space="  ", level=0)

    # Write SDF file.
    sdf_path = output_dir / f"{name}.sdf"
    tree = ET.ElementTree(sdf)
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)

    console_logger.info(f"Generated Drake SDF: {sdf_path}")

    return sdf_path


def generate_floor_sdf(
    t_architecture: dict[str, trimesh.Trimesh],
    output_dir: Path,
    use_rigid_hydroelastic: bool = False,
    wall_expansion: float = 0.5,
    floor_expansion: float = 0.5,
) -> Path:
    """Generate SDF file for floor/architecture using box primitives.

    Uses Drake box primitives for collision geometry instead of mesh geometry.
    This is more robust for hydroelastic contact and avoids degenerate mesh issues.
    Boxes are expanded outward from the room to avoid reducing usable space.

    Args:
        t_architecture: Dictionary of architecture trimesh objects.
        output_dir: Directory to write SDF and OBJ files.
        use_rigid_hydroelastic: If True, adds rigid hydroelastic properties.
        wall_expansion: Amount to expand walls outward from room (meters).
        floor_expansion: Amount to expand floor downward (meters).

    Returns:
        Path to the generated SDF file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create SDF XML structure.
    sdf = ET.Element("sdf", version="1.7")
    model = ET.SubElement(sdf, "model", name="floor_plan")

    # Add single link for all architecture.
    link = ET.SubElement(model, "link", name="base_link")

    # Process each architecture element.
    for arch_id, t_arch in t_architecture.items():
        safe_name = arch_id.replace(" ", "_").lower()

        # Save architecture mesh as OBJ for visual geometry.
        arch_filename = f"floor_{safe_name}.obj"
        arch_path = output_dir / arch_filename
        t_arch.export(arch_path)

        # Visual geometry uses the mesh.
        visual = ET.SubElement(link, "visual", name=f"visual_{safe_name}")
        visual_geom = ET.SubElement(visual, "geometry")
        visual_mesh_elem = ET.SubElement(visual_geom, "mesh")
        visual_uri = ET.SubElement(visual_mesh_elem, "uri")
        visual_uri.text = arch_filename

        # Collision geometry uses box primitive derived from bounding box.
        # This is more robust for Drake than mesh geometry.
        bounds = t_arch.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bbox_min = bounds[0].copy()  # Copy to avoid mutating original
        bbox_max = bounds[1].copy()
        extents = bbox_max - bbox_min

        # Determine expansion direction based on architecture type.
        # Walls expand outward from room, floor expands downward.
        if arch_id.startswith("wall"):
            # Determine which axis is the "thin" wall direction.
            # The thin axis should expand outward (away from room center).
            # Walls typically have one dimension much smaller than the others.
            thin_axis = np.argmin(extents[:2])  # Only consider x, y for thin axis

            # Expand the thin axis outward from room.
            # We need to know which side of the room this wall is on.
            # For simplicity, expand in the direction away from origin (room center).
            wall_center = (bbox_min + bbox_max) / 2
            if thin_axis == 0:  # Thin in x, expand x
                if wall_center[0] > 0:
                    bbox_max[0] += wall_expansion
                else:
                    bbox_min[0] -= wall_expansion
            else:  # Thin in y, expand y
                if wall_center[1] > 0:
                    bbox_max[1] += wall_expansion
                else:
                    bbox_min[1] -= wall_expansion

        elif arch_id.startswith("floor"):
            # Floor expands downward (negative z).
            bbox_min[2] -= floor_expansion

        # Calculate final extents and center after expansion.
        extents = bbox_max - bbox_min
        center = (bbox_min + bbox_max) / 2

        # Collision geometry using box primitive.
        collision = ET.SubElement(link, "collision", name=f"collision_{safe_name}")
        collision_geom = ET.SubElement(collision, "geometry")
        box_elem = ET.SubElement(collision_geom, "box")
        size_elem = ET.SubElement(box_elem, "size")
        size_elem.text = f"{extents[0]:.6f} {extents[1]:.6f} {extents[2]:.6f}"

        # Add pose for the collision box.
        pose_elem = ET.SubElement(collision, "pose")
        pose_elem.text = f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} 0 0 0"

        # Add rigid hydroelastic properties for box collision.
        # Boxes always work with hydroelastic (no degenerate face issues).
        if use_rigid_hydroelastic:
            proximity_item = ET.SubElement(
                collision, "{drake.mit.edu}proximity_properties"
            )
            ET.SubElement(proximity_item, "{drake.mit.edu}rigid_hydroelastic")

    # Format XML with indentation.
    ET.indent(sdf, space="  ", level=0)

    # Write SDF file.
    sdf_path = output_dir / "floor_plan.sdf"
    tree = ET.ElementTree(sdf)
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)

    console_logger.info(f"Generated floor plan SDF: {sdf_path}")

    return sdf_path


def create_drake_plant_from_scene(
    scene: Scene,
    time_step: float = 0.01,
    temp_dir: Path | None = None,
    weld_to_world: list[str] | None = None,
    use_trimesh_inertia: bool = False,
    density: float = 1000.0,
    coacd_threshold: float = 0.05,
    decomposition_method: Literal["coacd", "vhacd"] = "coacd",
    hydroelastic_modulus: float | None = None,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
) -> tuple[DiagramBuilder, MultibodyPlant, SceneGraph, dict[str, str]]:
    """Create Drake plant from SceneEval scene.

    Note: This function uses the world-transformed trimesh objects directly
    (scene.t_objs) and places them at the origin in Drake. This avoids
    dependency on Blender's matrix_world which may be garbage collected.

    Args:
        scene: SceneEval scene to load.
        time_step: Drake simulation time step (use >0 for dynamics, 0 for static).
        temp_dir: Directory for temporary SDF files. If None, uses tempfile.
        weld_to_world: List of object IDs to weld to world (make static).
        use_trimesh_inertia: If True, compute mass/inertia from mesh.
        density: Density in kg/m³ (only used if use_trimesh_inertia=True).
        coacd_threshold: CoACD approximation threshold (only used for coacd).
        decomposition_method: Convex decomposition method ("coacd" or "vhacd").
        hydroelastic_modulus: If set, adds compliant hydroelastic properties
            with this modulus (Pa). If None, no hydroelastic properties are added.
        hunt_crossley_dissipation: Optional Hunt-Crossley dissipation (s/m).
        mu_dynamic: Optional dynamic friction coefficient.
        mu_static: Optional static friction coefficient.

    Returns:
        Tuple of (builder, plant, scene_graph, obj_id_to_model_name).
    """
    start_time = time.time()

    weld_to_world = weld_to_world or []

    # Create temporary directory if needed.
    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
    else:
        temp_dir_obj = None
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate SDF files for all objects.
        # NOTE: We use scene.t_objs directly (world-transformed) and place at origin
        # in Drake. This avoids Blender dependency issues.
        obj_ids = scene.get_obj_ids()
        obj_id_to_model_name = {}

        for obj_id in obj_ids:
            # Skip architecture objects (floor, walls) - they're handled separately.
            if "architecture" in obj_id.lower() or "floor" in obj_id.lower() or "wall" in obj_id.lower():
                console_logger.info(f"Skipping architecture object: {obj_id}")
                continue

            # Get mesh in world coordinates (already transformed in TrimeshScene).
            t_obj_world = scene.t_objs[obj_id]

            # Generate safe model name.
            safe_name = obj_id.replace(" ", "_").replace("-", "_").lower()
            model_name = f"obj_{safe_name}"
            obj_id_to_model_name[obj_id] = model_name

            # Generate SDF using world-transformed mesh.
            # Objects will be placed at origin in Drake since mesh is already positioned.
            generate_sdf_from_trimesh(
                mesh=t_obj_world,
                output_dir=temp_dir,
                name=model_name,
                use_trimesh_inertia=use_trimesh_inertia,
                density=density,
                coacd_threshold=coacd_threshold,
                decomposition_method=decomposition_method,
                hydroelastic_modulus=hydroelastic_modulus,
                hunt_crossley_dissipation=hunt_crossley_dissipation,
                mu_dynamic=mu_dynamic,
                mu_static=mu_static,
            )

        # Generate floor plan SDF (architecture is also already in world coords).
        # Use rigid hydroelastic for floor if objects use hydroelastic.
        generate_floor_sdf(
            scene.t_architecture,
            temp_dir,
            use_rigid_hydroelastic=(hydroelastic_modulus is not None),
        )

        # Build Drake directives YAML.
        # Objects are at world position in their mesh, so use identity transform.
        directives_yaml = _build_drake_directives_world_coords(
            temp_dir=temp_dir,
            obj_id_to_model_name=obj_id_to_model_name,
            weld_to_world=weld_to_world,
        )

        # Write directives to file.
        directives_path = temp_dir / "scene.dmd.yaml"
        with open(directives_path, "w") as f:
            f.write(directives_yaml)

        # Also write visualization-only directives (no welds except floor).
        # This can be used with Drake's model_visualizer to show inertia ellipsoids:
        #   python -m pydrake.visualization.model_visualizer scene_viz.dmd.yaml
        viz_directives_yaml = _build_drake_directives_world_coords(
            temp_dir=temp_dir,
            obj_id_to_model_name=obj_id_to_model_name,
            weld_to_world=[],  # No object welds
            include_welds=False,
        )
        viz_directives_path = temp_dir / "scene_viz.dmd.yaml"
        with open(viz_directives_path, "w") as f:
            f.write(viz_directives_yaml)

        # Create Drake plant.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)

        # Add kLagged contact approximation for stability.
        if time_step > 0.0:
            plant.set_discrete_contact_approximation(
                DiscreteContactApproximation.kLagged
            )

        # Load directives.
        directives = LoadModelDirectives(str(directives_path))
        ProcessModelDirectives(directives, plant, parser=None)

        # Finalize plant.
        plant.Finalize()

        end_time = time.time()
        console_logger.info(
            f"Created Drake plant with {len(obj_ids)} objects in "
            f"{end_time - start_time:.2f} seconds"
        )

        return builder, plant, scene_graph, obj_id_to_model_name

    finally:
        # Clean up temporary directory if we created it.
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def _build_drake_directives_world_coords(
    temp_dir: Path,
    obj_id_to_model_name: dict[str, str],
    weld_to_world: list[str],
    include_welds: bool = True,
) -> str:
    """Build Drake model directives YAML for world-coord meshes.

    Since meshes are already in world coordinates, all objects use identity
    transform (placed at origin with no rotation).

    Args:
        temp_dir: Directory containing SDF files.
        obj_id_to_model_name: Mapping from object ID to model name.
        weld_to_world: List of object IDs to weld to world.
        include_welds: If True, add weld directives for objects in weld_to_world.
            If False, skip all welds except floor (for visualization-only files
            that can be used with Drake's model_visualizer to show inertia).

    Returns:
        YAML string for Drake directives.
    """
    # Ensure absolute path for Drake URIs.
    temp_dir = Path(temp_dir).resolve()

    lines = ["directives:"]

    # Add floor plan (always welded to world - it's static architecture).
    lines.extend([
        "- add_model:",
        f'    name: "floor_plan"',
        f'    file: "file://{temp_dir}/floor_plan.sdf"',
        "- add_weld:",
        '    parent: "world"',
        '    child: "floor_plan::base_link"',
    ])

    # Add each object at origin (mesh is already in world coords).
    for obj_id, model_name in obj_id_to_model_name.items():
        lines.extend([
            "- add_model:",
            f'    name: "{model_name}"',
            f'    file: "file://{temp_dir}/{model_name}.sdf"',
            f"    default_free_body_pose:",
            f"      base_link:",
            f"        translation: [0.0, 0.0, 0.0]",
            f"        rotation: !Rpy {{ deg: [0.0, 0.0, 0.0] }}",
        ])

        # Weld if requested and welds are enabled.
        if include_welds and obj_id in weld_to_world:
            lines.extend([
                "- add_weld:",
                '    parent: "world"',
                f'    child: "{model_name}::base_link"',
            ])

    return "\n".join(lines)


def detect_penetrating_pairs(
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    context,
    threshold: float = 0.001,
    obj_id_to_model_name: dict[str, str] | None = None,
) -> list[tuple[str, str, float]]:
    """Detect penetrating object pairs using Drake's collision queries.

    Args:
        plant: Drake MultibodyPlant.
        scene_graph: Drake SceneGraph.
        context: Drake diagram context.
        threshold: Minimum penetration depth to report (meters).
        obj_id_to_model_name: Mapping from object ID to model name.

    Returns:
        List of (obj_a_id, obj_b_id, penetration_depth) tuples.
    """
    # Get query object.
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    inspector = query_object.inspector()

    # Build reverse mapping from model name to object ID.
    model_name_to_obj_id = {}
    if obj_id_to_model_name:
        model_name_to_obj_id = {v: k for k, v in obj_id_to_model_name.items()}

    # Get all collision geometry IDs.
    geometry_ids = list(inspector.GetAllGeometryIds())
    collision_geometry_ids = [
        gid
        for gid in geometry_ids
        if inspector.GetProximityProperties(gid) is not None
    ]

    console_logger.debug(
        f"Found {len(collision_geometry_ids)} collision geometries"
    )

    # Check all pairs.
    penetrating_pairs = []
    for i, gid_a in enumerate(collision_geometry_ids):
        for gid_b in collision_geometry_ids[i + 1:]:
            try:
                distance_result = query_object.ComputeSignedDistancePairClosestPoints(
                    geometry_id_A=gid_a, geometry_id_B=gid_b
                )

                if distance_result.distance < -threshold:
                    # Map geometry IDs to object IDs.
                    frame_a = inspector.GetFrameId(gid_a)
                    frame_b = inspector.GetFrameId(gid_b)
                    name_a = inspector.GetName(frame_a).split("::")[0]
                    name_b = inspector.GetName(frame_b).split("::")[0]

                    # Convert model names to object IDs.
                    obj_a = model_name_to_obj_id.get(name_a, name_a)
                    obj_b = model_name_to_obj_id.get(name_b, name_b)

                    # Skip self-collisions (different CoACD pieces of same object).
                    if obj_a == obj_b:
                        continue

                    penetration_depth = abs(distance_result.distance)
                    penetrating_pairs.append((obj_a, obj_b, penetration_depth))

            except Exception as e:
                console_logger.debug(
                    f"Collision check failed for geometry pair: {e}"
                )

    # Deduplicate pairs.
    seen = set()
    unique_pairs = []
    for obj_a, obj_b, depth in penetrating_pairs:
        pair_key = tuple(sorted([obj_a, obj_b]))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append((obj_a, obj_b, depth))

    console_logger.info(
        f"Detected {len(unique_pairs)} penetrating pairs with threshold {threshold}m"
    )

    return unique_pairs


def measure_displacement(
    plant: MultibodyPlant,
    initial_context,
    final_context,
    obj_id_to_model_name: dict[str, str],
) -> dict[str, dict]:
    """Measure position and rotation displacement for each object.

    Args:
        plant: Drake MultibodyPlant.
        initial_context: Drake context at initial state.
        final_context: Drake context at final state.
        obj_id_to_model_name: Mapping from object ID to model name.

    Returns:
        Dictionary with per-object displacement data:
        {
            obj_id: {
                "displacement": float,  # meters
                "rotation": float,  # radians
                "initial_position": [x, y, z],
                "final_position": [x, y, z],
            }
        }
    """
    results = {}

    for obj_id, model_name in obj_id_to_model_name.items():
        try:
            # Get body.
            body = plant.GetBodyByName("base_link", plant.GetModelInstanceByName(model_name))

            # Get initial and final poses.
            initial_pose = plant.EvalBodyPoseInWorld(initial_context, body)
            final_pose = plant.EvalBodyPoseInWorld(final_context, body)

            # Compute position displacement.
            initial_pos = initial_pose.translation()
            final_pos = final_pose.translation()
            displacement = np.linalg.norm(final_pos - initial_pos)

            # Compute rotation displacement.
            initial_rot = initial_pose.rotation()
            final_rot = final_pose.rotation()
            # Angle of rotation between initial and final.
            delta_rot = initial_rot.InvertAndCompose(final_rot)
            rotation = delta_rot.ToAngleAxis().angle()

            results[obj_id] = {
                "displacement": float(displacement),
                "rotation": float(rotation),
                "initial_position": initial_pos.tolist(),
                "final_position": final_pos.tolist(),
            }

        except Exception as e:
            console_logger.warning(
                f"Could not measure displacement for '{obj_id}': {e}"
            )
            results[obj_id] = {
                "displacement": float("nan"),
                "rotation": float("nan"),
                "initial_position": [float("nan")] * 3,
                "final_position": [float("nan")] * 3,
            }

    return results


def run_simulation(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    simulation_time: float = 2.0,
    scene_graph: SceneGraph | None = None,
    output_html_path: Path | None = None,
    show_collision_geometry: bool = True,
) -> tuple:
    """Run Drake simulation and return initial and final contexts.

    Args:
        builder: Drake DiagramBuilder (plant must be finalized).
        plant: Drake MultibodyPlant.
        simulation_time: Time to simulate in seconds.
        scene_graph: Drake SceneGraph (required if output_html_path is set).
        output_html_path: If provided, save meshcat visualization to this HTML file.
        show_collision_geometry: If True and output_html_path is set, also show
            collision geometry in semi-transparent red for debugging.

    Returns:
        Tuple of (diagram, initial_context, final_context).
    """
    meshcat = None
    visualizer = None

    # Set up visualization if HTML output is requested.
    if output_html_path is not None:
        if scene_graph is None:
            raise ValueError("scene_graph is required when output_html_path is set")
        meshcat = StartMeshcat()
        meshcat.SetProperty("/Background", "top_color", [1.0, 1.0, 1.0])
        meshcat.SetProperty("/Background", "bottom_color", [1.0, 1.0, 1.0])
        meshcat.SetProperty("/Grid", "visible", False)

        # Add visual geometry visualizer (default).
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

        # Add collision geometry visualizer (semi-transparent red).
        if show_collision_geometry:
            collision_params = MeshcatVisualizerParams()
            collision_params.role = Role.kProximity
            collision_params.prefix = "collision"
            collision_params.default_color = Rgba(1.0, 0.0, 0.0, 0.3)
            MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, meshcat, collision_params
            )

    # Build diagram.
    diagram = builder.Build()

    # Create simulator.
    simulator = Simulator(diagram)

    # Get initial state.
    initial_context = diagram.CreateDefaultContext()
    initial_plant_context = plant.GetMyContextFromRoot(initial_context)

    # Initialize simulator.
    simulator.Initialize()

    # Copy initial state for later comparison.
    initial_state = initial_context.Clone()

    # Start recording if visualizing.
    if visualizer is not None:
        visualizer.StartRecording()

    # Run simulation.
    console_logger.info(f"Running simulation for {simulation_time} seconds")
    start_time = time.time()
    simulator.AdvanceTo(simulation_time)
    end_time = time.time()
    console_logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Stop recording and export HTML if visualizing.
    if visualizer is not None and meshcat is not None:
        visualizer.StopRecording()
        visualizer.PublishRecording()

        html = meshcat.StaticHtml()
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html_path, "w") as f:
            f.write(html)
        console_logger.info(f"Saved simulation HTML to {output_html_path}")

    # Get final context.
    final_context = simulator.get_context()

    return diagram, initial_state, final_context
