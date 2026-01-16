import trimesh
import numpy as np
from dataclasses import dataclass
from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric

# ----------------------------------------------------------------------------------------


def _aabb_overlap(bounds_a: np.ndarray, bounds_b: np.ndarray) -> bool:
    """Check if two axis-aligned bounding boxes overlap.

    Args:
        bounds_a: Bounding box as [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bounds_b: Bounding box as [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    Returns:
        True if the bounding boxes overlap, False otherwise.
    """
    return (
        bounds_a[0, 0] <= bounds_b[1, 0] and bounds_a[1, 0] >= bounds_b[0, 0] and
        bounds_a[0, 1] <= bounds_b[1, 1] and bounds_a[1, 1] >= bounds_b[0, 1] and
        bounds_a[0, 2] <= bounds_b[1, 2] and bounds_a[1, 2] >= bounds_b[0, 2]
    )


# ----------------------------------------------------------------------------------------

@dataclass
class CollisionMetricConfig:
    """
    Configuration for the collision metric.

    Attributes:
        move_direction_amount: the distance to move objects when double-checking collisions
    """

    move_direction_amount: float = 0.005

# ----------------------------------------------------------------------------------------

@register_non_vlm_metric(config_class=CollisionMetricConfig)
class CollisionMetric(BaseMetric):
    """
    Metric to evaluate object collision.
    """

    def __init__(self, scene: Scene, cfg: CollisionMetricConfig, **kwargs) -> None:
        """
        Initialize the metric.

        Args:
            scene: the scene to evaluate
            cfg: the configuration for the metric
        """

        self.scene = scene
        self.cfg = cfg

    def run(self, verbose: bool = False) -> MetricResult:
        """
        Run the metric.

        Args:
            verbose: whether to visualize during the run

        Returns:
            result: the result of running the metric
        """

        # Early return for empty scenes
        if self.scene.is_empty:
            result = MetricResult(
                message="Skipped CollisionMetric: scene has no objects",
                data={
                    "scene_in_collision": False,
                    "num_obj_in_collision": 0,
                    "num_collision_pairs": 0,
                    "max_penetration_depth": 0.0,
                    "mean_penetration_depth": 0.0,
                    "total_contact_points": 0,
                    "collision_results": {},
                    "excluded_carpet_ids": [],
                    "num_excluded_carpets": 0,
                    "excluded_placeholder_ids": [],
                    "num_excluded_placeholders": 0,
                    "skip_reason": "empty_scene",
                }
            )
            print(f"\n{result.message}\n")
            return result

        collision_manager = trimesh.collision.CollisionManager()

        # Get non-carpet object IDs (carpets are excluded from collision checks)
        carpet_ids = self.scene.carpet_obj_ids
        non_carpet_obj_ids = [
            obj_id for obj_id in self.scene.get_obj_ids()
            if obj_id not in carpet_ids
        ]

        # Exclude SceneWeaver placeholder objects (spawn_placeholder, bbox_placeholder)
        placeholder_ids = {
            obj_id for obj_id in self.scene.get_obj_ids()
            if 'placeholder' in obj_id.lower()
        }
        non_carpet_obj_ids = [
            obj_id for obj_id in non_carpet_obj_ids
            if obj_id not in placeholder_ids
        ]

        if carpet_ids:
            print(f"Excluding {len(carpet_ids)} carpet object(s) from collision check: {list(carpet_ids)}")

        if placeholder_ids:
            print(f"Excluding {len(placeholder_ids)} placeholder object(s) from collision check: {list(placeholder_ids)}")

        # Initialize the collision results (only for non-carpet objects)
        collision_results = {
            obj_id: {
                "in_collision": False,
                "colliding_with": [],
                "collision_details": []  # Store depth/contact info per collision
            }
        for obj_id in non_carpet_obj_ids}

        # Pre-compute bounding boxes for broadphase culling
        obj_bounds = {}
        for obj_id in non_carpet_obj_ids:
            t_obj = self.scene.t_objs[obj_id]
            obj_bounds[obj_id] = t_obj.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]

        # Track broadphase statistics
        total_pairs = len(non_carpet_obj_ids) * (len(non_carpet_obj_ids) - 1) // 2
        pairs_checked = 0
        pairs_skipped = 0

        print(f"Broadphase culling enabled: {len(non_carpet_obj_ids)} objects, {total_pairs} potential pairs")

        # For each object, check if it is in collision with any other object
        for i, obj_id in enumerate(non_carpet_obj_ids):

            # Add the object to the collision manager
            t_obj = self.scene.t_objs[obj_id]
            collision_manager.add_object(obj_id, t_obj)
            bounds_a = obj_bounds[obj_id]

            # Check for collision with each of the other objects
            for other_obj_id in non_carpet_obj_ids[i+1:]:

                # Broadphase: skip if AABBs don't overlap
                bounds_b = obj_bounds[other_obj_id]
                if not _aabb_overlap(bounds_a, bounds_b):
                    pairs_skipped += 1
                    continue

                pairs_checked += 1

                # Narrowphase: detailed collision check
                t_other_obj = self.scene.t_objs[other_obj_id]
                in_collision, contact_data = collision_manager.in_collision_single(t_other_obj, return_data=True)
                
                # If in collision, double check by separating the objects slightly and checking again
                if in_collision:
                    
                    # Get the contact point locations
                    contact_pts = np.asarray([contact.point for contact in contact_data])
                    
                    # Move the object slightly away from the other object
                    move_direction = t_other_obj.centroid - np.mean(contact_pts, axis=0)
                    move_direction /= np.linalg.norm(move_direction)
                    moved_t_other_obj = t_other_obj.copy()
                    moved_t_other_obj.apply_translation(move_direction * self.cfg.move_direction_amount)
                    
                    # Check for collision again
                    double_check_in_collision = collision_manager.in_collision_single(moved_t_other_obj)
                    
                    # If still in collision, add the other object to the collision results
                    if double_check_in_collision:
                        # Extract collision severity metrics from contact data
                        depths = [contact.depth for contact in contact_data if hasattr(contact, 'depth')]
                        max_depth = float(max(depths)) if depths else 0.0
                        mean_depth = float(np.mean(depths)) if depths else 0.0
                        num_contacts = len(contact_data)

                        collision_detail = {
                            "other_obj": other_obj_id,
                            "max_depth": max_depth,
                            "mean_depth": mean_depth,
                            "num_contact_points": num_contacts
                        }

                        collision_results[obj_id]["in_collision"] = True
                        collision_results[obj_id]["colliding_with"].append(other_obj_id)
                        collision_results[obj_id]["collision_details"].append(collision_detail)

                        collision_results[other_obj_id]["in_collision"] = True
                        collision_results[other_obj_id]["colliding_with"].append(obj_id)
                        collision_results[other_obj_id]["collision_details"].append({
                            "other_obj": obj_id,
                            "max_depth": max_depth,
                            "mean_depth": mean_depth,
                            "num_contact_points": num_contacts
                        })
                    
                print((
                    f"Checked: {obj_id} and {other_obj_id} - 1st check: {in_collision}, 2nd check: {double_check_in_collision if in_collision else 'N/A'} -> "
                    f"{'Collision - O' if in_collision and double_check_in_collision else 'No Collision - X'}"
                ))
            
            # Remove the object from the collision manager after checking for collision with all other objects
            collision_manager.remove_object(obj_id)

        # Summarize the collision results
        num_obj_in_collision = sum(obj_result["in_collision"] for obj_result in collision_results.values())
        scene_in_collision = num_obj_in_collision > 0

        # Compute aggregate collision severity metrics
        all_max_depths = []
        all_mean_depths = []
        total_contact_points = 0
        num_collision_pairs = 0

        for obj_result in collision_results.values():
            for detail in obj_result["collision_details"]:
                all_max_depths.append(detail["max_depth"])
                all_mean_depths.append(detail["mean_depth"])
                total_contact_points += detail["num_contact_points"]
                num_collision_pairs += 1

        # Each collision pair is counted twice (once per object), so divide by 2
        num_collision_pairs = num_collision_pairs // 2
        total_contact_points = total_contact_points // 2

        overall_max_depth = float(max(all_max_depths)) if all_max_depths else 0.0
        overall_mean_depth = float(np.mean(all_max_depths)) if all_max_depths else 0.0

        # Calculate broadphase efficiency
        broadphase_skip_rate = (pairs_skipped / total_pairs * 100) if total_pairs > 0 else 0.0

        print(f"Broadphase stats: {pairs_skipped}/{total_pairs} pairs skipped ({broadphase_skip_rate:.1f}%), {pairs_checked} pairs checked")

        result = MetricResult(
            message=f"Scene is in collision: {scene_in_collision}, with {num_obj_in_collision}/{len(non_carpet_obj_ids)} objects in collision. Max depth: {overall_max_depth:.4f}m, {num_collision_pairs} collision pairs. ({len(carpet_ids)} carpets, {len(placeholder_ids)} placeholders excluded). Broadphase: {pairs_skipped}/{total_pairs} skipped ({broadphase_skip_rate:.1f}%)",
            data={
                "scene_in_collision": scene_in_collision,
                "num_obj_in_collision": num_obj_in_collision,
                "num_collision_pairs": num_collision_pairs,
                "max_penetration_depth": overall_max_depth,
                "mean_penetration_depth": overall_mean_depth,
                "total_contact_points": total_contact_points,
                "collision_results": collision_results,
                "excluded_carpet_ids": list(carpet_ids),
                "num_excluded_carpets": len(carpet_ids),
                "excluded_placeholder_ids": list(placeholder_ids),
                "num_excluded_placeholders": len(placeholder_ids),
                "broadphase_total_pairs": total_pairs,
                "broadphase_pairs_checked": pairs_checked,
                "broadphase_pairs_skipped": pairs_skipped,
                "broadphase_skip_rate": broadphase_skip_rate,
            }
        )

        print(f"\n{result.message}\n")

        return result
