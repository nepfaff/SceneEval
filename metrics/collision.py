import trimesh
import numpy as np
from dataclasses import dataclass
from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric

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

        collision_manager = trimesh.collision.CollisionManager()

        # Get non-carpet object IDs (carpets are excluded from collision checks)
        carpet_ids = self.scene.carpet_obj_ids
        non_carpet_obj_ids = [
            obj_id for obj_id in self.scene.get_obj_ids()
            if obj_id not in carpet_ids
        ]

        if carpet_ids:
            print(f"Excluding {len(carpet_ids)} carpet object(s) from collision check: {list(carpet_ids)}")

        # Initialize the collision results (only for non-carpet objects)
        collision_results = {
            obj_id: {
                "in_collision": False,
                "colliding_with": [],
                "collision_details": []  # Store depth/contact info per collision
            }
        for obj_id in non_carpet_obj_ids}

        # For each object, check if it is in collision with any other object
        for i, obj_id in enumerate(non_carpet_obj_ids):
            
            # Add the object to the collision manager
            t_obj = self.scene.t_objs[obj_id]
            collision_manager.add_object(obj_id, t_obj)
            
            # Check for collision with each of the other objects
            for other_obj_id in non_carpet_obj_ids[i+1:]:
                
                # Add the other object to the collision manager and check for collision
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

        result = MetricResult(
            message=f"Scene is in collision: {scene_in_collision}, with {num_obj_in_collision}/{len(non_carpet_obj_ids)} objects in collision. Max depth: {overall_max_depth:.4f}m, {num_collision_pairs} collision pairs. ({len(carpet_ids)} carpets excluded)",
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
            }
        )

        print(f"\n{result.message}\n")

        return result
