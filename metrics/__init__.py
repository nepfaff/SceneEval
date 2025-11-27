from .base import BaseMetric, MetricResult
from .obj_matching import ObjMatching, ObjMatchingResults
from .obj_count import ObjCountMetric
from .obj_attribute import ObjAttributeMetric
from .obj_obj_relationship import ObjObjRelationshipMetric
from .obj_arch_relationship import ObjArchRelationshipMetric
from .collision import CollisionMetric, CollisionMetricConfig
from .support import SupportMetric, SupportMetricConfig
from .navigability import NavigabilityMetric, NavigabilityMetricConfig
from .accessibility import AccessibilityMetric, AccessibilityMetricConfig
from .out_of_bound import OutOfBoundMetric, OutOfBoundMetricConfig
from .opening_clearance import OpeningClearanceMetric, OpeningClearanceMetricConfig
from .drake_collision import (
    DrakeCollisionMetricCoACD,
    DrakeCollisionMetricConfigCoACD,
    DrakeCollisionMetricVHACD,
    DrakeCollisionMetricConfigVHACD,
)
from .static_equilibrium import (
    StaticEquilibriumMetricCoACD,
    StaticEquilibriumMetricConfigCoACD,
    StaticEquilibriumMetricVHACD,
    StaticEquilibriumMetricConfigVHACD,
)
from .welded_equilibrium import (
    WeldedEquilibriumMetricCoACD,
    WeldedEquilibriumMetricConfigCoACD,
    WeldedEquilibriumMetricVHACD,
    WeldedEquilibriumMetricConfigVHACD,
)

from .registry import MetricRegistry, register_non_vlm_metric, register_vlm_metric

__all__ = [
    "BaseMetric",
    "MetricResult",
    "ObjMatching",
    "ObjMatchingResults",
    "ObjCountMetric",
    "ObjAttributeMetric",
    "ObjObjRelationshipMetric",
    "ObjArchRelationshipMetric",
    "CollisionMetric",
    "CollisionMetricConfig",
    "SupportMetric",
    "SupportMetricConfig",
    "NavigabilityMetric",
    "NavigabilityMetricConfig",
    "AccessibilityMetric",
    "AccessibilityMetricConfig",
    "OutOfBoundMetric",
    "OutOfBoundMetricConfig",
    "OpeningClearanceMetric",
    "OpeningClearanceMetricConfig",
    # Drake collision metrics (CoACD and VHACD variants)
    "DrakeCollisionMetricCoACD",
    "DrakeCollisionMetricConfigCoACD",
    "DrakeCollisionMetricVHACD",
    "DrakeCollisionMetricConfigVHACD",
    # Static equilibrium metrics (CoACD and VHACD variants)
    "StaticEquilibriumMetricCoACD",
    "StaticEquilibriumMetricConfigCoACD",
    "StaticEquilibriumMetricVHACD",
    "StaticEquilibriumMetricConfigVHACD",
    # Welded equilibrium metrics (CoACD and VHACD variants)
    "WeldedEquilibriumMetricCoACD",
    "WeldedEquilibriumMetricConfigCoACD",
    "WeldedEquilibriumMetricVHACD",
    "WeldedEquilibriumMetricConfigVHACD",

    # Registry components
    "MetricRegistry",
    "register_non_vlm_metric",
    "register_vlm_metric",
]
