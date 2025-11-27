from dataclasses import dataclass

@dataclass
class SceneConfig:
    """
    Configuration for loading scenes.

    Attributes:
        skip_missing_obj: whether to skip missing object files peacefully when loading objects
        simple_arch_wall_height: the height of the wall if simple architectural elements are used
        use_simple_architecture: whether to use a simple architecture rather than the scene state's architecture
        use_converted_architecture: whether to load converted architecture GLBs (e.g., from SceneWeaver) if available
    """

    skip_missing_obj: bool = False
    simple_arch_wall_height: float = 3
    use_simple_architecture: bool = False
    use_converted_architecture: bool = False
