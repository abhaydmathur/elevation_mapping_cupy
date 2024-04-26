import cupy as cp
from typing import List
from .plugin_manager import PluginBase


class NameOfYourPlugin(PluginBase):
    def __init__(self, add_value:float=1.0, **kwargs):
        super().__init__()
        self.add_value = float(add_value)

    def __call__(self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_layers: cp.ndarray,
        semantic_layer_names: List[str],
        *args
    )->cp.ndarray:
        # Process maps here
        # You can also use the other plugin's data through plugin_layers.
        new_elevation = elevation_map[0] + self.add_value
        return new_elevation