import cupy as cp
from typing import List
from .plugin_manager import PluginBase
import cupyx.scipy.ndimage as ndimage

def hough_line_transform(image, theta_res=1, rho_res=1):
    # Define the range of theta and rho
    thetas = cp.deg2rad(cp.arange(-90, 90, theta_res))
    width, height = image.shape
    diag_len = cp.ceil(cp.sqrt(width * width + height * height))  # maximum rho
    rhos = cp.linspace(-diag_len, diag_len, cp.ceil(2 * diag_len / rho_res))

    # Initialize accumulator array
    accumulator = cp.zeros((len(rhos), len(thetas)), dtype=cp.uint64)

    # Indices of the non-zero pixels in the binary image
    y_idxs, x_idxs = cp.nonzero(image)

    # Vote in the accumulator array
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)):
            rho = cp.round(x * cp.cos(thetas[j]) + y * cp.sin(thetas[j]))
            rho_idx = cp.argmin(cp.abs(rhos - rho))
            accumulator[rho_idx, j] += 1

    return accumulator, thetas, rhos

class EdgeDetection(PluginBase):
    def __init__(self, input_layer_name, algo, sigma=1, min_h = 0.50, **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.possible_types = ['sobel', 'prewitt', 'laplace', 'gaussian_laplace']
        self.default_algo = "sobel"
        self.flag = 1
        self.sigma = sigma
        self.algo = algo
        self.min_h = min_h
        self.thresh = 0.1

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        
        if self.input_layer_name in layer_names:
            idx = layer_names.index(self.input_layer_name)
            h = elevation_map[idx]
        elif self.input_layer_name in plugin_layer_names:
            idx = plugin_layer_names.index(self.input_layer_name)
            h = plugin_layers[idx]
        else:
            print(
                "layer name {} was not found. Using elevation layer.".format(
                    self.input_layer_name
                )
            )
            h = elevation_map[0]
        if self.algo.lower() not in self.possible_types:
            print(f"Undefined edge detection algo. Defaulting to {self.default_algo}.")
            self.algo = self.default_algo

        if self.algo.lower() == "sobel":
            x, y = ndimage.sobel(h, axis = 0, mode="nearest"), ndimage.sobel(h, axis = 1, mode="nearest")
        elif self.algo.lower() == "prewitt":
            x, y = ndimage.prewitt(h, axis = 0, mode="nearest"), ndimage.prewitt(h, axis = 1, mode="nearest")
        elif self.algo.lower() == "laplace":
            hs1 = cp.absolute(ndimage.laplace(h))
            hs1 /= cp.max(hs1)
            hs1 = (hs1>=self.thresh) * (h>=self.min_h)
            return hs1
        elif self.algo.lower() == "gaussian_laplace":
            hs1 = cp.absolute(ndimage.gaussian_laplace(h, self.sigma))
            hs1 /= cp.max(hs1)
            hs1 = (hs1>=self.thresh) * (h>=self.min_h)
            return hs1
    
        hs1 = cp.sqrt(x**2 + y**2)
        hs1 /= cp.max(hs1)
        hs1 = (hs1>=self.thresh) * (h>=self.min_h)
        return hs1
