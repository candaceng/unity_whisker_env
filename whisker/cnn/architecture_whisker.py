"""Anatomical parameter calculations for the whisker MouseNet branch."""

import math

import numpy as np


class WhiskerArchitecture:
    """Translate estimated neuron counts into a tractable convolutional model.

    MouseNet assigns channels from the neuron count represented at every spatial
    feature-map location.  A literal one-model-unit-per-neuron mapping would
    require thousands of channels on the small barrel grid, so this branch uses
    one model activation for ``neurons_per_model_unit`` estimated neurons.  The
    same compression is applied to every area and layer, preserving relative
    neuron-count differences.
    """

    def __init__(
        self,
        neurons_per_model_unit=20,
        spatial_shape=(5, 7),
        in_degree=1000,
        kernel_extent_sigma=1.0,
        neurons_per_channel=None,
    ):
        if neurons_per_channel is not None:
            if neurons_per_model_unit != 20:
                raise ValueError(
                    "Specify only one of neurons_per_model_unit and the "
                    "legacy neurons_per_channel alias"
                )
            neurons_per_model_unit = neurons_per_channel
        if neurons_per_model_unit <= 0:
            raise ValueError("neurons_per_model_unit must be positive")
        if len(spatial_shape) != 2 or any(int(v) <= 0 for v in spatial_shape):
            raise ValueError("spatial_shape must contain two positive integers")
        if in_degree <= 0:
            raise ValueError("in_degree must be positive")
        if kernel_extent_sigma <= 0:
            raise ValueError("kernel_extent_sigma must be positive")

        self.neurons_per_model_unit = float(neurons_per_model_unit)
        # Backward-compatible attribute name used in early notebooks.
        self.neurons_per_channel = self.neurons_per_model_unit
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.in_degree = float(in_degree)
        self.kernel_extent_sigma = float(kernel_extent_sigma)

        self.neuron_counts = {
            'SSp-bfd': {'2/3': 175481, '4': 135943, '5': 117751},
            'SSs': {'2/3': 95958, '4': 48397, '5': 65758},
            'VISrl': {'2/3': 11831, '4': 4927, '5': 8161},
        }
        self.areas = ['SSp-bfd', 'SSs', 'VISrl']
        self.layers = ['2/3', '4', '5']

        self.d_w = {
            'SSp-bfd4 --> SSp-bfd2/3': 333.45187201506764,
            'SSp-bfd2/3 --> SSp-bfd5': 360.9601087973823,
            'SSs4 --> SSs2/3': 669.0915520825824,
            'SSs2/3 --> SSs5': 685.6151811397214,
            'SSp-bfd2/3 --> SSs4': 378.79212975098477,
            'SSp-bfd4 --> SSs4': 371.80719034383293,
            'SSp-bfd5 --> SSs4': 388.33602475942547,
            'VISrl4 --> VISrl2/3': 103.80973051521892,
            'VISrl2/3 --> VISrl5': 85.9964158059525,
            'SSs2/3 --> VISrl4': 676.663926314319,
            'SSs4 --> VISrl4': 634.205923982863,
            'SSs5 --> VISrl4': 672.7665540952046,
            'SSp-bfd2/3 --> VISrl4': 375.60246193309837,
            'SSp-bfd4 --> VISrl4': 375.3593497233801,
            'SSp-bfd5 --> VISrl4': 397.6682507518624,
        }

        # estimate_area_surface_mm2() in util.py
        self.surface_areas = {
            'SSp-bfd': 9.289074661919935,
            'SSs': 16.87476087000696,
            'VISrl': 3.0787379412331273,
        }

    @property
    def spatial_positions(self):
        return self.spatial_shape[0] * self.spatial_shape[1]

    def get_hierarchical_level(self, area):
        return {'SSp-bfd': 1, 'SSs': 2, 'VISrl': 3}[area]

    def get_channels(self, area, layer):
        """Return channels derived from neurons, map sites, and compression.

        The uncompressed MouseNet rule is ``neurons / (height * width)``.
        Dividing once more by ``neurons_per_model_unit`` makes that rule
        tractable without discarding its relative anatomical scaling.
        """
        neurons = self.neuron_counts[area][layer]
        channels = math.floor(
            neurons / (self.spatial_positions * self.neurons_per_model_unit)
        )
        return max(1, channels)

    def get_pixels_per_micrometer(self, area, layer):
        """Return an isotropic pixel-density approximation for an area.

        Gaussian kernels need one scalar spatial scale.  For a rectangular
        map, ``sqrt(H * W)`` is its equivalent linear resolution and
        ``sqrt(surface area)`` is its equivalent linear cortical extent.
        This is the rectangular counterpart of MouseNet's original
        ``sqrt(neurons / channels)`` calculation.
        """
        del layer  # all layers in an area use the same somatotopic grid
        linear_pixels = math.sqrt(self.spatial_positions)
        linear_micrometers = math.sqrt(self.surface_areas[area]) * 1000.0
        return linear_pixels / linear_micrometers

    def get_connection_probability(self, source, target, area=None):
        """Return the layer-to-layer connection probability."""
        layers = ['2/3', '4', '5', '6']

        # MouseNet probabilities for visual cortex.
        prob_vis = [
            [0.160, 0.016, 0.083, 0.000],
            [0.140, 0.243, 0.104, 0.032],
            [0.021, 0.007, 0.116, 0.047],
            [0.000, 0.000, 0.012, 0.026],
        ]

        # S1 barrel cortex (Lefort 2009 C2 column), collapsed to four layers.
        prob_s1 = [
            [0.134, 0.137, 0.021, 0.000],
            [0.019, 0.243, 0.007, 0.000],
            [0.093, 0.095, 0.081, 0.014],
            [0.000, 0.009, 0.055, 0.028],
        ]

        probabilities = prob_vis if area.startswith('VIS') else prob_s1
        return probabilities[layers.index(source)][layers.index(target)]

    def get_hit_rate_width(self, source_layer, target_layer):
        """Width (micrometers) of the interlaminar connection profile."""
        l4_to_l4 = 114
        cat = {
            '2/3': {'2/3': 225, '4': 50, '5': 100, '6': 50},
            '4': {'2/3': 220, '4': 180, '5': 140, '6': 110},
            '5': {'2/3': 150, '4': 100, '5': 210, '6': 125},
            '6': {'2/3': 120, '4': 20, '5': 150, '6': 150},
        }
        return cat[source_layer][target_layer] / cat['4']['4'] * l4_to_l4

    def get_hit_rate_peak(self, source_layer, target_layer, area):
        hit_rate = self.get_connection_probability(
            source_layer, target_layer, area=area
        )
        offset = 75 if area.startswith('VIS') else 0
        width = self.get_hit_rate_width(source_layer, target_layer)
        fraction_of_peak = np.exp(-(offset ** 2) / (2 * width ** 2))
        return float(hit_rate / fraction_of_peak)

    def get_kernel_width_pixels(
        self, source_area, source_layer, target_area, target_layer
    ):
        pixels_per_um = self.get_pixels_per_micrometer(
            source_area, source_layer
        )
        if source_area == target_area:
            width_um = self.get_hit_rate_width(source_layer, target_layer)
        else:
            key = (
                f'{source_area}{source_layer} --> '
                f'{target_area}{target_layer}'
            )
            width_um = self.d_w[key]
        return float(width_um * pixels_per_um)

    def get_kernel_peak_probability(
        self,
        source_area,
        source_layer,
        target_area,
        target_layer,
        external_source_count=1,
    ):
        if source_area == target_area:
            return self.get_hit_rate_peak(
                source_layer, target_layer, source_area
            )

        if external_source_count <= 0:
            raise ValueError("external_source_count must be positive")

        d_w = self.get_kernel_width_pixels(
            source_area, source_layer, target_area, target_layer
        )
        source_channels = self.get_channels(source_area, source_layer)
        kernel_size = self.get_kernel_size(
            source_area, source_layer, target_area, target_layer
        )
        radius = kernel_size // 2
        coords = np.arange(-radius, radius + 1)
        x_grid, y_grid = np.meshgrid(coords, coords)
        gaussian = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * d_w ** 2))

        # Compress the biological in-degree by the same global factor as the
        # activation units.  Share it across projections converging on the
        # same target instead of assigning the full in-degree to every edge.
        modeled_in_degree = (
            self.in_degree
            / self.neurons_per_model_unit
            / external_source_count
        )
        peak = modeled_in_degree / (source_channels * gaussian.sum())
        return float(min(1.0, peak))

    def get_kernel_size(
        self, source_area, source_layer, target_area, target_layer
    ):
        d_w = self.get_kernel_width_pixels(
            source_area, source_layer, target_area, target_layer
        )
        # ceil (rather than floor) retains at least the nearest-neighbour
        # support on this deliberately small 5 x 7 map.
        radius = max(1, math.ceil(self.kernel_extent_sigma * d_w))
        return 2 * radius + 1

    def get_padding(self, source_area, source_layer, target_area, target_layer):
        kernel_size = self.get_kernel_size(
            source_area, source_layer, target_area, target_layer
        )
        return kernel_size // 2
