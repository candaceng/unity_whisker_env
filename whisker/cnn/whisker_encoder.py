"""Temporal, somatotopic whisker encoder with MouseNet-style connectivity."""

from collections import OrderedDict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anatomy_whisker import gen_anatomy


class ConvParam:
    """Container for an anatomically parameterized projection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        gsh,
        gsw,
        kernel_size,
        padding,
        stride=1,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gsh = float(gsh)
        self.gsw = float(gsw)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.stride = int(stride)


class ConvLayer:
    def __init__(self, source_name, target_name, params, out_shape):
        self.params = params
        self.source_name = source_name
        self.target_name = target_name
        self.out_shape = tuple(out_shape)


class SharedFrameEncoder(nn.Module):
    """Apply the same tactile feature encoder to every whisker and frame."""

    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, N, T, in_dim)
        return self.frame_mlp(x)


def index_whisker_grid(whisker_names):
    """Map the 30 right-whisker names to their anatomical 5 x 7 grid."""
    grid = [
        ["RA0", "RA1", "RA2", "RA3", "RA4", None, None],
        ["RB0", "RB1", "RB2", "RB3", "RB4", None, None],
        ["RC0", "RC1", "RC2", "RC3", "RC4", "RC5", "RC6"],
        ["RD0", "RD1", "RD2", "RD3", "RD4", "RD5", "RD6"],
        [None, "RE1", "RE2", "RE3", "RE4", "RE5", "RE6"],
    ]
    if len(whisker_names) != 30 or len(set(whisker_names)) != 30:
        raise ValueError("whisker_names must contain 30 unique right whiskers")
    whisker_to_index = {whisker: i for i, whisker in enumerate(whisker_names)}
    expected_names = {
        name for row in grid for name in row if name is not None
    }
    if set(whisker_names) != expected_names:
        missing = sorted(expected_names - set(whisker_names))
        unexpected = sorted(set(whisker_names) - expected_names)
        raise ValueError(
            f"Unexpected right-whisker layout; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return [
        [whisker_to_index.get(name) if name else None for name in row]
        for row in grid
    ]


def gridify_time(x, grid_indices):
    """Place a (B, 30, T, C) sequence on the padded 5 x 7 grid."""
    batch, _, time, channels = x.shape
    height = len(grid_indices)
    width = len(grid_indices[0])
    grid = x.new_zeros(batch, time, channels, height, width)
    for row_index, row in enumerate(grid_indices):
        for column_index, whisker_index in enumerate(row):
            if whisker_index is not None:
                grid[:, :, :, row_index, column_index] = x[
                    :, whisker_index, :, :
                ]
    return grid


class AnatomicalConv2d(nn.Conv2d):
    """Conv2d with a fixed, Gaussian-sampled MouseNet connectivity mask."""

    def __init__(self, params, seed):
        super().__init__(
            params.in_channels,
            params.out_channels,
            params.kernel_size,
            stride=params.stride,
            padding=params.padding,
            bias=False,
        )
        radius = params.kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        gaussian = torch.exp(
            -(x_grid.square() + y_grid.square()) / (2 * params.gsw ** 2)
        )
        probability = (params.gsh * gaussian).clamp_(0.0, 1.0)
        probability = probability.expand(
            params.out_channels,
            params.in_channels,
            params.kernel_size,
            params.kernel_size,
        )
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        mask = torch.rand(probability.shape, generator=generator) < probability

        # A sampled projection should not leave a target channel permanently
        # disconnected, especially in the small higher-area layers.
        disconnected = ~mask.flatten(1).any(dim=1)
        if disconnected.any():
            target_indices = disconnected.nonzero(as_tuple=False).flatten()
            source_indices = target_indices.remainder(params.in_channels)
            mask[target_indices, source_indices, radius, radius] = True

        self.register_buffer('connectivity_mask', mask.to(self.weight.dtype))

    @property
    def mask_density(self):
        return float(self.connectivity_mask.mean().item())

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.connectivity_mask,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class WhiskerEncoder(nn.Module):
    """Encode a right-whisker sequence through a MouseNet-style area graph."""

    def __init__(
        self,
        arch,
        num_whiskers=30,
        in_dim=2,
        output_dim=128,
        temporal_hidden_dim=64,
        connectivity_seed=17,
    ):
        super().__init__()
        if num_whiskers != 30:
            raise ValueError(
                "The right-only WhiskerEncoder requires num_whiskers=30"
            )
        if in_dim != 2:
            raise ValueError(
                "The right-only tactile input requires two features: "
                "s and theta_deg"
            )
        if temporal_hidden_dim <= 0:
            raise ValueError("temporal_hidden_dim must be positive")
        if tuple(arch.spatial_shape) != (5, 7):
            raise ValueError(
                "The right-whisker barrel topology requires spatial_shape=(5, 7)"
            )

        self.arch = arch
        self.num_whiskers = num_whiskers
        self.in_dim = in_dim
        self.output_dim = int(output_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.spatial_shape = tuple(arch.spatial_shape)

        self.whisker_names = [
            "RA0", "RA1", "RA2", "RA3", "RA4",
            "RB0", "RB1", "RB2", "RB3", "RB4",
            "RC0", "RC1", "RC2", "RC3", "RC4", "RC5", "RC6",
            "RD0", "RD1", "RD2", "RD3", "RD4", "RD5", "RD6",
            "RE1", "RE2", "RE3", "RE4", "RE5", "RE6",
        ]
        self.grid_indices = index_whisker_grid(self.whisker_names)
        valid_grid = torch.tensor(
            [
                [index is not None for index in row]
                for row in self.grid_indices
            ],
            dtype=torch.float32,
        )
        self.register_buffer(
            'valid_barrel_mask', valid_grid.view(1, 1, *self.spatial_shape)
        )

        l4_channels = arch.get_channels('SSp-bfd', '4')
        self.frame_encoder = SharedFrameEncoder(
            in_dim=in_dim, embed_dim=l4_channels
        )

        self.anet = gen_anatomy(arch)
        self.construct_from_anatomy(
            self.anet, arch, connectivity_seed=connectivity_seed
        )

        l23_channels = self.area_channels['SSp-bfd2/3']
        # Each frame first passes through the anatomical L4 -> L2/3 spatial
        # projection. The same compact GRU is then shared independently across
        # the 35 grid sites, preserving topology without a huge recurrent state.
        self.bfd23_temporal = nn.GRU(
            input_size=l23_channels,
            hidden_size=self.temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_to_bfd23 = nn.Linear(
            self.temporal_hidden_dim, l23_channels
        )
        self.l4_norm = nn.BatchNorm2d(l4_channels)
        self.bfd23_temporal_norm = nn.BatchNorm2d(l23_channels)
        self.output_fc = nn.Linear(
            self.area_channels['VISrl5'], self.output_dim
        )

    @staticmethod
    def _node_name(node):
        return node.area + node.depth

    @staticmethod
    def _projection_seed(source_name, target_name, base_seed):
        text = f'{source_name}->{target_name}'
        offset = sum((index + 1) * ord(char) for index, char in enumerate(text))
        return int(base_seed + offset)

    def construct_from_anatomy(
        self, anet, architecture, connectivity_seed=17
    ):
        self.layers = []
        self.convs = nn.ModuleDict()
        self.area_norms = nn.ModuleDict()
        self.area_size = {}
        self.area_channels = {}

        graph, _ = anet.make_graph()
        self.topological_order = [
            self._node_name(node) for node in nx.topological_sort(graph)
        ]
        self.predecessors = {
            self._node_name(target): [
                self._node_name(source) for source in graph.predecessors(target)
            ]
            for target in graph.nodes
        }

        for node in graph.nodes:
            name = self._node_name(node)
            self.area_size[name] = self.spatial_shape
            self.area_channels[name] = architecture.get_channels(
                node.area, node.depth
            )
            if name != 'SSp-bfd4':
                self.area_norms[name] = nn.BatchNorm2d(
                    self.area_channels[name]
                )

        external_source_counts = {}
        for target in graph.nodes:
            external_source_counts[self._node_name(target)] = sum(
                source.area != target.area
                for source in graph.predecessors(target)
            )

        for source, target in graph.edges:
            source_name = self._node_name(source)
            target_name = self._node_name(target)
            in_channels = self.area_channels[source_name]
            out_channels = self.area_channels[target_name]
            external_count = max(1, external_source_counts[target_name])
            gsw = architecture.get_kernel_width_pixels(
                source.area, source.depth, target.area, target.depth
            )
            gsh = architecture.get_kernel_peak_probability(
                source.area,
                source.depth,
                target.area,
                target.depth,
                external_source_count=external_count,
            )
            kernel_size = architecture.get_kernel_size(
                source.area, source.depth, target.area, target.depth
            )
            padding = architecture.get_padding(
                source.area, source.depth, target.area, target.depth
            )
            params = ConvParam(
                in_channels,
                out_channels,
                gsh,
                gsw,
                kernel_size,
                padding,
            )
            layer = ConvLayer(
                source_name, target_name, params, self.spatial_shape
            )
            self.layers.append(layer)
            key = f'{source_name}_{target_name}'
            self.convs[key] = AnatomicalConv2d(
                params,
                seed=self._projection_seed(
                    source_name, target_name, connectivity_seed
                ),
            )

    def _validate_input(self, whisker_input):
        if whisker_input.ndim != 4:
            raise ValueError(
                "whisker_input must have shape (batch, 30, time, 2)"
            )
        if whisker_input.shape[1] != self.num_whiskers:
            raise ValueError(
                f"Expected {self.num_whiskers} right whiskers, "
                f"received {whisker_input.shape[1]}"
            )
        if whisker_input.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected {self.in_dim} tactile features, "
                f"received {whisker_input.shape[-1]}"
            )

    @staticmethod
    def _shape(tensor):
        return tuple(int(size) for size in tensor.shape)

    def _forward_area_maps(self, whisker_input, return_shapes=False):
        """Run the encoder and return the gridded sequence and area maps."""
        self._validate_input(whisker_input)
        batch, _, time, _ = whisker_input.shape
        height, width = self.spatial_shape
        shapes = OrderedDict()
        shapes['input'] = self._shape(whisker_input)

        frame_features = self.frame_encoder(whisker_input)
        shapes['frame_features'] = self._shape(frame_features)
        grid_sequence = gridify_time(frame_features, self.grid_indices)
        shapes['barrel_grid_sequence'] = self._shape(grid_sequence)

        shapes['SSp-bfd4_sequence'] = self._shape(grid_sequence)

        # Keep a clearly defined static L4 summary for direct anatomical
        # SSp-bfd4 -> SSs4/VISrl4 projections. Temporal order is modeled in
        # L2/3; the direct L4 branches receive the mean across all frames.
        l4_map = self.l4_norm(grid_sequence.mean(dim=1))
        l4_map = F.relu(l4_map) * self.valid_barrel_mask
        shapes['SSp-bfd4_temporal_mean'] = self._shape(l4_map)
        shapes['SSp-bfd4'] = self._shape(l4_map)

        l4_to_l23 = self.convs['SSp-bfd4_SSp-bfd2/3']
        l23_frames = l4_to_l23(
            grid_sequence.reshape(batch * time, -1, height, width)
        )
        l23_frames = F.relu(
            self.area_norms['SSp-bfd2/3'](l23_frames)
        )
        l23_frames = l23_frames * self.valid_barrel_mask
        l23_sequence = l23_frames.view(
            batch, time, -1, height, width
        )
        shapes['SSp-bfd4->SSp-bfd2/3_sequence'] = self._shape(
            l23_sequence
        )
        shapes['SSp-bfd2/3_sequence'] = self._shape(l23_sequence)

        temporal_input = (
            l23_sequence.permute(0, 3, 4, 1, 2)
            .contiguous()
            .view(batch * height * width, time, -1)
        )
        shapes['SSp-bfd2/3_temporal_input_per_grid_site'] = self._shape(
            temporal_input
        )
        temporal_sequence, temporal_hidden = self.bfd23_temporal(
            temporal_input
        )
        shapes['SSp-bfd2/3_gru_sequence_per_grid_site'] = self._shape(
            temporal_sequence
        )
        shapes['SSp-bfd2/3_gru_hidden'] = self._shape(temporal_hidden)

        l23_flat = self.temporal_to_bfd23(temporal_hidden[-1])
        l23_map = (
            l23_flat.view(batch, height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        l23_map = F.relu(self.bfd23_temporal_norm(l23_map))
        l23_map = l23_map * self.valid_barrel_mask
        shapes['SSp-bfd2/3'] = self._shape(l23_map)

        outputs = {
            'SSp-bfd4': l4_map,
            'SSp-bfd2/3': l23_map,
        }
        for target_name in self.topological_order:
            if target_name in outputs:
                continue
            incoming = []
            for source_name in self.predecessors[target_name]:
                key = f'{source_name}_{target_name}'
                projection = self.convs[key](outputs[source_name])
                shapes[f'{source_name}->{target_name}'] = self._shape(
                    projection
                )
                incoming.append(projection)
            area_map = torch.stack(incoming, dim=0).sum(dim=0)
            area_map = F.relu(self.area_norms[target_name](area_map))
            if target_name.startswith('SSp-bfd'):
                area_map = area_map * self.valid_barrel_mask
            outputs[target_name] = area_map
            shapes[target_name] = self._shape(area_map)

        if return_shapes:
            return grid_sequence, outputs, shapes
        return grid_sequence, outputs

    def _project_output(self, outputs, shapes=None):
        visrl5_map = outputs['VISrl5']
        pooled = visrl5_map.mean(dim=(-1, -2))
        embedding = self.output_fc(pooled)
        if shapes is not None:
            shapes['VISrl5_spatial_mean'] = self._shape(pooled)
            shapes['output_embedding'] = self._shape(embedding)
        return embedding

    def encode(self, whisker_input, area='SSp-bfd2/3'):
        """Return a sequence-integrated vector from a named cortical area."""
        _, outputs = self._forward_area_maps(whisker_input)
        if area not in outputs:
            raise ValueError(
                f"Unknown whisker area {area!r}; "
                f"choose from {sorted(outputs)}"
            )
        return outputs[area].mean(dim=(-1, -2))

    def forward_with_shapes(self, whisker_input):
        """Return the embedding plus an ordered trace of tensor shapes."""
        _, outputs, shapes = self._forward_area_maps(
            whisker_input, return_shapes=True
        )
        embedding = self._project_output(outputs, shapes=shapes)
        return embedding, shapes

    def architecture_details(self):
        """Return thesis-ready channel and projection hyperparameters."""
        areas = OrderedDict()
        for name in self.topological_order:
            node = next(
                layer
                for layer in self.anet.layers
                if layer.area + layer.depth == name
            )
            areas[name] = {
                'estimated_neurons': int(node.num),
                'spatial_shape': self.spatial_shape,
                'spatial_positions': int(self.arch.spatial_positions),
                'neurons_per_model_unit': self.arch.neurons_per_model_unit,
                'channels': int(self.area_channels[name]),
            }

        projections = []
        for layer in self.layers:
            params = layer.params
            conv = self.convs[
                f'{layer.source_name}_{layer.target_name}'
            ]
            projections.append({
                'source': layer.source_name,
                'target': layer.target_name,
                'in_channels': params.in_channels,
                'out_channels': params.out_channels,
                'kernel_size': params.kernel_size,
                'padding': params.padding,
                'stride': params.stride,
                'gaussian_sigma_pixels': params.gsw,
                'gaussian_peak_probability': params.gsh,
                'sampled_mask_density': conv.mask_density,
                'allocated_weight_count': int(conv.weight.numel()),
                'active_weight_count': int(
                    conv.connectivity_mask.sum().item()
                ),
            })

        return {
            'input_features_per_whisker': self.in_dim,
            'observed_whiskers': self.num_whiskers,
            'grid_shape': self.spatial_shape,
            'grid_positions': int(self.arch.spatial_positions),
            'padded_grid_positions': int(
                self.arch.spatial_positions - self.num_whiskers
            ),
            'channel_formula': (
                'floor(estimated_neurons / '
                '(grid_positions * neurons_per_model_unit))'
            ),
            'frame_mlp_hidden': 64,
            'temporal_integration_area': 'SSp-bfd2/3',
            'temporal_gru_input': self.area_channels['SSp-bfd2/3'],
            'temporal_gru_hidden': self.temporal_hidden_dim,
            'temporal_gru_layers': 1,
            'direct_l4_temporal_reduction': 'mean',
            'output_dim': self.output_dim,
            'areas': areas,
            'projections': projections,
        }

    def forward(self, whisker_input):
        _, outputs = self._forward_area_maps(whisker_input)
        return self._project_output(outputs)
