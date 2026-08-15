import networkx as nx
import matplotlib.pyplot as plt

class AnatomicalLayer:
    def __init__(self, area, depth, num):
        self.area = area
        self.depth = depth
        self.num = num

class Projection:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post

class AnatomicalNet:
    def __init__(self):
        self.layers = []
        self.projections = []

    def find_layer(self, area, depth):
        for l in self.layers:
            if l.area == area and l.depth == depth:
                return l
        return None

    def find_projection(self, source_area, source_depth,
                        target_area, target_depth):
        for proj in self.projections:
            if ( proj.pre.area == source_area and
                 proj.pre.depth == source_depth and
                 proj.post.area == target_area and
                 proj.post.depth == target_depth):
                return proj
        return None

    def add_layer(self, layer):
        assert(isinstance(layer, AnatomicalLayer))
        if self.find_layer(layer.area, layer.depth):
            print("%s %s already exist!"%(layer.area, layer.depth))
            return
        self.layers.append(layer)

    def add_projection(self, projection):
        assert(isinstance(projection, Projection))
        if self.find_projection(projection.pre.area, projection.pre.depth,
                                projection.post.area, projection.post.depth):
            print("Projection %s %s to %s %s already exist!"%(projection.pre.area,
                   projection.pre.depth, projection.post.area, projection.post.depth))
            return
        self.projections.append(projection)

    def make_graph(self):
        G = nx.DiGraph()
        edges = [(p.pre, p.post) for p in self.projections]
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = { layer:layer.area + layer.depth for layer in G.nodes()}
        return G, node_label_dict

    def draw_graph(self, node_size=1600, node_color='yellow', edge_color='red'):
        G, node_label_dict = self.make_graph()
        plt.figure(figsize=(10,10))
        pos = nx.nx_pydot.graphviz_layout(G, prog=r"C:\Program Files (x86)\Graphviz\bin\dot.exe")
        # pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color)
        nx.draw_networkx_labels(G, pos, node_label_dict)
        plt.show()

def gen_anatomy(data, input_depths = ['4'],
            output_depths = ['4', '2/3', '5'],
            laminar_connections = [('4', '2/3'), ('2/3', '5')]):
    """
    generate anatomy structure from data class
    """
    anet = AnatomicalNet()
    areas = data.areas
    depths = data.layers
    output_map = {} # collect output layers for each hierarchy

    for hierarchy in [1,2,3]:
        output_map[hierarchy] = []
        for area in areas:
            if data.get_hierarchical_level(area) == hierarchy:
                # create anatomical module for one area
                # add layers
                area_layers = {}
                for depth in depths:
                    layer = AnatomicalLayer(area, depth, data.neuron_counts[area][depth])
                    area_layers[depth] = layer
                    anet.add_layer(layer)
                    if depth in output_depths:
                        output_map[hierarchy].append(layer)

                # add LaminarProjection
                for source, target in laminar_connections:
                    anet.add_projection(Projection(area_layers[source], area_layers[target]))

                # add AreaProjection
                for depth in depths:
                    if depth in input_depths and not hierarchy == 1:
                        for l in output_map[hierarchy-1]:
                            anet.add_projection(Projection(l, area_layers[depth]))
                        if hierarchy == 3:
                            for l in output_map[hierarchy-2]:
                                anet.add_projection(Projection(l, area_layers[depth]))
    return anet
