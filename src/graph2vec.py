"""Graph2Vec module."""

import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from argparse import Namespace
from scipy import spatial
import numpy as np
import scipy
import matplotlib.pyplot as plt


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.if_edge_has_features = False
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing

        if list(self.graph.edges.data())[0][2]:
            print("Adding Edge Features as well...")
            for each_edge in self.graph.edges.data():
                edge_name = each_edge[0]+'_'+each_edge[1]
                edge_features = str(list(each_edge[2].values())[0])
                # edge_features = "_".join(edge_features)
                edge_hash_object = hashlib.md5(edge_features.encode())
                edge_hashing = edge_hash_object.hexdigest()
                new_features[edge_name] = edge_hashing

        self.extracted_features = self.extracted_features + list(new_features.values())
        # print("self.extracted_feratures:", self.extracted_features)
        # print("features:", new_features)
        # print("Edge data:", self.graph.edges.data())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """

    name = path.strip(".gexf").split("/")[-1]

    # name = path.strip(".json").split("/")[-1]
    # data = json.load(open(path))
    # graph = nx.from_edgelist(data["edges"])
    # nx.write_gexf(graph, "1.gexf")
    graph = nx.read_gexf(path)

    # nx.write_gexf(graph, "Graph2VecFinalGraph.gefx")


    # if "features" in data.keys():
    #     features = data["features"]
    # else:
    #     features = nx.degree(graph)
    features = dict(nx.degree(graph))
    features = {k: v for k, v in features.items()}
    return graph, features, name


def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding( model, files):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = f.split("/")[-1].strip(".gexf")

        # identifier = f.split("/")[-1].strip(".json")
        print("g_"+identifier)
        embbedings = model.docvecs["g_"+identifier]
        print(embbedings)
    #     out.append([int(identifier.split("\\")[-1])] + list(model.docvecs["g_"+identifier]))
    # column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    # out = pd.DataFrame(out, columns=column_names)
    # out = out.sort_values(["type"])
    # out.to_csv(output_path, index=None)

def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    # graphs = glob.glob(args.input_path + "*.json")
    graphs = glob.glob(args.input_path+"*.gexf")
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)

    out = {}
    for f in graphs:
        identifier = f.split("/")[-1].strip(".gexf")

        # identifier = f.split("/")[-1].strip(".json")
        # print("g_"+identifier)
        embbedings = model.docvecs["g_"+identifier]
        out[identifier.split('\\')[-1]] = embbedings
        # print(embbedings)

    print(out)

    distance12 = np.linalg.norm(out['10n1'] - out['10n2'])
    distance34 = np.linalg.norm(out['10n_edgeF1'] - out['10n_edgeF2'])
    distance13 = np.linalg.norm(out['10n_edgeF1semi'] - out['10n_edgeF1'])
    distance23 = np.linalg.norm(out['10n_edgeF1full'] - out['10n_edgeF1'])
    # distance23 = np.linalg.norm(out['p10'] - out['p8'])
    # distance = np.linalg.norm(out['c11'] - out['p8'])
    # distance_path_complete = np.linalg.norm(out['p8'] - out['c12'])
    # distance_complete_grid = np.linalg.norm(out['c12'] - out['g16'])

    #

    # distance12 = spatial.distance.cosine(out['4'], out['5'])
    # distance34 = spatial.distance.cosine(out['11'], out['12'])
    # distance13 = spatial.distance.cosine(out['4'], out['12'])
    # distance13 = spatial.distance.cosine(out[0], out[2])
    # distance14 = spatial.distance.cosine(out[0], out[3])
    # distance32 = spatial.distance.cosine(out[2], out[1])
    # distance34 = spatial.distance.cosine(out[2], out[3])

    print("Distance between two no features graphs::", distance12)
    print("Distance between two featured graphs::", distance34)
    print("Distance between feature and semi featured::", distance13)
    print("Distance between  featured and full featured::", distance23)
    # print("Distance between path and Complete graph", distance)
    # print("Distance complete and path", distance_path_complete)
    # print("Distance complete and grid", distance_complete_grid)




    # save_embedding(model, graphs)

def create_graphs():
    G2 = nx.Graph()
    G1 = nx.Graph()
    G3 = nx.Graph()
    G4 = nx.Graph()

    G1.add_nodes_from(
        [(1, {'id': 10}), (2, {'id': 20}), (3, {'id': 30}), (4, {'id': 40}), (5, {'id': 50}), (6, {'id': 60}),
         (7, {'id': 70}), (8, {'id': 80}), (9, {'id': 90}), (10, {'id': 100}), (11, {'id': 110})])
    G2.add_nodes_from(
        [(1, {'id': 10}), (2, {'id': 20}), (3, {'id': 30}), (4, {'id': 40}), (5, {'id': 50}), (6, {'id': 60}),
         (7, {'id': 70}), (8, {'id': 80}), (9, {'id': 90}), (10, {'id': 100}), (11, {'id': 110})])
    G3.add_nodes_from(
        [(1, {'id': 10}), (2, {'id': 20}), (3, {'id': 30}), (4, {'id': 40}), (5, {'id': 50}), (6, {'id': 60}),
         (7, {'id': 70}), (8, {'id': 80}), (9, {'id': 90}), (10, {'id': 100}), (11, {'id': 110})])
    G4.add_nodes_from(
        [(1, {'id': 10}), (2, {'id': 20}), (3, {'id': 30}), (4, {'id': 40}), (5, {'id': 50}), (6, {'id': 60}),
         (7, {'id': 70}), (8, {'id': 80}), (9, {'id': 90}), (10, {'id': 100}), (11, {'id': 110})])

    G1.add_edges_from([(1, 2), (2, 3),(3, 4), (4, 5),(5, 6), (6, 7),(7, 8), (8, 9), (9, 10), (10, 1)])

    G2.add_edges_from([(1, 2, {'route': 20}), (2, 3, {'route': 24})])
    G2.add_edges_from([(3, 4, {'route': 28}), (4, 5, {'route': 32})])
    G2.add_edges_from([(5, 6, {'route': 33}), (6, 7, {'route': 34})])
    G2.add_edges_from([(7, 8, {'route': 35}), (8, 9, {'route': 36})])
    G2.add_edges_from([(9, 10, {'route': 37}), (10, 1, {'route': 38})])

    G3.add_edges_from([(1, 2, {'route': 1}), (2, 3, {'route': 24})])
    G3.add_edges_from([(3, 4, {'route': 208}), (4, 5, {'route': 32})])
    G3.add_edges_from([(5, 6, {'route': 45}), (6, 7, {'route': 34})])
    G3.add_edges_from([(7, 8, {'route': 1000}), (8, 9, {'route': 36})])
    G3.add_edges_from([(9, 10, {'route': 317}), (10, 1, {'route': 38})])

    G4.add_edges_from([(1, 2, {'route': 1}), (2, 3, {'route': 121})])
    G4.add_edges_from([(3, 4, {'route': 208}), (4, 5, {'route': 204})])
    G4.add_edges_from([(5, 6, {'route': 45}), (6, 7, {'route': 0})])
    G4.add_edges_from([(7, 8, {'route': 1000}), (8, 9, {'route': 5})])
    G4.add_edges_from([(9, 10, {'route': 317}), (10, 1, {'route': 987})])

    nx.write_gexf(G1, "../test_gefx/10n1.gexf")
    nx.write_gexf(G1, "../test_gefx/10n2.gexf")

    nx.write_gexf(G2, "../test_gefx/10n_edgeF1.gexf")
    nx.write_gexf(G2, "../test_gefx/10n_edgeF2.gexf")

    nx.write_gexf(G3, "../test_gefx/10n_edgeF1semi.gexf")
    nx.write_gexf(G4, "../test_gefx/10n_edgeF1full.gexf")





if __name__ == "__main__":

    args = Namespace(dimensions=128, down_sampling=10, epochs=1000, input_path=r'D:\WORK\GRAPHS\graph2vec\test_gefx\\', learning_rate=0.0001, min_count=5, output_path=r'D:\WORK\GRAPHS\graph2vec\extracted_features\\', wl_iterations=2, workers=4)
    # create_graphs()
    main(args)
