from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, problem, common

from tools.visualize_labeled import visualize_labeled, network_dict

possible_activations = [common.ACT.sigmoid, common.ACT.tanh, common.ACT.relu]
activation_labels = ["SGMD", "TANH", "RELU"]

algorithmm = algorithm.NEAT(
    pop_size=500,
    species_size=10,
    survival_threshold=0.01,
    genome=genome.DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        node_gene=genome.BiasNode(
            activation_options=possible_activations,

        )
    ),
)

problemm = problem.XOR3d()


pipeline = Pipeline(
    algorithmm,
    problemm,
    generation_limit=10,
    fitness_target=-1e-6,
    seed=2,
)
state = pipeline.setup()
# run until termination
state, best = pipeline.auto_run(state)
# show results
pipeline.show(state, best)


network = network_dict(algorithmm.genome, state, *best)
visualize_labeled(algorithmm.genome,network,activation_labels, rotate=90, save_path="network.svg", with_labels=True)