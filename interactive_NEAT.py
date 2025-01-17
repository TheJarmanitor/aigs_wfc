from tensorneat.algorithm  import NEAT
from tensorneat.genome import BaseGenome

class InteractiveNEAT(NEAT):
    def __init__(self,
    genome: BaseGenome,
    pop_size: int,
    species_size: int = 10,
    max_stagnation: int = 15,
    species_elitism: int = 2,
    spawn_number_change_rate: float = 0.5,
    genome_elitism: int = 2,
    survival_threshold: float = 0.1,
    min_species_size: int = 1,
    compatibility_threshold: float = 2.0) -> None:
        self.genome = genome
        self.pop_size = pop_size
