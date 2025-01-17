from tensorneat.algorithm import NEAT
from tensorneat.genome import BaseGenome
from interactive_species import InteractiveSpeciesController


class InteractiveNEAT(NEAT):
    def __init__(
        self,
        genome: BaseGenome,
        pop_size: int,
    ) -> None:
        self.genome = genome
        self.pop_size = pop_size
        self.species_controller = InteractiveSpeciesController(pop_size)

    def tell(self, state, selected_indices):
        state = state.update(generation=state.generation + 1)

        state, winner, loser, elite_mask = self.species_controller.update_species(
            state, selected_indices
        )
