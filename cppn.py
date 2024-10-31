from jax import random, nn, lax, jit, value_and_grad, vmap
from jaxtyping import Array
import jax.numpy as jnp
import typing

class CPPN:
    '''
    Compositional Pattern Producing Network (CPPN).
    '''
    class Connection:
        '''
        Connection between 2 nodes in the CPPN.
        Is defined by the input node, output node, weight and if it is enabled.
        '''
        def __init__(self, input_node: int, output_node: int, weight: float, enabled: bool = True):
            self.input_node = input_node
            self.output_node = output_node
            self.weight = weight
            self.enabled = enabled
    
    # TODO: Implement locking, which will precalculate connection or create layers and while the node is locked,
    #       it will used a precalculated version (faster multiple evaluations).
    #       mutating this should unlock the node and new locking/calcs should be done before the next evaluation of NN
    class Node:
        '''
        Node in the CPPN.
        Is defined by the id, bias, activation function and input value.
        '''
        def __init__(self, id: int, bias: float, activation_fn: typing.Callable[[float], float]):
            self.id = id
            self.bias = bias
            self.activation_fn = activation_fn

        def pass_init(self, input_connections: int, values: typing.List[float], awaited_connections: typing.List[int]):
            values[self.id] = self.bias
            awaited_connections[self.id] = input_connections
            
        def add_value(self, value: float, values: typing.List[float], awaited_connections: typing.List[int]):
            values[self.id] += value
            awaited_connections[self.id] -= 1
            #if(awaited_connections[self.id] <= 0):
            #    print(f"Node {self.id} is locked with value {values[self.id]} -> {self.activation_fn(values[self.id])}")

        def set_value(self, value: float, values: typing.List[float], awaited_connections: typing.List[int]):
            values[self.id] = value
            awaited_connections[self.id] = 0

        def locked(self, awaited_connections: typing.List[int]) -> bool:
            return awaited_connections[self.id] == 0
        
        def evaluate(self,values: typing.List[float]) -> float:
            return self.activation_fn(values[self.id])
    
    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.connections = []

        
        self.input_nodes = [CPPN.Node(i, 0, lambda x: x) for i in range(input_dim)]
        self.output_nodes = [CPPN.Node(i+len(self.input_nodes), 0, lambda x: x) for i in range(output_dim)]
        self.nodes = self.input_nodes + self.output_nodes

        #for now, fully connected input to output will suffice
        rng = random.PRNGKey(seed)
        for input_node in self.input_nodes:
            for output_node in self.output_nodes:
                rng, key = random.split(rng)
                self.connections.append(CPPN.Connection(input_node.id, output_node.id, random.normal(key, ())))
    
    def forward_pass(self, inputs: Array) -> Array:
        #initialize values and awaited_connections
        values = [0 for _ in self.nodes]
        awaited_connections = [0 for _ in self.nodes]
        for node in self.nodes:
            node.pass_init(sum([1 for connection in self.connections if connection.output_node == node.id]), values, awaited_connections)

        for input_node, value in zip(self.input_nodes, inputs):
            input_node.set_value(value, values, awaited_connections)


        # TODO: Instead of this, a layers or sequence will be created and we will iterate over that
        connections_to_evaluate = [connection for connection in self.connections]
        i = 0
        while len(connections_to_evaluate) != 0:
            connection = connections_to_evaluate[i]
            if self.nodes[connection.input_node].locked(awaited_connections):
                self.nodes[connection.output_node].add_value(self.nodes[connection.input_node].evaluate(values) * connection.weight, values, awaited_connections)
                connections_to_evaluate.remove(connection)
            else:
                i += 1
                if i >= len(connections_to_evaluate):
                    i = 0
        return jnp.array([node.evaluate(values) for node in self.output_nodes])


    def add_node_on_connection(self, connection: Connection, activation_fn: typing.Callable[[float], float], key):
        key1, key2, key3 = random.split(key, 3)
        new_node = CPPN.Node(len(self.nodes), random.normal(key1), activation_fn)
        self.nodes.append(new_node)
        self.connections.remove(connection)
        self.connections.append(CPPN.Connection(connection.input_node, new_node.id, random.normal(key2)))
        self.connections.append(CPPN.Connection(new_node.id, connection.output_node, random.normal(key3)))


    def describe(self):
        print(f"CPPN with {len(self.input_nodes)} input nodes and {len(self.output_nodes)} output nodes.")
        print(f"Hidden nodes: {len(self.nodes) - len(self.input_nodes) - len(self.output_nodes)}")
        print(f"Connections: {len(self.connections)}")
        for connection in self.connections:
            print(f"Connection from {connection.input_node} to {connection.output_node} with weight {connection.weight}")


if __name__ == "__main__":
    cppn = CPPN(2, 2)
    #possible mutation showcase
    cppn.add_node_on_connection(cppn.connections[0], nn.relu, random.PRNGKey(1))
    cppn.describe()
    print("----------")
    jitted_forward_pass = jit(cppn.forward_pass)
    print(jitted_forward_pass(jnp.array([1, 2])))

    # Output:
    #   CPPN with 2 input nodes and 2 output nodes.
    #   Hidden nodes: 1
    #   Connections: 5
    #   Connection from 0 to 3 with weight -0.19947023689746857
    #   Connection from 1 to 2 with weight -2.298278331756592
    #   Connection from 1 to 3 with weight 0.44459623098373413
    #   Connection from 0 to 4 with weight -0.2493748515844345
    #   Connection from 4 to 2 with weight 1.096184492111206
    #   ----------
    #   Node 3 is locked with value 0.6897222399711609 -> 0.6897222399711609
    #   Node 4 is locked with value -0.3775901198387146 -> 0.0
    #   Node 2 is locked with value -4.596556663513184 -> -4.596556663513184
    #   [-4.5965567   0.68972224]
    