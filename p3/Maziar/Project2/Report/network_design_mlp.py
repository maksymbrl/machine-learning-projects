mlp2.new_layer({'input_size': 25, 'number_of_nodes': 20, 'activation_function': 'tanh'})
mlp2.new_layer({'input_size': 20, 'number_of_nodes': 15, 'activation_function': 'sigmoid'})
mlp2.new_layer({'input_size': 15, 'number_of_nodes': 5, 'activation_function': 'sigmoid'})
mlp2.new_layer({'input_size': 5, 'number_of_nodes': 1, 'activation_function': 'tanh'})