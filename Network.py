import numpy as np

class Network:
    def __str__(self):
        return "<object> RNN neural network for language model"

    def __init__(self, params):
        self.input_size = params["dimensions"][0]
        self.hidden_size = params["dimensions"][1]
        self.output_size = params["dimensions"][2]
        self.w_first = np.random.random([self.hidden_size,self.input_size]) 
        self.w_second = np.random.random([self.output_size,self.hidden_size]) 
        self.w_reverse = np.random.random([self.hidden_size,self.hidden_size])
    def sofmax(self, vector):
        return np.exp(vector)/np.sum(np.exp(vector))
    def forward(self, input_array_of_words):
        #need for loop
        prev_state = np.zeros([self.hidden_size,])
        states = []
        input_vectorized = np.zeros([len(input_array_of_words), len(input_array_of_words)])
        index = 0
        for i in input_array_of_words:
            if i!=0:
                input_vectorized[index][i-1] = 1
            index += 1
        
        #doing the recurrent 
        for vector in input_vectorized:
            Z1 = np.dot(self.w_first, vector) + np.dot(self.w_reverse, prev_state)
            state = np.tanh(Z1)
            prev_state = state
            states.append(state)
            
        Z2 = np.dot(self.w_second, state)
        outputs = self.sofmax(Z2)

        return states, outputs
    def coast(self, true_label, prediction):
        return -np.sum(true_label*np.log(prediction))

params = {
    "dimensions": [3, 2, 3]
}

network = Network(params)
print(network.w_first.shape, network.w_second.shape, network.w_reverse.shape)

state, output = network.forward([2, 0, 1])
print ("cost", network.coast([1,0,1], output))
print(state)
print(output)
