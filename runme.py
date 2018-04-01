import math;

class NeurosNetwork:

    values = [[0,0],[0,1],[1,0],[1,1]]
    #values = [[1,0]]

    weights = {}
    out_layer = 0
    '''neural_network = [
        [False, False],
        [[0.45,-0.12],[0.78,0.13]],
        [[1.5,-2.3]]
    ]'''
    neural_network = [
        [False, False],
        [[2.1,-1],[0.5,2],[0.2,1.1]],
        [[1.5,-1.5,1.2],[1.5,1.5,1.3],[0.7,3,2.1]],
        [[1.5,-5.3,1.7]]
    ]
    N_inputs = {}
    N_outputs = {}
    N_delta = {}
    W_gradients = {}
    W_difs = {}
    
    # Кол-во эпох
    count_epoch = 15000
    
    #Все результаты за эпоху
    results = []
    
    E = 0.7 #Скорость обучения
    A = 0.3 #Момент

    def __init__(self):
        self.initDifWeights()
        self.out_layer = len(self.neural_network)
        #self.initNeuronsLayers()
        #print(self.weights)
        while self.count_epoch > 0:
            self.epoch(self.values)
            self.count_epoch -= 1
            
        
    # Выставляем разницу изменения веса на сете в 0
    def initDifWeights(self):
        layer_key = 1
        for layer in self.neural_network:
            neuron_key = 1
            for neuron in layer:
                weight_key = 1
                if neuron:
                    for weight in neuron:
                        self.W_difs[("W_l%s_n%s_w%s") % (layer_key, neuron_key, weight_key)] = 0
                        weight_key += 1
                neuron_key += 1
            layer_key += 1
        
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def getError(self, results):
        numerator = 0
        i = 0
        for _set in results:
            #_set_numerator = ((_set[1] - _set[0]) ** 2)
            _set_numerator = math.fabs(_set[1] - _set[0])
            numerator = _set_numerator + numerator
            i=i+1
            #print(_set_numerator/1) #Ошибка за set
        return numerator/i
                    
    # Получаем веса выходящих синапсов
    def getWeightsSynaps(self, level, neuron):
        results = []
        i = 0
        for item in self.neural_network[level]:
            results.insert(i, item[neuron-1])
            i += 1
        return results
        
    # Получить все входы и выходы нейронов
    def getInputsAndOutputs(self, input_values):
        Out_output = False
        layer_key = 1
        for layer in self.neural_network:
            neuron_key = 1
            for neuron in layer:
                if neuron:
                    # for weight in neuron:
                    # weight_key += 1
                    if layer_key == 2: #if the second layer (after input layer)
                        key_input_value = 1
                        result = 0
                        for value in input_values:
                            result += value * neuron[key_input_value - 1]
                            key_input_value += 1
                        self.N_inputs[("N_l%s_n%s") % (layer_key, neuron_key)] = result
                        self.N_outputs[("N_l%s_n%s") % (layer_key, neuron_key)] = self.sigmoid(result)
                    else:
                        result = 0
                        weight_neuron_key = 1
                        for weight in neuron:
                            result += self.N_outputs[("N_l%s_n%s") % (layer_key-1, weight_neuron_key)] * weight
                            weight_neuron_key += 1
                        self.N_inputs[("N_l%s_n%s") % (layer_key, neuron_key)] = result
                        self.N_outputs[("N_l%s_n%s") % (layer_key, neuron_key)] = self.sigmoid(result)
                        # if last layer, it means that it's Out Neuron
                        if layer_key == self.out_layer:
                            Out_output = self.sigmoid(result)
                else: # Выставляем выходы для входных нейронов
                    #self.N_inputs[("N_l%s_n%s") % (layer_key, neuron_key)] = result
                    self.N_outputs[("N_l%s_n%s") % (layer_key, neuron_key)] = input_values[neuron_key-1]
                neuron_key += 1
            layer_key += 1
        return Out_output
    
    # Получить дельты у всех нейронов
    def getDeltas(self, Out_ideal):
        # reverse way from OUT to INPUT
        i_reverse_layer = self.out_layer
        while i_reverse_layer > 0:
            neuron_key = 1
            for neuron in self.neural_network[i_reverse_layer-1]: # It's list. Items start from 0. Because -1.
                if neuron:
                    if i_reverse_layer == self.out_layer: # it means this is OUT neuron
                        neuron_delta_key = ("Nd_l%s_n%s") % (i_reverse_layer, neuron_key)
                        level_delta_key = ("Nd_l%s") % (i_reverse_layer)
                        n_output = self.N_outputs[("N_l%s_n%s") % (i_reverse_layer, neuron_key)]
                        n_delta = (Out_ideal - n_output) * ( (1 - n_output) * n_output )
                        self.N_delta[neuron_delta_key] = n_delta
                        if level_delta_key not in self.N_delta:
                            self.N_delta[level_delta_key] = {}
                        self.N_delta[neuron_delta_key] = n_delta
                        self.N_delta[level_delta_key][neuron_key] = n_delta
                    else:                        
                        neuron_delta_key = ("Nd_l%s_n%s") % (i_reverse_layer, neuron_key)
                        level_delta_key = ("Nd_l%s") % (i_reverse_layer)
                        last_level_delta_key = ("Nd_l%s") % (i_reverse_layer + 1)
                        n_output = self.N_outputs[("N_l%s_n%s") % (i_reverse_layer, neuron_key)]
                        sum_wd = 0
                        weights_synaps = self.getWeightsSynaps(i_reverse_layer, neuron_key)
                        i_key_synaps = 1
                        for prev_delta_key in self.N_delta[last_level_delta_key]:
                            prev_delta = self.N_delta[last_level_delta_key][prev_delta_key]
                            sum_wd += ( weights_synaps[i_key_synaps-1] * prev_delta )
                            i_key_synaps += 1
                        n_delta = ( (1 - n_output) * n_output ) * sum_wd
                        self.N_delta[neuron_delta_key] = n_delta
                        if level_delta_key not in self.N_delta:
                            self.N_delta[level_delta_key] = {}
                        self.N_delta[neuron_delta_key] = n_delta
                        self.N_delta[level_delta_key][neuron_key] = n_delta
                neuron_key += 1
            i_reverse_layer -= 1
            
    #Получаем градиенты весов
    def getWeightsGradients(self):
        layer_key = 1
        for layer in self.neural_network:
            neuron_key = 1
            for neuron in layer:
                weight_key = 1
                if neuron:
                    for weight in neuron:
                        delta = self.N_delta[("Nd_l%s_n%s") % (layer_key, neuron_key)]
                        out = self.N_outputs[("N_l%s_n%s") % (layer_key-1, weight_key)]
                        gradient = delta * out
                        self.W_gradients[("W_l%s_n%s_w%s") % (layer_key, neuron_key, weight_key)] = gradient
                        weight_key += 1
                neuron_key += 1
            layer_key += 1
            
    #Получаем новые разницы изменения весов
    def getNewDifWeights(self):
        layer_key = 1
        for layer in self.neural_network:
            neuron_key = 1
            for neuron in layer:
                weight_key = 1
                if neuron:
                    for weight in neuron:
                        previous_dif = self.W_difs[("W_l%s_n%s_w%s") % (layer_key, neuron_key, weight_key)]
                        grad = self.W_gradients[("W_l%s_n%s_w%s") % (layer_key, neuron_key, weight_key)]
                        W_dif_new = self.E * grad + previous_dif * self.A
                        self.W_difs[("W_l%s_n%s_w%s") % (layer_key, neuron_key, weight_key)] = W_dif_new;
                        weight_key += 1
                neuron_key += 1
            layer_key += 1
            
    # обновляем веса
    def setNewWeights(self):
        layer_key = 1
        for layer in self.neural_network:
            neuron_key = 1
            for neuron in layer:
                weight_key = 1
                if neuron:
                    for weight in neuron:
                        W_new = weight + self.W_difs[("W_l%s_n%s_w%s") % (layer_key, neuron_key, weight_key)]
                        self.neural_network[layer_key-1][neuron_key-1][weight_key-1] = W_new
                        weight_key += 1
                neuron_key += 1
            layer_key += 1
            
    # init to Epoch
    def epoch(self, values):
        self.results = []
        i=0
        for input_values in values:
            self.oneSet(input_values, self.E, self.A)
            i += 1
        error = self.getError(self.results)
        print( ("epoch_error: '%s'") % (error) )
        print( '-------' )
            
    # Set
    def oneSet(self, input_values, E, A):
        #print( ("set a:'%s', b:'%s'") % (input_values[0], input_values[1]) )
        Out_ideal = input_values[0]^input_values[1] #Ожидаемый результат        
        Out_output = self.getInputsAndOutputs(input_values)

        result = [Out_output, Out_ideal, input_values[0], input_values[1]]
        self.results.insert(len(self.results), result)
        error = self.getError([result])
        print( ("set_error: '%s'") % (error) )
        print( ("Out_ideal: %s, Out_output: %s") % (Out_ideal, Out_output) )
        
        #print('N_inputs')
        #print(self.N_inputs)
        #print('N_outputs')
        #print(self.N_outputs)
                
        self.getDeltas(Out_ideal)
            
        #print('N_delta')
        #print(self.N_delta)
        
        self.getWeightsGradients()
        
        #print('W_gradients')
        #print(self.W_gradients)
        
        #print('W_difs_previous')
        #print(self.W_difs)
        
        self.getNewDifWeights()
        
        #print('W_difs_new')
        #print(self.W_difs)
        
        #print('neural_network_OLD')
        #print(self.neural_network)
        
        self.setNewWeights()
        
        #print('neural_network_NEW')
        #print(self.neural_network)
        
        #print( '-------' )
                    
    #Error
    #getError(results)

NeurosNetwork()
