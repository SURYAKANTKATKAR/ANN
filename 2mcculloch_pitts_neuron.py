def mcculloch_pitts_neuron(inputs, weights, threshold):
    """McCulloch-Pitts neuron activation function"""
    weighted_sum = sum([x * w for x, w in zip(inputs, weights)])
    if weighted_sum >= threshold:
        return 1
    else:
        return 0

def ANDNOT(x1, x2):
    """ANDNOT function"""
    # Define the weights and threshold for ANDNOT function
    weights = [1, -1]  # weights for x1 and x2 respectively
    threshold = 1  # threshold for activation

    # Feed inputs to the neuron
    return mcculloch_pitts_neuron([x1, x2], weights, threshold)

# Test the ANDNOT function
print("ANDNOT(0, 0) =", ANDNOT(0, 0))
print("ANDNOT(0, 1) =", ANDNOT(0, 1))
print("ANDNOT(1, 0) =", ANDNOT(1, 0))
print("ANDNOT(1, 1) =", ANDNOT(1, 1))
