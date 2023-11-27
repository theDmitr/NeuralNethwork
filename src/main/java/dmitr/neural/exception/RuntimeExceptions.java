package dmitr.neural.exception;

public class RuntimeExceptions {

    public static final RuntimeException input = new RuntimeException("[NeuralNetwork Error] The number of input values does not match the number of input neurons!");
    public static final RuntimeException output = new RuntimeException("[NeuralNetwork Error] The number of expected output values does not match the number of output layer!");
    public static final RuntimeException layersCount = new RuntimeException("[NeuralNetwork Error] A neural network must have at least 1 input and 1 output neuron!");
    public static final RuntimeException neuronsCount = new RuntimeException("[NeuralNetwork Error] A layer must have at least 1 neuron!");
    public static final RuntimeException moment = new RuntimeException("[NeuralNetwork Error] The moment is not in the range [0;1]!");

}
