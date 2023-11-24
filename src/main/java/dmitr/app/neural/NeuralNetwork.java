package dmitr.app.neural;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class NeuralNetwork {

    private final boolean hasBias;

    private Neuron[][] layers;
    private double[][][] weights;
    private double[][][] deltaStorage;

    public NeuralNetwork(int inputNeuronsCount, int[] hiddenNeuronsCounts, int outputNeuronsCount, boolean hasBias) {
        this.hasBias = hasBias;

        buildLayers(inputNeuronsCount, hiddenNeuronsCounts, outputNeuronsCount);
        setupWeights();
    }

    private int getIntBias() {
        return hasBias ? 1 : 0;
    }

    private void buildLayers(int inputNeuronsCount, int[] hiddenNeuronsCounts, int outputNeuronsCount) {
        layers = new Neuron[1 + hiddenNeuronsCounts.length + 1][];

        layers[0] = IntStream.range(0, inputNeuronsCount + getIntBias())
                .mapToObj(n -> new Neuron(ActivationFunction::linear, NeuronType.DEFAULT))
                .toList().toArray(new Neuron[0]);

        for (int i = 0; i < hiddenNeuronsCounts.length; i++)
            layers[i + 1] = IntStream.range(0, hiddenNeuronsCounts[i] + getIntBias())
                    .mapToObj(n -> new Neuron(ActivationFunction::sigmoid, NeuronType.DEFAULT))
                    .toList().toArray(new Neuron[0]);

        layers[layers.length - 1] = IntStream.range(0, outputNeuronsCount)
                .mapToObj(n -> new Neuron(ActivationFunction::sigmoid, NeuronType.DEFAULT))
                .toList().toArray(new Neuron[0]);

        if (hasBias)
            IntStream.range(0, layers.length - 1)
                    .forEach(n -> layers[n][layers[n].length - 1].setNeuronType(NeuronType.BIAS));
    }

    private void setupWeights() {
        weights = new double[layers.length][][];
        deltaStorage = new double[layers.length][][];

        for (int i = 0; i < layers.length - 1; i++) {
            weights[i] = new double[layers[i].length][];
            deltaStorage[i] = new double[layers[i].length][];
            for (int j = 0; j < layers[i].length; j++) {
                weights[i][j] = ThreadLocalRandom.current().doubles(layers[i + 1].length).toArray();
                deltaStorage[i][j] = new double[layers[i + 1].length];
            }
        }
    }

    public double[] predict(double[] input) throws RuntimeException {
        if (input.length != layers[0].length - getIntBias())
            throw new RuntimeException("[NeuralNetwork Error] The number of input values does not match the number of input neurons!");

        for (int i = 0; i < input.length; i++)
            layers[0][i].setData(input[i]);

        for (int i = 1; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                if (layers[i][j].getNeuronType() == NeuronType.BIAS)
                    continue;

                int finalI = i;
                int finalJ = j;
                double value = IntStream.range(0, layers[i - 1].length)
                        .mapToDouble(n -> layers[finalI - 1][n].getData() * weights[finalI - 1][n][finalJ]).sum();

                layers[i][j].setData(layers[i][j].getNeuralActivation().activate(value, false));
            }
        }

        return Arrays.stream(layers[layers.length - 1]).mapToDouble(Neuron::getData).toArray();
    }

    public void learn(double[] input, double[] expectedOutput, float learningRate, float moment) throws RuntimeException {
        if (input.length != layers[0].length - getIntBias())
            throw new RuntimeException("[NeuralNetwork Error] The number of input values does not match the number of input layer!");

        if (expectedOutput.length != layers[layers.length - 1].length)
            throw new RuntimeException("[NeuralNetwork Error] The number of expected output values does not match the number of output layer!");

        double[] currentOutput = predict(input);

        double[][] delta = new double[layers.length][];

        delta[layers.length - 1] = new double[layers[layers.length - 1].length];
        for (int i = 0; i < layers[layers.length - 1].length; i++)
            delta[layers.length - 1][i] = (expectedOutput[i] - currentOutput[i]) * layers[layers.length - 1][i]
                    .getNeuralActivation().activate(currentOutput[i], true);

        for (int i = layers.length - 2; i > 0; i--) {
            delta[i] = new double[layers[i].length];
            for (int j = 0; j < layers[i].length; j++) {
                int finalI = i;
                int finalJ = j;
                delta[i][j] = layers[i][j].getNeuralActivation().activate(layers[i][j].getData(), true) *
                        IntStream.range(0, layers[i + 1].length)
                                .mapToDouble(n -> weights[finalI][finalJ][n] * delta[finalI + 1][n]).sum();
            }
        }

        for (int i = 1; i < layers.length; i++)
            for (int j = 0; j < layers[i].length; j++)
                for (int k = 0; k < layers[i - 1].length; k++) {
                    double value = learningRate * delta[i][j] * layers[i - 1][k].getData() +
                            deltaStorage[i - 1][k][j] * moment;
                    weights[i - 1][k][j] += value;
                    deltaStorage[i - 1][k][j] = value;
                }
    }

}