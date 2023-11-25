package dmitr.neural;

import dmitr.neural.activation.NeuronActivation;

import java.util.Random;

public class NeuralNetwork {

    private final boolean hasBias;

    private Neuron[][] layers;
    private double[][][] weights;
    private double[][][] deltaStorage;

    private final NeuronActivation neuronActivation;

    public NeuralNetwork(int[] layersNeuronsCounts, boolean hasBias, NeuronActivation neuronActivation) {
        this.hasBias = hasBias;
        this.neuronActivation = neuronActivation;

        buildLayers(layersNeuronsCounts);
        resetWeights();
    }

    public NeuralNetwork(int[] layersNeuronsCounts, boolean hasBias, NeuronActivation neuronActivation, double[][][] weights) {
        this.hasBias = hasBias;
        this.neuronActivation = neuronActivation;
        this.weights = weights;

        buildLayers(layersNeuronsCounts);
        setupDeltaStorage();
    }

    private void buildLayers(int[] layersNeuronsCounts) {
        layers = new Neuron[layersNeuronsCounts.length][];

        layers[0] = new Neuron[layersNeuronsCounts[0] + getIntBias()];
        for (int i = 0; i < layers[0].length; i++)
            layers[0][i] = new Neuron(NeuronType.DEFAULT);

        for (int i = 0; i < layersNeuronsCounts.length - 2; i++) {
            layers[i + 1] = new Neuron[layersNeuronsCounts[i + 1] + getIntBias()];
            for (int j = 0; j < layers[i + 1].length; j++)
                layers[i + 1][j] = new Neuron(NeuronType.DEFAULT);
        }

        layers[layers.length - 1] = new Neuron[layersNeuronsCounts[layersNeuronsCounts.length - 1]];
        for (int i = 0; i < layers[layers.length - 1].length; i++)
            layers[layers.length - 1][i] = new Neuron(NeuronType.DEFAULT);

        if (hasBias)
            for (int i = 0; i < layers.length - 1; i++)
                layers[i][layers[i].length - 1].setNeuronType(NeuronType.BIAS);
    }

    public void resetWeights() {
        weights = new double[layers.length - 1][][];
        deltaStorage = new double[layers.length][][];

        Random random = new Random();

        for (int i = 0; i < layers.length - 1; i++) {
            weights[i] = new double[layers[i].length][];
            deltaStorage[i] = new double[layers[i].length][];
            for (int j = 0; j < layers[i].length; j++) {
                weights[i][j] = new double[layers[i + 1].length];

                for (int k = 0; k < weights[i][j].length; k++)
                    weights[i][j][k] = random.nextDouble();

                deltaStorage[i][j] = new double[layers[i + 1].length];
            }
        }
    }

    private void setupDeltaStorage() {
        deltaStorage = new double[layers.length][][];
        for (int i = 0; i < layers.length - 1; i++) {
            deltaStorage[i] = new double[layers[i].length][];
            for (int j = 0; j < layers[i].length; j++)
                deltaStorage[i][j] = new double[layers[i + 1].length];
        }
    }

    public double[] predict(double[] input) throws RuntimeException {
        if (input.length != layers[0].length - getIntBias())
            throw new RuntimeException("[NeuralNetwork Error] The number of input values does not match the number of input neurons!");

        for (int i = 0; i < input.length; i++)
            layers[0][i].setData(input[i]);

        for (int i = 1; i < layers.length; i++)
            for (int j = 0; j < layers[i].length; j++) {
                if (layers[i][j].getNeuronType() == NeuronType.BIAS)
                    continue;

                double value = 0.0;
                for (int k = 0; k < layers[i - 1].length; k++)
                    value += layers[i - 1][k].getData() * weights[i - 1][k][j];

                layers[i][j].setData(neuronActivation.activation.activate(value, false));
            }

        double[] result = new double[layers[layers.length - 1].length];

        for (int i = 0; i < result.length; i++)
            result[i] = layers[layers.length - 1][i].getData();

        return result;
    }

    private double[] train(double[] input, double[] expectedOutput, float learningRate, float moment) throws RuntimeException {
        double[] currentOutput = predict(input);

        double[][] delta = new double[layers.length][];
        double[] error = new double[layers[layers.length - 1].length];

        delta[layers.length - 1] = new double[layers[layers.length - 1].length];
        for (int i = 0; i < layers[layers.length - 1].length; i++) {
            delta[layers.length - 1][i] = (expectedOutput[i] - currentOutput[i]) * neuronActivation.activation.activate(currentOutput[i], true);
            error[i] = (Math.pow(expectedOutput[i] - currentOutput[i], 2));
        }

        for (int i = layers.length - 2; i > 0; i--) {
            delta[i] = new double[layers[i].length];
            for (int j = 0; j < layers[i].length; j++) {
                double weightsDeltaSum = 0.0;

                for (int k = 0; k < layers[i + 1].length; k++)
                    weightsDeltaSum += weights[i][j][k] * delta[i + 1][k];

                delta[i][j] = neuronActivation.activation.activate(layers[i][j].getData(), true) * weightsDeltaSum;
            }
        }

        for (int i = 1; i < layers.length; i++)
            for (int j = 0; j < layers[i].length; j++)
                for (int k = 0; k < layers[i - 1].length; k++) {
                    double value = learningRate * delta[i][j] * layers[i - 1][k].getData() + deltaStorage[i - 1][k][j] * moment;
                    weights[i - 1][k][j] += value;
                    deltaStorage[i - 1][k][j] = value;
                }

        return error;
    }

    private void warnLearnInconsistencies(double[][] input, double[][] expectedOutput, float moment) {
        if (moment < 0.0f || moment > 1.0f)
            throw new RuntimeException("[NeuralNetwork Error] The moment is not in the range [0;1]!");

        if (input[0].length != layers[0].length - getIntBias())
            throw new RuntimeException("[NeuralNetwork Error] The number of input values does not match the number of input layer!");

        if (expectedOutput[0].length != layers[layers.length - 1].length)
            throw new RuntimeException("[NeuralNetwork Error] The number of expected output values does not match the number of output layer!");
    }

    public void learn(double[][] input, double[][] expectedOutput, float learningRate,
                       float moment, int iterations) throws RuntimeException {
        warnLearnInconsistencies(input, expectedOutput, moment);

        for (int i = 0; i < iterations; i++)
            for (int k = 0; k < input.length; k++)
                train(input[k], expectedOutput[k], learningRate, moment);
    }

    public void learn(double[][] input, double[][] expectedOutput, float learningRate,
                       float moment, float error, int maxIterations) throws RuntimeException {
        warnLearnInconsistencies(input, expectedOutput, moment);

        int iter = 0;
        loop:
        while (iter < maxIterations) {
            trainLoop:
            for (int i = 0; i < input.length; i++) {
                double[] errors = train(input[i], expectedOutput[i], learningRate, moment);

                for (double e : errors)
                    if (e > error)
                        continue trainLoop;

                break loop;
            }
            iter++;
        }
    }

    public int getIntBias() {
        return hasBias ? 1 : 0;
    }

    public NeuronActivation getNeuronActivation() {
        return neuronActivation;
    }

    public Neuron[][] getLayers() {
        return layers;
    }

    public double[][][] getWeights() {
        return weights;
    }

}
