package dmitr.neural;

import dmitr.neural.activation.NeuronActivation;
import dmitr.neural.exception.RuntimeExceptions;

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
        if (layersNeuronsCounts.length < 2)
            throw RuntimeExceptions.layersCount;

        for (int layersNeuronsCount : layersNeuronsCounts)
            if (layersNeuronsCount < 1)
                throw RuntimeExceptions.neuronsCount;

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
            throw RuntimeExceptions.input;

        for (int i = 0; i < input.length; i++)
            layers[0][i].setData(input[i]);

        for (int i = 1; i < layers.length; i++)
            for (int j = 0; j < layers[i].length; j++) {
                if (layers[i][j].getNeuronType() == NeuronType.BIAS)
                    continue;

                double value = 0.0;
                for (int k = 0; k < layers[i - 1].length; k++)
                    value += layers[i - 1][k].getData() * weights[i - 1][k][j];

                layers[i][j].setData(neuronActivation.getDefault(value));
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
            delta[layers.length - 1][i] = (expectedOutput[i] - currentOutput[i]) * neuronActivation.getDerivative(currentOutput[i]);
            error[i] = (Math.pow(expectedOutput[i] - currentOutput[i], 2));
        }

        for (int i = layers.length - 2; i > 0; i--) {
            delta[i] = new double[layers[i].length];
            for (int j = 0; j < layers[i].length; j++) {
                double weightsDeltaSum = 0.0;

                for (int k = 0; k < layers[i + 1].length; k++)
                    weightsDeltaSum += weights[i][j][k] * delta[i + 1][k];

                delta[i][j] = neuronActivation.getDerivative(layers[i][j].getData()) * weightsDeltaSum;
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

    private void warnLearnInconsistencies(LearnDataset dataset, float moment) {
        if (moment < 0.0f || moment > 1.0f)
            throw RuntimeExceptions.moment;

        if (dataset.getInputNeurons() != layers[0].length - getIntBias())
            throw RuntimeExceptions.input;

        if (dataset.getOutputNeurons() != layers[layers.length - 1].length)
            throw RuntimeExceptions.output;
    }

    public void learn(LearnDataset dataset, float learningRate, float moment, int iterations) throws RuntimeException {
        warnLearnInconsistencies(dataset, moment);

        for (int i = 0; i < iterations; i++)
            for (int k = 0; k < dataset.getSize(); k++)
                train(dataset.getInputPart(k), dataset.getOutputPart(k), learningRate, moment);
    }

    public void learn(LearnDataset dataset, float learningRate, float moment, float error, 
                      int maxIterations, ErrorCalcType errorCalcType) throws RuntimeException {
        warnLearnInconsistencies(dataset, moment);

        int iter = 0;
        double[] errors = new double[layers[layers.length - 1].length];
        loop:
        while (iter < maxIterations) {
            trainLoop:
            for (int i = 0; i < dataset.getSize(); i++) {
                double[] iterationErrors = train(dataset.getInputPart(i), dataset.getOutputPart(i), learningRate, moment);
                errorCalcType.insert(errors, iterationErrors);

                for (double e : errors)
                    if (errorCalcType.getGlobalError(e, iter + 1) > error)
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
