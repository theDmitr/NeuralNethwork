package dmitr.neural.parse;

import dmitr.neural.NeuralNetwork;
import dmitr.neural.Neuron;
import dmitr.neural.activation.NeuronActivation;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class NeuralNetworkParser {

    public static void write(NeuralNetwork network, OutputStream out) {
        Neuron[][] networkLayers = network.getLayers();
        double[][][] networkWeights = network.getWeights();

        int bias = network.getIntBias();
        int activation = network.getNeuronActivation().id;

        int[] layers = Arrays.stream(networkLayers)
                .mapToInt(networkLayer -> networkLayer.length - network.getIntBias())
                .toArray();
        layers[layers.length - 1] += network.getIntBias();

        double[] weights = Arrays.stream(networkWeights)
                .flatMap(Arrays::stream)
                .flatMapToDouble(Arrays::stream)
                .toArray();

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8))) {
            writer.write(bias + "#");
            writer.write(activation + "#");

            StringBuilder layersOut = new StringBuilder();
            for (int l : layers)
                layersOut.append(l).append(",");
            writer.write(layersOut.substring(0, layersOut.length() - 1) + "#");

            writer.write(weights.length + "#");

            StringBuilder weightsOut = new StringBuilder();
            for (double w : weights)
                weightsOut.append(w).append(",");
            writer.write(weightsOut.substring(0, weightsOut.length() - 1));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static NeuralNetwork get(InputStream in) {
        NeuralNetwork neuralNetwork = null;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line = reader.readLine();
            String[] parts = line.split("#");

            boolean bias = parts[0].equals("1");
            NeuronActivation activation = NeuronActivation.get(Integer.parseInt(parts[1]));

            String[] layerCountsInput = parts[2].split(",");
            int[] layerCounts = new int[layerCountsInput.length];
            for (int i = 0; i < layerCounts.length; i++)
                layerCounts[i] = Integer.parseInt(layerCountsInput[i]) + (bias ? 1 : 0);
            layerCounts[layerCounts.length - 1] -= (bias ? 1 : 0);

            int weightsCount = Integer.parseInt(parts[3]);
            String[] weightsInput = parts[4].split(",");
            double[] weightsSolid = new double[weightsCount];
            for (int i = 0; i < weightsSolid.length; i++)
                weightsSolid[i] = Double.parseDouble(weightsInput[i]);

            double[][][] weights = new double[layerCounts.length - 1][][];

            for (int i = 0; i < layerCounts.length - 1; i++) {
                weights[i] = new double[layerCounts[i]][];
                for (int j = 0; j < layerCounts[i]; j++) {
                    weights[i][j] = new double[layerCounts[i + 1]];
                    for (int k = 0; k < weights[i][j].length; k++)
                        weights[i][j][k] = weightsSolid[layerCounts.length * i + j + k];
                }
            }

            for (int i = 0; i < layerCounts.length; i++)
                layerCounts[i] -= (bias ? 1 : 0);
            layerCounts[layerCounts.length - 1] += (bias ? 1 : 0);

            neuralNetwork = new NeuralNetwork(layerCounts, bias, activation, weights);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return neuralNetwork;
    }

}
