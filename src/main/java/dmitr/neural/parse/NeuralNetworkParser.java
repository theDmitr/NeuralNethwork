package dmitr.neural.parse;

import dmitr.neural.NeuralNetwork;
import dmitr.neural.Neuron;
import dmitr.neural.activation.NeuronActivation;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class NeuralNetworkParser {

    public static void write(NeuralNetwork network, OutputStream out) {
        Neuron[][] networkLayers = network.getLayers();

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8))) {
            writer.write(network.getIntBias() + "\n");
            writer.write(network.getNeuronActivation().id + "\n");
            writer.write(networkLayers.length + "\n");

            for (int i = 0; i < networkLayers.length; i++)
                writer.write(networkLayers[i].length - ((i != networkLayers.length - 1) ? network.getIntBias() : 0) + "\n");

            for (double[][] i : network.getWeights())
                for (double[] j : i)
                    for (double k : j)
                        writer.write(k + "\n");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static NeuralNetwork get(InputStream in) {
        NeuralNetwork neuralNetwork;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            boolean bias = reader.readLine().equals("1");
            NeuronActivation activation = NeuronActivation.get(Integer.parseInt(reader.readLine()));

            int[] layerCounts = new int[Integer.parseInt(reader.readLine())];
            for (int i = 0; i < layerCounts.length; i++)
                layerCounts[i] = Integer.parseInt(reader.readLine());

            double[][][] weights = new double[layerCounts.length - 1][][];

            for (int i = 0; i < layerCounts.length - 1; i++) {
                weights[i] = new double[layerCounts[i] + (bias ? 1 : 0)][];
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = new double[layerCounts[i + 1] + ((i != layerCounts.length - 2) ? (bias ? 1 : 0) : 0)];
                    for (int k = 0; k < weights[i][j].length; k++)
                        weights[i][j][k] = Double.parseDouble(reader.readLine());
                }
            }

            neuralNetwork = new NeuralNetwork(layerCounts, bias, activation, weights);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return neuralNetwork;
    }

}
