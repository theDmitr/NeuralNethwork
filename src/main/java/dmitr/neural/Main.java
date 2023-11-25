package dmitr.neural;

import dmitr.neural.activation.NeuronActivation;
import dmitr.neural.parse.NeuralNetworkParser;

import java.io.*;

public class Main {

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{2, 2, 2, 1}, true, NeuronActivation.SIGMOID);

        float learnRate = 1f;
        float moment = 0.93f;

        neuralNetwork.learn(
                new double[][]{new double[]{0, 0}, new double[]{0, 1}, new double[]{1, 0}, new double[]{1, 1}},
                new double[][]{new double[]{1}, new double[]{0}, new double[]{0}, new double[]{1}},
                learnRate, moment, 1000
        );

        System.out.println(neuralNetwork.predict(new double[]{0, 0})[0]);
        System.out.println(neuralNetwork.predict(new double[]{0, 1})[0]);
        System.out.println(neuralNetwork.predict(new double[]{1, 0})[0]);
        System.out.println(neuralNetwork.predict(new double[]{1, 1})[0]);

        File out = new File("test");
        OutputStream outputStream = null;
        try {
            outputStream = new FileOutputStream(out);
        } catch (IOException  ignored) {}

        NeuralNetworkParser.write(neuralNetwork, outputStream);

        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(out);
        } catch (IOException ignored) {}
        NeuralNetwork nn = NeuralNetworkParser.get(inputStream);

        System.out.println(nn.predict(new double[]{0, 0})[0]);
        System.out.println(nn.predict(new double[]{0, 1})[0]);
        System.out.println(nn.predict(new double[]{1, 0})[0]);
        System.out.println(nn.predict(new double[]{1, 1})[0]);
    }

}