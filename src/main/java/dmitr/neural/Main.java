package dmitr.neural;

import dmitr.neural.activation.NeuronActivation;
import dmitr.neural.parse.NeuralNetworkParser;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{2, 1}, true, NeuronActivation.SIGMOID);

        float learnRate = 1f;
        float moment = 0.93f;

        neuralNetwork.learn(
                new double[][]{new double[]{0, 0}, new double[]{0, 1}, new double[]{1, 0}, new double[]{1, 1}},
                new double[][]{new double[]{0}, new double[]{0}, new double[]{0}, new double[]{1}},
                learnRate, moment, 0.01f, 1000000000, ErrorCalcType.ROOT_MSE
        );

        System.out.println(neuralNetwork.predict(new double[]{0, 0})[0]);
        System.out.println(neuralNetwork.predict(new double[]{0, 1})[0]);
        System.out.println(neuralNetwork.predict(new double[]{1, 0})[0]);
        System.out.println(neuralNetwork.predict(new double[]{1, 1})[0]);
    }

}