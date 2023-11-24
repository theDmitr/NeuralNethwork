package dmitr.app;

import dmitr.app.neural.NeuralNetwork;

public class Main {

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(1, new int[]{}, 1, true);

        float learnRate = 2f;
        float moment = 0.8f;

        for (int i = 0; i < 60; i++) {
            neuralNetwork.learn(new double[]{0}, new double[]{1}, learnRate, moment);
            neuralNetwork.learn(new double[]{1}, new double[]{0}, learnRate, moment);
        }

        System.out.println(neuralNetwork.predict(new double[]{1})[0]);
        System.out.println(neuralNetwork.predict(new double[]{0})[0]);
    }

}