package dmitr.neural.activation;

import static java.lang.Math.exp;

public enum NeuronActivation {

    LINEAR(0),
    SIGMOID(1) {
        @Override
        public double getDefault(double value) {
            return 1 / (1 + exp(-value));
        }

        @Override
        public double getDerivative(double value) {
            return (1 - value) * value;
        }
    };

    public final int id;

    NeuronActivation(final int id) {
        this.id = id;
    }

    public double getDefault(double value) {
        return value;
    }

    public double getDerivative(double value) {
        return value;
    }

    public static NeuronActivation get(int id) {
        for (NeuronActivation activation : NeuronActivation.values())
            if (activation.id == id)
                return activation;
        return null;
    }

}
