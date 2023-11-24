package dmitr.neural.activation;

public enum NeuronActivation {

    LINEAR(0, ActivationFunction::linear), SIGMOID(1, ActivationFunction::sigmoid);

    public final INeuronActivation activation;
    public final int id;

    NeuronActivation(final int id, INeuronActivation activation) {
        this.activation = activation;
        this.id = id;
    }

    public static NeuronActivation get(int id) {
        for (NeuronActivation activation : NeuronActivation.values())
            if (activation.id == id)
                return activation;
        return null;
    }

}
