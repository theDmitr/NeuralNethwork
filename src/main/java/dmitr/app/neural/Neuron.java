package dmitr.app.neural;

public class Neuron {

    private final INeuronActivation neuralActivation;
    private NeuronType neuronType;
    private double data;

    public Neuron(INeuronActivation neuronActivation, NeuronType neuronType) {
        this.neuralActivation = neuronActivation;
        this.neuronType = neuronType;
    }

    public double getData() {
        return neuronType == NeuronType.BIAS ? 1 : data;
    }

    public void setData(double data) {
        if (neuronType == NeuronType.BIAS)
            return;
        this.data = data;
    }

    public NeuronType getNeuronType() {
        return neuronType;
    }

    public void setNeuronType(NeuronType neuronType) {
        this.neuronType = neuronType;
    }

    public INeuronActivation getNeuralActivation() {
        return neuralActivation;
    }

    @Override
    public String toString() {
        return Double.toString(getData());
    }

}
