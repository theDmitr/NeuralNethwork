package dmitr.neural;

public class Neuron {

    private NeuronType neuronType;
    private double data;

    public Neuron(NeuronType neuronType) {
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

    @Override
    public String toString() {
        return Double.toString(getData());
    }

}
