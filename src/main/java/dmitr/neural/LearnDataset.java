package dmitr.neural;

import dmitr.neural.exception.RuntimeExceptions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LearnDataset {

    private final int inputNeurons;
    private final int outputNeurons;
    private final List<double[]> set;

    public LearnDataset(int inputNeurons, int outputNeurons, List<double[]> set) {
        if (inputNeurons < 1 || outputNeurons < 1)
            throw RuntimeExceptions.neuronsCount;

        this.inputNeurons = inputNeurons;
        this.outputNeurons = outputNeurons;
        this.set = set;
    }

    public LearnDataset(int inputNeurons, int outputNeurons) {
        this(inputNeurons, outputNeurons, new ArrayList<>());
    }

    public void insert(double[] set) {
        if (set.length != inputNeurons + outputNeurons)
            throw RuntimeExceptions.inputOutput;

        this.set.add(set);
    }

    public int getInputNeurons() {
        return inputNeurons;
    }

    public int getOutputNeurons() {
        return outputNeurons;
    }

    public List<double[]> getSet() {
        return set;
    }

    public double[] getInputPart(int index) {
        return Arrays.copyOfRange(set.get(index), 0, inputNeurons);
    }

    public double[] getOutputPart(int index) {
        return Arrays.copyOfRange(set.get(index), inputNeurons, inputNeurons + outputNeurons);
    }

    public int getSize() {
        return set.size();
    }

}
