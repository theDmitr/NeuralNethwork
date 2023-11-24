package dmitr.app.neural;

import static java.lang.Math.exp;

public class ActivationFunction {

    public static double linear(double value, boolean derivative) {
        return value;
    }

    public static double sigmoid(double value, boolean derivative) {
        return derivative ? (1 - value) * value : 1 / (1 + exp(-value));
    }

}
