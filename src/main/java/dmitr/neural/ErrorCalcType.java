package dmitr.neural;

import static java.lang.Math.pow;
import static java.lang.Math.atan;
import static java.lang.Math.sqrt;

public enum ErrorCalcType {

    MSE,

    ROOT_MSE {
        public double getGlobalError(double error, int passedIterations) {
            return sqrt(error / passedIterations);
        }
    },

    ARCTAN {
        public void insert(double[] errors, double[] newErrors) {
            for (int i = 0; i < errors.length; i++)
                errors[i] += pow(atan(newErrors[i]), 2);
        }
    };

    public void insert(double[] errors, double[] newErrors) {
        for (int i = 0; i < errors.length; i++)
            errors[i] += newErrors[i];
    }

    public double getGlobalError(double error, int passedIterations) {
        return error / passedIterations;
    }

}
