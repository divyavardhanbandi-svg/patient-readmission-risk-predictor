/**
 * Logistic regression trained with batch gradient descent.
 * Features are standardized (z-score) using statistics from the training data.
 */
public class LogisticRegression {

    private double[] weights;
    private double bias;
    private double[] mean;
    private double[] std;

    public void fit(double[][] x, int[] y, int epochs, double learningRate, double l2) {
        int n = x.length;
        int d = x[0].length;
        computeScaling(x);

        double[][] xs = new double[n][];
        for (int i = 0; i < n; i++) {
            xs[i] = scale(x[i]);
        }

        weights = new double[d];
        bias = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradW = new double[d];
            double gradB = 0;
            for (int i = 0; i < n; i++) {
                double error = sigmoid(dot(xs[i])) - y[i];
                for (int j = 0; j < d; j++) {
                    gradW[j] += error * xs[i][j];
                }
                gradB += error;
            }
            for (int j = 0; j < d; j++) {
                weights[j] -= learningRate * (gradW[j] / n + l2 * weights[j]);
            }
            bias -= learningRate * (gradB / n);
        }
    }

    /** Probability (0..1) that the patient is readmitted. */
    public double predictProbability(double[] features) {
        return sigmoid(dot(scale(features)));
    }

    /** Learned weights on standardized features (larger magnitude = stronger effect). */
    public double[] getWeights() {
        return weights.clone();
    }

    private void computeScaling(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        mean = new double[d];
        std = new double[d];
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                mean[j] += row[j] / n;
            }
        }
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                std[j] += Math.pow(row[j] - mean[j], 2) / n;
            }
        }
        for (int j = 0; j < d; j++) {
            std[j] = Math.sqrt(std[j]);
            if (std[j] == 0) {
                std[j] = 1;
            }
        }
    }

    private double[] scale(double[] row) {
        double[] out = new double[row.length];
        for (int j = 0; j < row.length; j++) {
            out[j] = (row[j] - mean[j]) / std[j];
        }
        return out;
    }

    private double dot(double[] scaled) {
        double sum = bias;
        for (int j = 0; j < scaled.length; j++) {
            sum += weights[j] * scaled[j];
        }
        return sum;
    }

    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
}
