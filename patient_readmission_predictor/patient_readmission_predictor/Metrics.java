/** Classification metrics for evaluating the readmission model. */
public class Metrics {

    public final int tp, fp, tn, fn;
    public final double auc;

    public Metrics(int[] actual, double[] probabilities, double threshold) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (int i = 0; i < actual.length; i++) {
            boolean predicted = probabilities[i] >= threshold;
            if (predicted && actual[i] == 1) tp++;
            else if (predicted) fp++;
            else if (actual[i] == 1) fn++;
            else tn++;
        }
        this.tp = tp;
        this.fp = fp;
        this.tn = tn;
        this.fn = fn;
        this.auc = computeAuc(actual, probabilities);
    }

    public double accuracy() {
        return (double) (tp + tn) / (tp + tn + fp + fn);
    }

    public double precision() {
        return tp + fp == 0 ? 0 : (double) tp / (tp + fp);
    }

    public double recall() {
        return tp + fn == 0 ? 0 : (double) tp / (tp + fn);
    }

    public double f1() {
        double p = precision(), r = recall();
        return p + r == 0 ? 0 : 2 * p * r / (p + r);
    }

    /** AUC = probability a random readmitted patient scores higher than a random non-readmitted one. */
    private static double computeAuc(int[] actual, double[] scores) {
        double wins = 0;
        long pairs = 0;
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != 1) continue;
            for (int j = 0; j < actual.length; j++) {
                if (actual[j] != 0) continue;
                pairs++;
                if (scores[i] > scores[j]) wins += 1;
                else if (scores[i] == scores[j]) wins += 0.5;
            }
        }
        return pairs == 0 ? 0 : wins / pairs;
    }

    public void print() {
        System.out.printf("Confusion matrix:  TP=%d  FP=%d  TN=%d  FN=%d%n", tp, fp, tn, fn);
        System.out.printf("Accuracy : %.3f%n", accuracy());
        System.out.printf("Precision: %.3f%n", precision());
        System.out.printf("Recall   : %.3f%n", recall());
        System.out.printf("F1 score : %.3f%n", f1());
        System.out.printf("ROC AUC  : %.3f%n", auc);
    }
}
