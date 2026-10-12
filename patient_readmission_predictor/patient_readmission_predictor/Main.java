import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Patient Readmission Risk Predictor.
 *
 * Usage:
 *   java Main
 *       -> trains on synthetic data, prints metrics and a few sample predictions
 *   java Main <age> <lengthOfStay> <priorAdmissions> <medications> <diagnoses> <diabetes 0/1> <heartFailure 0/1> <emergency 0/1>
 *       -> trains, then predicts risk for the patient you describe
 */
public class Main {

    private static final double THRESHOLD = 0.20; // lower than 0.5 so fewer high-risk patients are missed

    public static void main(String[] args) {
        // 1. Data
        List<Patient> all = new DataGenerator(42).generate(10_000);
        Collections.shuffle(all, new Random(7));
        int split = (int) (all.size() * 0.8);
        List<Patient> train = all.subList(0, split);
        List<Patient> test = all.subList(split, all.size());

        double rate = train.stream().mapToInt(Patient::readmitted).average().orElse(0);
        System.out.printf("Patients: %d train / %d test | 30-day readmission rate: %.1f%%%n%n",
                train.size(), test.size(), rate * 100);

        // 2. Train
        LogisticRegression model = new LogisticRegression();
        model.fit(toMatrix(train), toLabels(train), 2000, 0.1, 0.001);

        // 3. Evaluate
        double[][] xTest = toMatrix(test);
        int[] yTest = toLabels(test);
        double[] probs = new double[xTest.length];
        for (int i = 0; i < xTest.length; i++) {
            probs[i] = model.predictProbability(xTest[i]);
        }
        System.out.printf("=== Test results (threshold = %.2f) ===%n", THRESHOLD);
        new Metrics(yTest, probs, THRESHOLD).print();

        // 4. Feature importance
        System.out.println("\n=== Risk factors (standardized weights) ===");
        double[] w = model.getWeights();
        Integer[] order = new Integer[w.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(Math.abs(w[b]), Math.abs(w[a])));
        for (int idx : order) {
            System.out.printf("%-26s %+.3f%n", Patient.FEATURE_NAMES[idx], w[idx]);
        }

        // 5. Predictions
        System.out.println("\n=== Patient risk predictions ===");
        if (args.length == 8) {
            double[] f = new double[8];
            try {
                for (int i = 0; i < 8; i++) f[i] = Double.parseDouble(args[i]);
            } catch (NumberFormatException e) {
                System.err.println("All 8 arguments must be numbers.");
                return;
            }
            report("Your patient", f, model);
        } else {
            //                                   age  los prior meds dx  dm hf er
            report("Young, healthy, elective   ", new double[]{35, 2, 0, 3, 1, 0, 0, 0}, model);
            report("Middle-aged, diabetic      ", new double[]{55, 5, 1, 8, 4, 1, 0, 1}, model);
            report("Elderly, heart failure     ", new double[]{78, 9, 3, 14, 7, 1, 1, 1}, model);
        }
    }

    private static void report(String label, double[] features, LogisticRegression model) {
        double p = model.predictProbability(features);
        String band = p < 0.15 ? "LOW" : p < 0.35 ? "MEDIUM" : "HIGH";
        System.out.printf("%-27s -> readmission risk %5.1f%%  [%s]%n", label, p * 100, band);
    }

    private static double[][] toMatrix(List<Patient> patients) {
        List<double[]> rows = new ArrayList<>();
        for (Patient p : patients) rows.add(p.features());
        return rows.toArray(new double[0][]);
    }

    private static int[] toLabels(List<Patient> patients) {
        return patients.stream().mapToInt(Patient::readmitted).toArray();
    }
}
