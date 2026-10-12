import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Generates synthetic patient records with a realistic readmission pattern. */
public class DataGenerator {

    private final Random rng;

    public DataGenerator(long seed) {
        this.rng = new Random(seed);
    }

    public List<Patient> generate(int n) {
        List<Patient> patients = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double age = clamp(Math.round(62 + rng.nextGaussian() * 15), 18, 95);
            double los = clamp(Math.round(4.5 + rng.nextGaussian() * 2.5), 1, 21);
            double prior = poisson(1.2);
            double meds = clamp(Math.round(8 + rng.nextGaussian() * 4), 1, 25);
            double diagnoses = clamp(1 + poisson(3.0), 1, 12);
            double diabetes = rng.nextDouble() < 0.30 ? 1 : 0;
            double heartFailure = rng.nextDouble() < 0.20 ? 1 : 0;
            double emergency = rng.nextDouble() < 0.45 ? 1 : 0;

            // Hidden "true" risk model used to create the labels
            double logit = -4.2
                    + 0.025 * (age - 60)
                    + 0.12 * los
                    + 0.45 * prior
                    + 0.06 * meds
                    + 0.10 * diagnoses
                    + 0.40 * diabetes
                    + 0.70 * heartFailure
                    + 0.50 * emergency;
            double p = 1.0 / (1.0 + Math.exp(-logit));
            int readmitted = rng.nextDouble() < p ? 1 : 0;

            patients.add(new Patient(age, los, prior, meds, diagnoses,
                    diabetes, heartFailure, emergency, readmitted));
        }
        return patients;
    }

    private int poisson(double lambda) {
        double limit = Math.exp(-lambda);
        double product = rng.nextDouble();
        int k = 0;
        while (product > limit) {
            k++;
            product *= rng.nextDouble();
        }
        return k;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
