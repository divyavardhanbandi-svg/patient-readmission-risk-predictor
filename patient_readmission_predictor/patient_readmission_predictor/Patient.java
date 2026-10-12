/**
 * One hospital discharge record. Binary flags are stored as 0.0 / 1.0.
 * readmitted: 1 if the patient was readmitted within 30 days, otherwise 0.
 */
public record Patient(
        double age,
        double lengthOfStay,
        double priorAdmissions,
        double medications,
        double diagnoses,
        double diabetes,
        double heartFailure,
        double emergencyAdmission,
        int readmitted) {

    public static final String[] FEATURE_NAMES = {
            "Age", "Length of stay (days)", "Prior admissions (1 yr)", "Medications",
            "Diagnoses", "Diabetes", "Heart failure", "Emergency admission"
    };

    public double[] features() {
        return new double[]{age, lengthOfStay, priorAdmissions, medications,
                diagnoses, diabetes, heartFailure, emergencyAdmission};
    }
}
