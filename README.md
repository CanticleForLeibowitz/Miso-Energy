The complete workflow is:

Raw Queue Data
      ↓
Read Data
      ↓
Data Manipulation
      ↓
Date Extraction
      ↓
MW Transformation
      ↓
Feature Selection
      ↓
Logistic Regression
      ↓
Probability Calibration
      ↓
Expected MW
      ↓
Model Evaluation
      ↓
Active Project Forecast
      ↓
CSV Output



log_mw

The project MW value is transformed into logarithmic form.

This helps reduce the influence of very large projects and can make the relationship between project size and realization probability easier for the model to learn.
type_clean
region
service
IA_phase_clean

These represent characteristics such as:

    Project type
    Geographic region
    Service/utility
    Interconnection or IA phase

Categorical variables are converted into numerical variables using one-hot encoding.

The primary model is logistic regression.

The logistic regression first produces a raw realization probability.
The model therefore preserves uncertainty rather than forcing every project into a simple Yes/No prediction.
Raw model probabilities are calibrated using isotonic regression.
Logistic Regression
        ↓
Raw Probability
        ↓
Isotonic Regression
        ↓
Calibrated Probability
This is the primary output of the model.

For every project:

Expected MW=Total MW×Calibrated Probability.
