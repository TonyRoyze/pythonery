#import "@preview/classy-tudelft-thesis:0.1.0": *
#import "@preview/physica:0.9.6": *
#import "@preview/unify:0.7.1": num, numrange, qty, qtyrange
#import "@preview/zero:0.5.0"

= Advanced Data Analysis

== Initail models

=== Logistic Regression

The Logistic Regression is the simplest model for a binary classification problem. It is a linear model that uses a logistic function to map the output of a linear equation to a probability between 0 and 1. The results are as follows:

#figure(
  image("./../img/logistic_regression_result.png", width: 100%),
  caption: [Logistic Regression Result],
) <fig:logistic-regression-result>

It can be seen that the model is not able to predict the positive class well. This is expected as the dataset is highly imbalanced. 

== Balanced Logistic Regression

#figure(
  image("./../img/balanced_logistic_regression_result.png", width: 100%),
  caption: [Balanced Logistic Regression Result],
) <fig:balanced-logistic-regression-result>

By Balancing the dataset, by weighting the minority class we can see that the model is able to predict the positive class significantly well. To see if there could be an improvement Machine Learning Models were trained.

== Machine Learning Models

A few Machine Learning Models including:
- RandomForest
- GradientBoosting
- XGBoost
including thier balanced versions were trained. The results are as follows:

#figure(
  image("./../img/model_comparison.png", width: 100%),
  caption: [Model Comparison],
) <fig:model-comparison>


By looking at the table of model comparisons it can be seen that the the RandomForest and Balanced RandomForest models seem to have higher Test F1 scores and ROC-AUC values they are bad at predicting defaulters accurately. Same goes for XGBoost. However GradientBoosting seems to be better at Precision as in being being able to reduce the number of false negative  is able to predict the positive class significantly well. Eventhough the Logistic Regression (Balanced) has a slighly lower Test F1 Score and Precision it has the best Recall and comparatively similar ROC-AUC measure as well, which in this case would make it the best model out of the set of models.

=== Balanced Machine Learning Models

To see if improvements could be made 


=== Model Stacking

== Summary