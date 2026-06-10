#import "@preview/touying:0.7.3": *
#import themes.university: *
#import "@preview/cetz:0.5.0"
#import "@preview/fletcher:0.5.8" as fletcher: edge, node
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.6.0": *
#import cosmos.clouds: *
#import "slides-template.typ": full-slide, split-slide
#show: show-theorion

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

// #set text(font: "Times New Roman")
#set page(margin: (top: 1.5cm, bottom: 1.5cm, left: 2cm, right: 2cm))

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(frozen-counters: (theorem-counter,)),
  config-info(
    title: [Credit Risk Analysis and Prediction],
    subtitle: [Using Machine Learning for Credit Card Fraud Detection],
    author: [Chanaka Gunawardana],
    institution: [CardiffMet ID: st20249383 \ ICBT ID: CL/MCSDS/CMU/10/18],
  ),
)

// #set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

== Dataset Overview

#split-slide[
  - *Dataset:* 25,134 applicant records
  - *Target:* Binary — default (1) or not (0)
  - *Imbalance:* 98.32% non-defaulters vs 1.68% defaulters (ratio 58:1)
][
  #figure(image("./img/target-vs-count.svg", width: 70%), caption: [Target variable distribution])
]

== Data Cleaning

#full-slide[
  - Removed unrealistic ages (>100)
  - Imputed missing values:
    - *YEARS_EMPLOYED* and *FAMILY_SIZE* → median
    - *INCOME_TYPE* → mode
  - Dropped *FLAG_MOBIL* (no variance)
]

== Exploratory Data Analysis — Univariate

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    Income is heavily right-skewed.
    Median income: 2,610,000.
    #figure(image("./img/income_dist.svg", width: 65%), caption: [Income Distribution])
  ],
  [
    Age is approximately normal.
    Mean age: 43.2 years.
    #figure(image("./img/age_dist.svg", width: 65%), caption: [Age Distribution])
  ],
)

== EDA — Bivariate

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    No strong correlation between income and age.
    #figure(image("./img/income_age.svg", width: 55%), caption: [Income vs Age])
  ],
  [
    No significant difference in age/income quartiles between classes.
    #figure(image("./img/age_box.svg", width: 55%), caption: [Age by Target])
  ],
)

== EDA — Correlation

#split-slide[
  - Low correlations among predictors
  - ML models (Random Forest, XGBoost) capture interactions effectively
][
  #figure(image("./img/corr.svg", width: 75%), caption: [Correlation matrix])
]

== Modelling

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Logistic Regression
    The imbalanced dataset causes standard logistic regression to predict only the majority class.
    #figure(image("./img/logistic_regression_result.png", width: 75%), caption: [Logistic regression result])
  ],
  [
    === Balanced Logistic Regression
    Weighting the minority class significantly improves recall for defaulters.
    #figure(image("./img/balanced_logistic_regression_result.png", width: 75%), caption: [Balanced logistic regression])
  ],
)

== Machine Learning Models

Models: RandomForest, GradientBoosting, XGBoost (standard + balanced versions).

Balanced Logistic Regression achieved the best recall with comparable ROC-AUC.
#figure(image("./img/model_comparison.png", width: 75%), caption: [Model comparison])


== Feature Engineering

#full-slide[
  Domain-informed features:

  - *INCOME_PER_PERSON* — Income / Family size
  - *AGE_AT_FIRST_JOB* — Age − Years employed
  - *TENURE_RATIO* — Years employed / Age
  - *CHILDREN_RATIO* — Children / Family size
  - *BEGIN_MONTH (binned)* — Application month grouped

  These features contributed ~15% increase in feature importance.
]

== Hyperparameter Tuning

#full-slide[
  RandomizedSearchCV with cross-validation:

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      *Balanced RF:* `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `sampling_strategy`
    ],
    [
      *XGBoost:* `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda`, `reg_alpha`
    ],
  )

  Threshold optimized on validation set to maximize recall @ precision ≥ 10%.
]

== Model Stacking

#full-slide[
  Stacking ensemble:

  - *Base learners:* Balanced Random Forest + Weighted XGBoost
  - *Meta-learner:* Logistic Regression (`passthrough=True`)

  The meta-learner learns to combine complementary strengths of both tree-based models.
]

== SHAP Analysis

#full-slide[
  Top contributors to predictions:
  - Time-based: *BEGIN_MONTH*, *TENURE_RATIO*
  - Demographic: *AGE*, *FAMILY_TYPE*
  - Financial: *INCOME_PER_PERSON*

  Key engineered features dominated feature importance rankings.
]

== Summary of Results

#full-slide[
  #align(center)[
    #table(
      columns: (auto, auto),
      inset: 10pt,
      [*Improvement*], [*Impact*],
      [Feature Engineering], [~15% boost in importance],
      [Recall-oriented scoring], [Optimized for credit risk],
      [Hyperparameter Tuning], [Better generalization],
      [Smart Thresholding], [PR-curve based selection],
      [Ensemble Stacking], [Leverages model complementarity],
    )
  ]

  Best model exported via Joblib and deployed.
]

== Application Development — Architecture

#full-slide[
  Monorepo with Turborepo + Bun:

  - *Frontend* (`apps/web`): React 19, Vite, TypeScript, Tailwind CSS, shadcn/ui
  - *Backend* (`apps/backend`): Python FastAPI, scikit-learn, joblib
  - *Shared UI* (`packages/ui`): Reusable shadcn/ui components

  Three main views: Data Overview, EDA, Modelling & Prediction.
]

== Backend & Deployment

#full-slide[
  *FastAPI endpoints:* `GET /` (health), `POST /predict` (fraud)

  Backend loads the stacking ensemble via joblib, applies the same preprocessing pipeline, and returns:

  - Binary prediction · Raw probability · Threshold · Classification label

  *Response:* `{"prediction": 0, "probability": 0.23, "threshold": 0.5, "classification": "Legitimate"}`
]

== Conclusion

#full-slide[
  - Fraud detection system handling extreme class imbalance (1.68% fraud rate)
  - Stacking ensemble (Balanced RF + XGBoost + Logistic Regression) — best recall-precision trade-off
  - Full-stack application deployed for real-time predictions
  - SHAP analysis provided model interpretability

  #align(center + bottom)[
    #text(size: 20pt)[*Thank You*]
  ]
]

==

#full-slide[
  #align(center + horizon)[
    #text(size: 24pt, weight: "bold")[Project Video]
    #v(1cm)
    #text(size: 16pt)[Watch the full walkthrough on YouTube:]
    #v(1cm)
    #link("https://youtu.be/Zh0hDqWQ7XM")[#text(size: 16pt, fill: blue)[https://youtu.be/DkwJBOFcs_E]]
  ]
]
