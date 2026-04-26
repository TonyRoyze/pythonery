#import "@preview/classy-tudelft-thesis:0.1.0": *
#import "@preview/physica:0.9.6": *
#import "@preview/unify:0.7.1": num, numrange, qty, qtyrange
#import "@preview/zero:0.5.0"

= Introduction

This project focused on developing a machine learning model to predict credit card default risk. The primary objective was to identify applicants who are likely to default on their credit card payments.

== Dataset Overview



#let fig = [
  #figure(
    image("./../img/target-vs-count.svg", width: 6cm),
    caption: [Counts of Target Variable],
  ) <fig:target-vs-count>
]

#wrap-content(
  fig,
  [The dataset consisted of 25,134 applicant records with 17 features, including variables such as gender, car ownership, housing status, employment details, income, and education. The target variable, labeled *TARGET*, indicated whether an applicant defaulted (1) or not (0). A key challenge identified early in the analysis was the severe class imbalance in the dataset.
  
  
  Non-defaulters made up 98.32% of the data (24,712 records), while defaulters accounted for only 1.68% (422 records), resulting in an imbalance ratio of approximately 58:1. This imbalance significantly influenced model design and evaluation strategies.],
)

== Data Cleaning

Data cleaning involved multiple steps to ensure data quality. Unrealistic age values above 100 were removed. Missing values in *YEARS_EMPLOYED* and *FAMILY_SIZE* were handled using median imputation, while INCOME_TYPE was imputed using the mode. The *FLAG_MOBIL* feature was dropped because it contained no variance and therefore provided no predictive value.
