#import "@preview/classy-tudelft-thesis:0.1.0": *
#import "@preview/physica:0.9.6": *
#import "@preview/unify:0.7.1": num, numrange, qty, qtyrange
#import "@preview/zero:0.5.0"

= Exploratory Data Analysis

== Univariate Analysis

The distribution of the two continuous variables, income and age, are shown in @fig:income-distribution and @fig:age-distribution respectively. The income distribution is heavy skewed to the right, with a median income of 2610000. The age distribution is approximately normal, with a mean age of 43.2 years. Therefore the choice of median income as a imputing value is justified. 

#let income_dist = [
  #figure(
    image("./../img/income_dist.svg", width: 7cm),
    caption: [Income Distribution],
  ) <fig:income-distribution>
]

#let age_dist = [
  #figure(
    image("./../img/age_dist.svg", width: 7cm),
    caption: [Age Distribution],
  ) <fig:age-distribution>
]


#grid(
  columns: (1fr, 1fr),
  rows: (auto, auto),
  gutter: 3pt,
  income_dist,
  age_dist,
)

== Bivariate Analysis

#let income_age = [
  #figure(
    image("./../img/income_age.svg", width: 5cm),
    caption: [Income vs Age Distribution],
  ) <fig:income-age>
]

#wrap-content(
  income_age,
  [Looking at the relationship between income and age, we cannot see that correlation between income and age.
  
  This would be natural as the income of a person is not directly related to their age. It could vary for many reasons such as 
  - Type of employment
  - Level of education
  - Experience
  ],
  align: (right),
)

#let age_box = [
  #figure(
    image("./../img/age_box.svg", width: 7cm),
    caption: [Age Box Plot by Target],
  ) <fig:age-box>
]

#let income_box = [
  #figure(
    image("./../img/income_box.svg", width: 7cm),
    caption: [Income Box Plot by Target],
  ) <fig:income-box>
]

#grid(
  columns: (1fr, 1fr),
  rows: (auto, auto),
  gutter: 3pt,
  age_box,
  income_box,
)

The two boxplots above show that there is no significant diference in the quartiles of the age and the income between defaulters and non-defaulters.

#let correlation_matrix = [
  #figure(
    image("./../img/corr.svg", width: 7cm),
    caption: [Correlation Matrix],
  ) <fig:correlation-matrix>
]


#wrap-content(
  correlation_matrix,
  [The correlation matrix shows the correlation between all the numeric features in the dataset. Which is important identify if there is a possibilty of having interactions amoung predictor varibles. Which would be important for ordinary statistical models. However, for machine learning models such as random forests, it is not that important as they can capture interactions.
  ],
  align: (top), 
)
  