
# Lost Along the Way: Statistical Modelling of Youth Dropouts in Orienteering

## Author: Viktória Kostercová, Lund University 2026

**Abstract:**

Youth dropouts in organized sports present a significant challenge across different disciplines
worldwide. This thesis aims to uncover the driving factors behind this phenomenon in a specific
setting – among youth orienteers in Sweden – using large-scale data analysis and statistical
learning. The dataset used in this study consists of historical competition results spanning the
years 2012 to 2025.

Following extensive data pre-processing, two logistic regression models utilizing natural cubic
regression splines were developed. The optimal set of explanatory variables was selected via
a forward selection algorithm optimized using cross-validation based on the area under the
precision-recall curve (PR-AUC). Due to the trade-off between predictive accuracy and in-
terpretability, two distinct specifications are proposed: Model A, which maximizes predictive
power, and Model B, which focuses on the underlying driving factors for dropouts.

The empirical results demonstrate that while predicting human behavior from competition data
only is highly challenging, significant patterns can still be identified. Specifically, we uncover a
sharp peak in dropout risk around ages 17–18, a lower risk for athletes who start the sport at an
early age, and a U-shaped relationship revealing an higher dropout risk among both the lowest-
and highest-performing youth. Based on these findings, a set of recommendations is offered to
Swedish orienteering clubs and the Swedish Orienteering Federation to mitigate the dropout
rate. These include expanding mentorship for beginners beyond initial introductory courses,
establishing dedicated recreational activities for juniors within the DH18–20 categories, or
adapting coaching strategies to decrease competitive pressure.


**DISCLAIMER:** This public repository serves as an overview of the source code and the work I have done as part of my Master thesis at Lund University. However, it does not contain any of the data (raw or processed) or jupyter notebooks that were part of the data analysis. Please cite my work appropriately.


**Data pre-processing:**

* To obtain the raw data in a .parquet format, run *src/excel_to_parquet.py*
* Before forming the modelling dataset, set up desired parameters in *src/parameters.py*
* To form the modelling dataset, run *src/run_data_preparation.py.* The resulting datasets will appear in *data/processed/modelling*

**Feature selection:**

* In either *model/forward_selection.py* or *model/backward_selection.py* set up the desired candidate features which will serve as a baseline for either selection algorithm
* Choose which features will be attempted to be modelled as splines
* Run

**Modelling:**

* In *model/all_models.py* select desired variables which should be included in the model
* Choose which will be modelled linearly or as splines
* Run
* PR curves, regression coefficient evolutions and spline visualizations appear in *results/figures*
* Performance metrics of the model appear in *results/metrics*
