# Analysis

The approach being followed is very straightforward in the interest of time

- Perform some initial analysis on the data
- Pick two models; compare the results from both
- Pick the better model based on the inference
- Create a workflow/pipeline/function which can attach the model to itself; load in data and then output the result (Sorted in likelihood)

Steps - 

- Prediscovery file illustrates the various data analysis steps performed on the dataset

- Several models were then tried and a best model was then picked (Details in Prediscovery file)

- Analysis.py file showcases the best model training and a sample inference

- No argument parsers have been implemented for the file

- The dataset remains imbalanced as such; but the imbalanced data points to better testing scores (F1 score) and ergo, this was picked. However, provided more data the imbalance could tip off prediction towards the majority class and this should be closely monitored. For now, the imbalanced dataset works; but this is not a steadfast rule.

- More improvement measures are found in the Prediscovery file


