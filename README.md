# Personality types and traits - Examining and leveraging the relationship between different personality models for mutual prediction

This is the code for running the experiments and replicating the results in the following paper:

Radisavljević D., Rzepka R., and Araki K., Personality types and traits - Examining and leveraging the relationship between different personality models for mutual prediction, 2023. 

This code builds upon and uses as baseline the research done by

Gjurković M., Karan M., Vukojević I., Bošnjak M., and Šnajder J., PANDORA Talks: Personality and Demographics on Reddit, arXiv, 2020. 

With the data for the experiments being available on demand from: https://psy.takelab.fer.hr/datasets/all/

The code consists of the following modules:

* baselines.py - expansion of the code used as a baseline. Includes all the methods necessary for running experiments. 
* features.py - separate python file including information on the feature columns. baselines.py uses it in order to construct feature set.
* draw_graphs.ipynb - python notebook with code in order to create figures for the Google Trends API data
* preparation.ipynb - python notebook file including code testing for various correlations in the dataset as well as download of the data from the PushShift website.
* summarize_res.py - python code used in order to generate concise report of the experiments after their execution
* test_corr.ipynb - python notebook containig code that tests for correlation with LIWC measures

## Running the code

In order to run the code, you can do so by calling

```
python3 baselines.py  -data_path DATA_LOCATION -label allbig5 -tasktype regression -folds big5 -feats FEATURE_LIST -model MODEL_TYPE -variant LR-NP
```

Where DATA_LOCATION is directory where dataset is located, FEATURE_LIST represents one or list of features from the features.py file (it is also possible to use 1gram as input here) and MODEL_TYPE indicates which algorithm should be applied. The list of algorithms is as follows:

* l2 - Ridge Regressor
* lr-l - Lasso Regressor
* lr-e - ElasticNet
* lr-h - HuberRegressor
* lr-s - Epsilon-Support Vector Regression
* lr-b - XGBoost Regressor
