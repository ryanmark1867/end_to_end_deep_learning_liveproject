# end_to_end_deep_learning_liveproject
- active repo for end to end deep learning liveproject for Manning 
- related to Manning book **Deep Learning with Structured Data** https://www.manning.com/books/deep-learning-with-structured-data
- the code in this repo cleans up the [Airbnb NYC dataset](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data), explores the data, creates and trains a DL model, and deploys the model

## Directory structure
- **data** - processed datasets and pickle files for intermediate datasets
- **models** - saved trained models
- **notebooks** - code
- **help** - versions of code with only comments and function signatures
- **pipelines** - pickled pipeline files

## To exercise the code

- Copy Airbnb NYC dataset to data directory in cloned repo
- Run [data_cleanup.ipynb](https://github.com/ryanmark1867/end_to_end_deep_learning_liveproject/blob/master/notebooks/data_cleanup.ipynb) notebook with appropriate config settings
- Run [model_training.ipynb](https://github.com/ryanmark1867/end_to_end_deep_learning_liveproject/blob/master/notebooks/model_training.ipynb) notebook on output of data_cleanup notebook with appropriate config settings
- Run [flask_server.py](https://github.com/ryanmark1867/end_to_end_deep_learning_liveproject/blob/master/deploy_web/flask_server.py) module with appropriate config settings


## Background

- [Airbnb NYC dataset](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
- [main repo for **Deep Learning with Structured Data**](https://github.com/ryanmark1867/deep_learning_for_structured_data)