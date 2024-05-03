from sklearn.ensemble import RandomForestRegressor


def getModel(model_name, configs):
    """ get the model with params specified from the configs"""

    if model_name == "random_forest":
        return getRandomForest(configs)
    elif model_name == "linear_regression":
        return getLinearRegression()
    elif model_name == "logistic_regression":
        return getLogisticRegression()
    
def getRandomForest(cfg):
    """
    return RandomForestRegressor model with itsparams

    params:
        inputs:
        -----------
        cfg: dict : hyperparameters for the model

        output:
        -----------
        model : RandomForestRegressor

    """
    params = cfg["random_forest"]

    return RandomForestRegressor(**params)
    

def getLinearRegression(cfg):
    pass

def getLogisticRegression(cfg):
    pass
    



import os
import json
import hydra

@hydra.main(config_path= "../configs", config_name="config.yaml", version_base=None)
def getconfigs(config):
    rf_config = os.path.abspath("MLOPs_workflow//src//project//modelling//rf_config.json")
    # with open(rf_config, "w+") as fp:
    #     json.dump(dict(config.model.models.items()), fp)
    print(dict(config.model.models.items()))


getconfigs()