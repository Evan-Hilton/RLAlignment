import yaml
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn

# environments
from src.environments.basic_image_env import *
from src.environments.no_image_psctp12_env import *
from src.environments.no_image_one_centroid_obs import *
from src.environments.fine_alignment.naive_image_env import *
from src.environments.fine_alignment.all_dict_obs_env import *
from src.environments.fine_alignment.stacked_image_env import *
from src.environments.fine_alignment.all_dict_obs_env_diff_img import *
from src.environments.fine_alignment.stacked_image_with_action_env import *

# feature extractors
from src.architectures.everythingDict import *
from src.architectures.configurableCNN import *
from src.architectures.stackedImageWithActionDict import *

ENVIRONMENTS = {
    "BasicImageEnv": BasicImageEnv,
    "NoImagePSCTP12Env": NoImagePSCTP12Env,
    "NoImageOneCentroidObs": NoImageOneCentroidObs,
    "NaiveImageEnv": NaiveImageEnv,
    "StackedImageEnv": StackedImageEnv,
    "StackedImageWithActionEnv": StackedImageWithActionEnv,
    "AllDictObsEnv": AllDictObsEnv,
    "AllDictObsEnvDiffImg": AllDictObsEnvDiffImg
}
FEATURE_EXTRACTORS = {
    "ConfigurableCNN": ConfigurableCNN,
    "StackedImageWithActionDict": StackedImageWithActionDict,
    "EverythingDict": EverythingDict
}
VEC_ENV_CLS = {
    "SubprocVecEnv": SubprocVecEnv
}

"""
    Takes in a path to a yaml file, and loads it by
    creating all necessary python objects for the config
    and initializes a python dictionary to act as a config
    and returns it.
"""
def load_experiment_config(path):
    experiment_dict = yaml_import(path)
    experiment_dict["environment"] = ENVIRONMENTS[experiment_dict["environment"]]
    experiment_dict["vec_env_cls"] = VEC_ENV_CLS[experiment_dict["vec_env_cls"]]
    experiment_dict["policy_kwargs"] = dict(
        net_arch=dict(
            pi=experiment_dict["network"]["pi"],
            vf=experiment_dict["network"]["vf"] 
        ),
        activation_fn=nn.ReLU,
    )
    if "feature_extractor_class" in experiment_dict["network"]:
        fe_kwargs = dict(config=experiment_dict["network"]["feature_extractor_config"])
        experiment_dict["policy_kwargs"]["features_extractor_class"] = FEATURE_EXTRACTORS[experiment_dict["network"]["feature_extractor_class"]]
        experiment_dict["policy_kwargs"]["features_extractor_kwargs"] = fe_kwargs
    
    # add the custom feature extractor to the ppo paramaters
    experiment_dict["train_config"]["policy_kwargs"] = experiment_dict["policy_kwargs"]

    return experiment_dict

def yaml_import(path):
    with open(
        path,
        "r"
    ) as f:
        config = yaml.safe_load(f)
    return config