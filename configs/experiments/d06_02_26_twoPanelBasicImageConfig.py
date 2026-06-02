from configs.experiments.d06_01_26_onePanelBasicImageConfig import config as one_panel_config

config = one_panel_config
config["env_params"]["n_panels"] = 2
config["train_config"]["tensorboard_log"] = "runs/06-02-2026_twoPanelImageCodeCheck/"
config["model_save_path"] = "runs/06-02-2026_twoPanelImageCodeCheck/agent"