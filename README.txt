The goal of this repository is to provide the code related to the alignment of simulated telescopes from CTAO using Reinforcement Learning.
The layout generally is focused on creating the tools needed to easily create and iterate agents and their environments, as well as good design to categorize and sort already trained agents and their environments.


The structure of this repository is as follows:

- The 'Environments' folder will contain all environments that the agents can be trained in. The environments will follow the Reinforcement Learning API specified in Stable-Baselines-3 (SB3)

- The 'Models' folder will contain all models. Each model is composed of several things: a reference to the environment it was trained in, some code that evaluates the agents (for example a pygame visualization of the agent in action), some code that creates the agent (this is where hyperparameters/architecture are defined), the agent file itself, a text file explaining the model, and might also contain a tensorboard log of the training curves.

- The 'Experiments' folder will contain all experiments. The purpose of an experiment is to train some or many models that are very similar to each other. Each experiment is composed of several things: some code that generated each model (the training code), a text file explaining the experiment, and every model that was trained. Here, a model is defined a bit differently than above. Each model contains the agent file, some code for evaluation (visualizaiton), and might also contain a tensorboard log of the training curve.

- The 'PSCT' folder will contain all files that are needed to specify the optics simulation of the prototype Schwarzchild-Couder telescope (pSCT). This includes things like response matrices and other useful stuff.