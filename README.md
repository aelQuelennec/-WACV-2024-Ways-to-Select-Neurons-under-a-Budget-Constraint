# Towards On-device Learning on the Edge: Ways to Select Neurons to Update under a Budget Constraint - WACV 2024, SCIoT workshop

### [[arXiV]](https://arxiv.org/abs/2312.05282)

Official repository for research presented on dynamic neuron selection at WACV 2024 SCIoT workshop.

![](figures/teaser-2.pdf)

# Reproduce our results

## 1. Downloading the datasets

* [CUB-200](https://data.caltech.edu/records/65de6-vp158)
* [Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* [Pets-37](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [VWW](https://github.com/Mxbonn/visualwakewords)

The corresponding root paths must be modified in the "NEq_configs.yaml" and "NEq/core/dataset/dataset_entry.py" files.

These datasets can be divided into train, test, and validation sets according to their definitions using code files stored in the [train_test_split](./train_test_split) folder.

## 2. Downloading MCUNet and Proxyless

```
cd NEq/classification/models/

git clone https://github.com/mit-han-lab/mcunet.git

cd mcunet

git checkout 8dd3347f4f47addcf68f9a220a1e2dbe4deae113

cd ../../../../
```

## 3. Launching the code 
### With python

* Install requirements.
* Read and modify the run settings in [NEq/configs/transfer.yaml](./NEq/configs/transfer.yaml) file (or create a new configuration file which can be passed in as an input of the parser).
* Launch the code with: `python train_classification.py`

### With WanDB

* It is strongly advised to run the code through the wandb sweep tool as it allows for parallel and autonomous launching of many runs alongside easy monitoring of results.
* Examples of WanDB config files can be found in [policies](./policies/) folder.
* Base on the variables defined in [NEq_configs.yaml](./NEq_configs.yaml) to define a new configuration
* To launch the code as a sweep with Wandb, follow these steps:

- **Step 1:** Ensure that `wandb_sweep` is set to `1` in [NEq/configs/transfer.yaml](./NEq/configs/transfer.yaml).

- **Step 2:** In terminal, navigate to the `-WACV-2024-Ways-to-Select-Neurons-under-a-Budget-Constraint` folder, and initiate the Wandb sweep using the command:

`wandb sweep --project project_name link_to_wandb_sweep_config_file`

Replace `project_name` with the name of your project on the Wandb system, and `link_to_wandb_sweep_config_file` with the path to your Wandb sweep config file. This command will create a sweep on the Wandb system with an ID of `sweep_id`.

- **Step 3:** After the sweep is created, use the following command to execute each Wandb agent:

`wandb agent username/project_name/sweep_id`

Each agent will be responsible for running one configuration generated from the Wandb sweep config file.

#### Usage Example
- **Step 2:**: `wandb sweep --project test ./policies/test_policy.yaml` create a sweep whose id is `xxxx`
- **Step 3:**: Use `wandb agent username/test/xxxx` to run an agent



## 4. Load the best model results

After training using the Wandb sweep, the best models are saved. To log the results of these models into an Excel file, use the following command: 

`python3 NEq/load_best_model.py -w link_to_wandb_sweep_config_files`

Replace `link_to_wandb_sweep_config_files` with the path to the Wandb sweep file(s).

For example:

`python3 NEq/load_best_model.py -w policies/c10/c10_mbv2.yaml policies/c10/c10_resnet18_baseline.yaml`

This command will load the best model results of pretrained MobileNetV2 with schemes 1, 3, and 5 using random initialization and three neuron update methods: SU, Velocity, and Random; and pretrained ResNet18 with scheme 7 (baseline) on the CIFAR-10 dataset. The output is stored in `output.xlsx` by default.

Alternatively, you can specify the output file to store the results by adding `-o link_to_output` at the end of the command. Supported formats include: `.xlsx`, `.xlsm`, `.xltx`, and `.xltm`.

For example:

`python3 NEq/load_best_model.py -w policies/c10/c10_mbv2.yaml policies/c10/c10_resnet18_baseline.yaml -o ABC.xlsx`

Then, the results will be logged into `ABC.xlsx`.


## Citation

```
@InProceedings{Quelennec_2023_WACV,
    author    = {Aël Quélennec, Enzo Tartaglione, Pavlo Mozharovskyi, Van-Tam Nguyen},
    title     = {Towards On-device Learning on the Edge: Ways to Select Neurons to Update under a Budget Constraint},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2023},
}
```
