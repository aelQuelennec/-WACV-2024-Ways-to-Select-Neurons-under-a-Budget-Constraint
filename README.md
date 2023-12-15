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

## 2. Launching the code with python

* Install requirements.
* Read and modify the run settings in 'NEq/configs/transfer.yaml' file (or create a new configuration file which can be passed in as an input of the parser).
* Launch the code with:

`python train_classification.py`

## 3. Launch the code as sweep with wandb

* It is strongly advised to run the code through the wandb sweep tool as it allows for parallel and autonomous launching of many runs alongside easy monitoring of results.
* An example of sweep configuration is provided in 'wandb_sweep_example.yaml'.

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
