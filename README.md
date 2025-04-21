# Molecular Graph Transformer PlusPlus (MGT-PlusPlus)

![results](https://github.com/rbSparky/MGT-PlusPlus/blob/main/output%20(11).png)

This repository contains an enhanced implementation of the Molecular Graph Transformer (MGT) for predicting material properties. MGT-PlusPlus builds upon the original MGT architecture with several key improvements focused on efficiency, speed, and stability, while maintaining or improving predictive performance.

**Key Improvements in MGT-PlusPlus:**

*   **Reduced Parameter Size:** The model size has been significantly reduced from approximately 30 million parameters to around 15 million.
*   **Enhanced Efficiency:** Incorporates Linear Attention (Linformer) and Low-Rank transformations in key layers for improved computational efficiency, especially with larger graphs.
*   **Faster Training:** Various PyTorch and cuDNN optimizations, along with architectural changes, lead to significantly faster training times per epoch.
*   **Improved Stability:** Achieves comparable or better accuracy with lower variance compared to the original MGT, particularly with reduced ALIGNN and GNN layer counts.
*   **Optimized Architecture:** State-of-the-art performance can be achieved with a more streamlined architecture, specifically using `ALIGNN=1` and `GNNs=1` layers per encoder, unlike the original paper which used 2 of each.

There are two functionalities provided with this package:

- Train and Test an MGT-PlusPlus model for your own dataset
- Run pretrained models to predict material properties for new materials

## Table of Contents
* [Usage and Examples](#-usage-and-examples)
  * [Installation](#-installation)
  * [Dataset](#-dataset)
  * [Using Pre-Trained Models](#-using-pre-trained-models)
  * [Training and Testing your own model](#-training-and-testing-your-own-model)
* [Technical Improvements](#-technical-improvements)
* [Funding](#-funding)

 <a name="usage"></a>
# Usage and Examples
-------------------------

<a name="install"></a>
## Installation
-------------------------
First create a conda environment:
Install the miniconda environment from https://docs.conda.io/en/latest/miniconda.html or the
anaconda environment from https://www.anaconda.com/products/distribution

Now create a conda environment and activate it (substitute my_env with your preferred name):
```
conda create -n my_env
conda activate my_env
```

Now install the necessary libraries into the environment:

- [Pytorch](https://pytorch.org/)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

- [Fabric](https://lightning.ai/docs/fabric/stable/)
```
conda install lightning -c conda-forge
```

- [Deep Graph Library](https://www.dgl.ai/)
```
conda install dgl -c dglteam
```

- [Pymatgen](https://pymatgen.org/)
```
conda install pymatgen -c conda-forge
```
Additionally, install the `linformer` library:
```
pip install linformer
```
And `torch-scatter` for the PAMNet adaptation:
```
pip install torch-scatter
```


<a name="dataset"></a>
## Dataset
-------------------------
A user needs the following to set-up a dataset to train, test and run inference using the model
(all of this should be inside the same directory):

1. `id_prop.csv` with name of the files of each structure and corresponding truth value/s for
   each structure,
2. `atom_init.json` a file to initialize the feature vector for each atom type. (can be found
   in examples)
3. A folder contatining the structure files (accepted formats: `.cif`, `.xyz`, `.pdb`, `POSCAR`)

An example dataset can be found in [examples/example_data](examples/example_data), testing and
training dataset have to be saved in different folders, each with all three components.

<a name="pretrain"></a>
## Using Pre-Trained Models
-------------------------
All the pre-trained models can be found in the [pretrained](pretrained), and they are saved with
name of the task/dataset on which they were trained.

The [run.py](run.py) document can be used to get predictions using the pre-trained or
custom-trained models. An example of using a pretrained model to predict the BANDGAP, HOMO and
LUMO of the files in the example dataset (found in [examples/example_data](examples/example_data)
) is shown below:

```
run.py --root ./examples/example_data/ --model_path ./pretrained/ --model_name qmof.ckpt --out_dims 3 --out_names BANDGAP HOMO LUMO
```

Help for the [run.py](run.py) file and its command line arguments can be obtained using ```
run.py -h ```

<a name="test_train"></a>
## Training and Testing your own model
-------------------------

### Training

To train your own model you'll first need to have made a [custom dataset](#-dataset), you can
then run the training and validation by running:

```
training.py --root ./examples/example_data/ --model_path ./saved_models/
```

You can specify the train and validation splits of the dataset by running:

```
training.py --root ./examples/example_data/ --model_path ./saved_models/ --train_split 0.8 --val_split 0.2
```

or the splits can also be set as absolute values (example assumes a dataset with 100 systems)

```
training.py --root ./examples/example_data/ --model_path ./saved_models/ --train_split 80 --val_split 20
```

**Note on Hyperparameters:** Due to the architectural enhancements in MGT-PlusPlus, comparable or improved performance can be achieved with fewer ALIGNN and GNN layers per encoder than the original MGT paper. We recommend starting with `--n_alignn 1 --n_gnn 1` for a significantly faster and smaller model (approx. 15M parameters) that often outperforms the original MGT (approx. 30M parameters with `--n_alignn 2 --n_gnn 2`).

after running the [training.py](training.py) file, you will obtain:

- ```model.ckpt```: contains the MGT-PlusPlus model at the last epoch (stored in the ```--model_path``` directory)
- ```lowest.ckpt```: contains the MGT-PlusPlus model with the lowest validation error (stored in the ```--model_path``` directory)
- ```results.csv```: contains the training and validation losses (stored in the ```--save_dir```
  directory, if no ```--save_dir``` is specified it will save the results in ```.
  /output/train/```)

Help for the [training.py](training.py) file and its command line arguments can be obtained using ```
training.py -h ```

### Testing

To test your own model you'll first need to have made a [custom dataset](#-dataset), you can
then run the testing by running:

```
testing.py --root ./examples/example_data/ --model_path ./saved_models/ --model_name model.ckpt
```

after running the [testing.py](testing.py) file, you will obtain:

- ```results.csv```: contains the test results (structure ID, target value, predicted value,
  overall error, per property error) for each structure in the test database (stored in
  the ```--save_dir```
  directory, if no ```--save_dir``` is specified it will save the results in ```.
  /output/test/```)

Help for the [testing.py](testing.py) file and its command line arguments can be obtained using ```
testing.py -h ```

<a name="technical-improvements"></a>
# Technical Improvements
-------------------------
MGT-PlusPlus incorporates several technical advancements to enhance the original MGT:

*   **Linear Attention (Linformer):** The standard quadratic self-attention mechanism in the Transformer encoder has been replaced with a linear attention variant based on the Linformer architecture. This reduces the computational complexity with respect to sequence length (number of atoms), leading to faster processing of larger structures.
*   **Low-Rank Transformations:** Linear layers within the ALIGNN layers, MLP blocks, and other parts of the network have been replaced or augmented with Low-Rank transformations (inspired by LoRA). This significantly reduces the total number of trainable parameters and improves training speed and memory usage.
*   **PAMNet Adaptation:** The core Edge-Gated Graph Convolution (EGGC) layers have been adapted based on principles from the PAMNet architecture, potentially improving message passing efficiency and effectiveness.
*   **Optimized Training Pipeline:** The training script includes various optimizations such as:
    *   Enabling `torch.backends.cudnn.benchmark` and `torch.backends.cuda.matmul.allow_tf32` for speedups on compatible hardware.
    *   Explicit memory management calls (`gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`) to mitigate memory fragmentation and improve stability during training, especially with varying graph sizes.
    *   Support for mixed precision training via Lightning Fabric.

These combined improvements allow MGT-PlusPlus to achieve strong performance with a smaller model size and faster training compared to the original MGT.
