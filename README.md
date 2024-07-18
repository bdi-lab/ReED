## PAC-Bayesian Generalization Bounds for Knowledge Graph Representation Learning

This is the official code and data of the following [paper](https://arxiv.org/abs/2405.06418):
> Jaejun Lee, Minsung Hwang, and Joyce Jiyoung Whang, PAC-Bayesian Generalization Bounds for Knowledge Graph Representation Learning, The 40th International Conference on Machine Learning (ICML), 2024.

All codes are written by Jaejun Lee (jjlee98@kaist.ac.kr). When you use this code or data, please cite our paper.

```bibtex
@inproceedings{reed,
	author={Jaejun Lee and Minsung Hwang and Joyce Jiyoung Whang},
	title={{PAC}-Bayesian Generalization Bounds for Knowledge Graph Representation Learning},
	booktitle={arXiv preprint},
	year={2024},
	pages={}
}
```

## Requirements

We used python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```setup
pip install -r requirements.txt
```


## Training & Evaluation

We used NVIDIA NVIDIA GeForce RTX 2080 Ti for all our experiments. It takes less than 4 minutes for a single run.

The commands we used to get the results in our paper:

### FB15K237

```python
python train.py --data_path ./data/ --dataset_name FB15K237_sampled --decoder <decoder_type> -m 0.5 -lr <learning_rate> -L <number_of_RAMP_layers> -d 96 -phi LeakyReLU -rho Identity -psi Identity -s <value_of_s> --aggr <aggregator_type> --seed <random_seed> -e 2000 -b 1
```

<learning_rate>: 0.0003 (RAMP+TD) or 0.0005 (RAMP+SM)

<decoder_type>: Translational_Distance or Semantic_Matching

<aggregator_type>: mean or sum

<number_of_RAMP_layers>: 1, 2, or 3

<value_of_s>: 10.0, 15.0, or 20.0

<random_seed>: 0, 10, 20, 30, 40, 50, 60, 70, 80, or 90

### FB15K237 w/ text features

```python
python train_txt.py --data_path ./data/ --dataset_name FB15K237_sampled_txt --decoder <decoder_type> -m 0.5 -lr <learning_rate> -L 2 -d <dimension> -phi LeakyReLU -rho Identity -psi Identity -s 15.0 --aggr mean --seed <random_seed> -e 2000 -b 1
```

<learning_rate>: 0.0002 (RAMP+TD) or 0.00005 (RAMP+SM)

<decoder_type>: Translational_Distance or Semantic_Matching

<dimension>: 64, 96, or 128

<random_seed>: 0, 10, 20, 30, 40, 50, 60, 70, 80, or 90




### CoDEx-M

```python
python train.py --data_path ./data/ --dataset_name CoDEx-M_sampled --decoder <decoder_type> -m 0.5 -lr 0.0005 -L <number_of_RAMP_layers> -d 64 -phi LeakyReLU -rho Identity -psi Identity -s <value_of_s> --aggr <aggregator_type> --seed <random_seed> -e 2000 -b 1
```

<decoder_type>: Translational_Distance or Semantic_Matching

<aggregator_type>: mean or sum

<number_of_RAMP_layers>: 1, 2, or 3

<value_of_s>: 10.0, 15.0, or 20.0

<random_seed>: 0, 10, 20, 30, 40, 50, 60, 70, 80, or 90

### 

### UMLS-43

```python
python train.py --data_path ./data/ --dataset_name UMLS-43 --decoder <decoder_type> -m 0.75 -lr <learning_rate> -L <number_of_RAMP_layers> -d 48 -phi LeakyReLU -rho Identity -psi Identity -s <value_of_s> --aggr <aggregator_type> --seed <random_seed> -e 2000 -b 1
```

<learning_rate>: 0.0002 (RAMP+TD) or 0.0005 (RAMP+SM)

<decoder_type>: Translational_Distance or Semantic_Matching

<aggregator_type>: mean or sum

<number_of_RAMP_layers>: 1, 2, or 3

<value_of_s>: 10.0, 12.5, or 15.0

<random_seed>: 0, 10, 20, 30, 40, 50, 60, 70, 80, or 90