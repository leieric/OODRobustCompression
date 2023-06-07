# RobustCompression
Code for the paper:

Lei, Eric, Hamed Hassani, and Shirin Saeedi Bidokhti. "[Out-of-Distribution Robustness in Deep Learning Compression,](https://arxiv.org/abs/2110.07007)" in ICML-21 Workshop on Information-Theoretic Methods for Rigorous, Responsible, and Reliable Machine Learning, 2021.

# Description
Training and evaluation for the end-to-end models (standard and OOD robust, i.e., solving the min-max problem) is contained in `robust_compression/`. The structured training/eval is contained in `structured_compression/`. 

## Evaluation of pre-trained models
To generate plots like those in the paper, see `structured_compression/eval_wass_ball.ipynb` and `structured_compression/eval_rotations.ipynb`. `eval_wass_ball.ipynb` loads a trained model, estimates different values of rho given the gammas defined, and evaluates the worst-case distortion for the ball at radius rho. `eval_rotations` loads the data augmentation-trained models, DRO models, and the angle prediction (structured) models described in Section 5 of the paper to generate Figure 6. 

## Training
To train models, run `bash scripts/{exp}.sh` in either of the directories, where `exp` is one of: 
- `compress_ae_data_augmentation_rotation`: standard end-to-end training, where half the data is clean, half is randomly rotated
- `compress_ae_data_augmentation`: standard end-to-end training, where half the data is clean, half is Gaussian noise perturbed
- `compress_ae_mnist_standard`: standard end-to-end training on clean data
- `compress_ae_wass_ball`: end-to-end model with min-max training over the wasserstein ball
- `compress_ae_rotations`: end-to-end model with min-max training over the rotation angles
- `structured`: structured model from Figure 2 for wasserstein ball
- `structured_groupshift_rotation`: structured model from Figure 2 for rotation angles


# Citation

    @article{lei2021out,
        title={Out-of-Distribution Robustness in Deep Learning Compression},
        author={Lei, Eric and Hassani, Hamed and Bidokhti, Shirin Saeedi},
        journal={arXiv preprint arXiv:2110.07007},
        year={2021}
    }
