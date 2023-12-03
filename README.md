# CBD: A Certified Backdoor Detector Based on Local Dominant Probability ([paper link](https://openreview.net/forum?id=H1CQZqpgdQ&referrer=%5Bthe%20profile%20of%20Zhen%20Xiang%5D(%2Fprofile%3Fid%3D~Zhen_Xiang1)))

## Overview

Thanks for your interest in our work. This is the first *certified* backdoor detector (CBD) that provides:

* A detection inference
* The conditions under which a backdoor attack is guaranteed to be detectable
* A probabilistic upper bound on the false positive rate

This repository contains the implementation details of CBD, including both the detection and certification procedures. In the following, we reproduce the main results in Section 5.1 of our paper.

## Procedure

### Set up

Train backdoored models and benign models for evaluation. Dataset can be chosen from 'cifar10', 'svhn', and 'gtsrb'.

  ```python
  python train_attack.py --dataset cifar10
  python train_clean.py --dataset cifar10
  ```
### Detection

First, train shadow models.
  ```python
  python train_shadow.py --dataset cifar10
  ```

Second, compute LDP statistics for calibration. The choice of 'sigma' can be different based on our Figure 2.
  ```python
  python get_stats.py --dataset cifar10 --sigma 1.14 --stat_type shadow
  ```
Third, obtain certified detection results (CTPR, TPR, and FPR).
  ```python
  python certified_detection.py --dataset cifar10 --sigma 1.14
  ```
Similar results as in Figure 2 will be obtained.

## Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{xiang2023cbd,
      title={{CBD}: A Certified Backdoor Detector Based on Local Dominant Probability},
      author={Zhen Xiang and Zidi Xiong and Bo Li},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
    }
