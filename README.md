# Federated Transfer Learning For EEG Signal Classification

Authors: Ce Ju, Dashan Gao, Ravikiran Mane, Ben Tan, Yang Liu and Cuntai Guan

Published in: 42nd Annual International Conferences of the IEEE Engineering in Medicine and Biology Society in 
conjunction with the 43rd Annual Conference of the Canadian Medical and Biological Engineering Society (EMBS)
July 20-24, 2020 via the EMBS Virtual Academy

---

## Introduction

<!--- ![Federated Learning](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/federated_learning.png =250*250)![Federated Learning EEG](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/federated_learning_eeg.png =250*250) --->

The impact of deep learning (DL) methodologies within the sphere of Brain-Computer Interfaces (BCI) for the categorization of electroencephalographic (EEG) transcriptions has been stymied by the dearth of expansive datasets. Constraints linked to privacy concerns surrounding EEG signals impede the creation of large EEG-BCI datasets through the amalgamation of various smaller datasets for the shared training of the machine learning model. Consequently, this paper presents a novel privacy-preserving DL framework named federated transfer learning (FTL) for EEG categorization, which is predicated on the federated learning structure. Utilizing the single-trial covariance matrix, this proposed framework extracts common discriminative information from multi-subject EEG data via domain adaptation techniques. The effectiveness of the proposed framework is assessed against the PhysioNet dataset for a two-class motor imagery classification. All while circumventing direct data sharing, our FTL method attains a 2% improvement in classification accuracy in a subject-adaptive analysis. Moreover, when multi-subject data is not available, our framework delivers a 6% increase in accuracy in comparison to other leading-edge DL frameworks.

---

## Network Architecture


Our proposed architecture comprises the following four layers: the manifold reduction layer, the common embedded space, the tangent projection layer, and the federated layer. The function of each layer is detailed below:

1. **Manifold Reduction Layer**: Spatial covariance matrices are consistently presumed to inhabit high-dimensional Symmetric Positive Definite (SPD) manifolds. This layer acts as a linear map from the high-dimensional SPD manifold to the low-dimensional counterpart, with undefined weights reserved for learning.

2. **Common Embedded Space**: The common space is the low-dimensional SPD manifold, whose elements are diminished from each high-dimensional SPD manifold. It is specifically designed for the transfer learning setting.

3. **Tangent Projection Layer**: The function of this layer is to project the matrices on SPD manifolds to its tangent space, which is a local linear approximation of the curved space.

4. **Federated Layer**: Deep neural networks are implemented in this layer. For the transfer learning setting, the parameters of neural networks are updated by federated aggregation.


---

### How To Run The Code

We extend our gratitude to the open-source community, which facilitates the wider dissemination of the work of other researchers as well as our own. The coding style in this repo is relatively rough. We welcome anyone to refactor it to make it more effective. The codebase for our models builds heavily on the following repositories:
[<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann) 
[<img src="https://img.shields.io/badge/GitHub-SPDNet(Z.W.Huang)-b31b1b"></img>](https://github.com/zhiwu-huang/SPDNet)
[<img src="https://img.shields.io/badge/GitHub-mne-b31b1b"></img>](https://github.com/mne-tools/mne-python)

### Data pre-processing
