


<div align="center">   

# XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge
</div>

<div align="center">   

[![Static Badge](https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green)](http://arxiv.org/abs/2506.22726) 
[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) 
[![Project Page](https://img.shields.io/badge/Project%20Page-XTransfer-yellow)]()

</div>

This is the official repository of the **XTransfer**, a pioneering and scalable method that enables modality-agnostic few-shot model transfer for advancing human sensing on edge systems. 

**XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge**
<br/>
[Yu Zhang<sup>1</sup>](https://yuzhang.dev/), [Xi Zhang<sup>1</sup>](), [Hualin Zhou<sup>1</sup>](), [Xinyuan Chen<sup>1</sup>](), [Shang Gao<sup>1</sup>](), [Hong Jia<sup>3</sup>](https://h-jia.github.io/), [Jianfei Yang<sup>2</sup>](https://marsyang.site/), [Yuankai Qi<sup>1</sup>](https://v3alab.github.io/author/yuankai-qi/), [Tao Gu<sup>1</sup>](https://taogu.site/)
<br/>
<br/>
<sup>1</sup>Macquarie University, <sup>2</sup>Nanyang Technological University, <sup>3</sup>The University of Melbourne
<br/>


## 📝 Abstract 

Deep learning for human sensing on edge systems presents significant potential for smart applications. However, its training and development are hindered by the limited availability of sensor data and resource constraints of edge systems. While transferring pre-trained models to different sensing applications is promising, existing methods often require extensive sensor data and computational resources, resulting in high costs and limited transferability. In this paper, we propose XTransfer, a first-of-its-kind method enabling modality-agnostic, few-shot model transfer with resource-efficient design. XTransfer flexibly uses pre-trained models and transfers knowledge across different modalities by (i) model repairing that safely mitigates modality shift by adapting pre-trained layers with only few sensor data, and (ii) layer recombining that efficiently searches and recombines layers of interest from source models in a layer-wise manner to restructure models. We benchmark various baselines across diverse human sensing datasets spanning different modalities. The results show that XTransfer achieves state-of-the-art performance while significantly reducing the costs of sensor data collection, model training, and edge deployment.


## 📦 Method
| ![pipeline.jpg](assets/system_v2.png) | 
|:--:| 
| <div align="left">***Figure 1. Overview**. XTransfer transfers source models across modalities with few sensor data through model repairing (SRR pipeline) and layer recombining (LWS control).  LWS control first segments source models into layers and operates layer-wise search across pools. At each pool, the pre-search check decides which layers need repairing, then SRR pipeline repairs them and LWS control selects layers of interest. These layers are incrementally recombined during the search, restructuring models for enabling human sensing at the edge. Subfigures (a)–(c) illustrate the feature space evolution before and after repairing.*</div> |


## 🔗 Citation
If you find our work helpful to your research, please consider citing:


```shell
@inproceedings{Yu_2026,
      title={XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge}, 
      author={Yu Zhang and Xi Zhang and Hualin Zhou and Xinyuan Chen and Shang Gao and Hong Jia and Jianfei Yang and Yuankai Qi and Tao Gu},
      booktitle={International Conference on Machine Learning},
      year={2026},
      series={ICML '26},
}
```



