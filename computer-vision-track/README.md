
# 欢迎来到计算机视觉分赛道！ | Welcome to the Computer Vision Track! 🚀

在本 notebook 中，我们将介绍 <b> [视觉质量检测 AI 参考套件](https://github.com/oneapi-src/visual-quality-inspection) </b>，并展示如何将英特尔优化库用于机器学习，助您赢得黑客马拉松！参考套件 GitHub 页面提供了各种数据集的链接，核心理念是在制造过程中对有缺陷产品进行视觉检查。未来，您可以将这项技术用于牙刷、瓷砖、木材等的检查，但是本 notebook 将重点讨论药片的质量。在该数据集中，面向消费者的非处方药被分类为品质良好和有缺陷两种类别。有缺陷的药片是指包含缺口、裂缝或形状不规则的药片。

In this notebook, we'll introduce the <b>[Visual Quality Inspection AI Reference Kit](https://github.com/oneapi-src/visual-quality-inspection)</b> and show you how to use Intel-optimized libraries for machine learning so you can win the Hackathon! The reference kit GitHub page provides a link to various datasets that all revolve around the concept of visually inspecting damaged products in the manufacturing process. In the future, you could play around with toothbrush, tiles, wood, etc. but in this notebook we will focus on pill quality. In this dataset, consumer over the counter medicial supplements are classfified into good or bad categories. Bad pills are ones that contain chips, cracks, or mishapen features. 

## 第一部分： 入门 | Part I: Getting Started

如需为此项目构建一个 conda 环境，请参考 conda 环境文件 cv_hack.yaml

To build a conda env for this project, please reference the conda env file cv_hack.yaml
```
conda env create --file cv_hack.yaml
conda activate cv_hack
python -m ipykernel install --name cv_hack --user
```

### 数据集 | Dataset

本示例中的数据集请参见：https://www.mvtec.com/company/research/datasets/mvtec-ad (仅需下载 Pill (262 MB) 即可)  
下载完数据集后，请参考视觉质量检测 AI 参考套件 [准备数据](https://github.com/oneapi-src/visual-quality-inspection#2-data-preparation) (执行 Data Cloning可提高模型精度)

The dataset in this example can be found at https://www.mvtec.com/company/research/datasets/mvtec-ad, you only need to download the file Pill (262 MB in size).
Once complete the download, please refer to the Visual Quality Inspection Reference Kit [data preparation](https://github.com/oneapi-src/visual-quality-inspection#2-data-preparation) section for details. (Data Cloning can be executed to improve model accuracy) 


## 第二部分 黑客马拉松 | Part II: Hackathon

黑客马拉松期间使用的数据集请参见：https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes 。 您可以在基于药片这个项目教程的 notebook 基础之上进行适当调整，以使用此数据集。

The dataset used during the hackathon can be found at https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes . You can adapt the pill tutorial notebook to this dataset. 
