
# <p align="center">欢迎来到机器学习分赛道！ | Welcome to the Machine Learning Track! 🚀</p>

在黑客马拉松的机器学习分赛道，您将会以[英特尔预测性资产分析 AI 参考套件](https://github.com/oneapi-src/predictive-asset-health-analytics)为蓝本，将模型训练、超参数调优和推理的优化技术用于您自己的挑战项目中。

In this track of the hackathon, you will use Intel's [Predictive Asset Analytics AI Reference Kit](https://github.com/oneapi-src/predictive-asset-health-analytics) as blueprint, adopt the optimization techniques for model training, hyperparameter tuning, and inference for your own challenge project. 

## 第一部分：入门 | Part I: Getting Started

### 安装 | Installation

本 notebook 使用的主要的库包括： | The main libraries used in this notebook include:
- [英特尔® Modin* 分发版  | Intel<sup>&reg;</sup> Distribution of Modin*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html#gs.9hqdj4)
- [英特尔® Extension for Scikit-learn* | Intel<sup>&reg;</sup> Extension for Scikit-learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html#gs.8txte9)
- [针对英特尔® 架构优化的 XGBoost | XGBoost Optimized for Intel<sup>&reg;</sup> Architecture](https://www.intel.com/content/www/us/en/developer/articles/technical/xgboost-optimized-architecture-getting-started.html)
- [英特尔® Daal4py | Intel<sup>&reg;</sup> Daal4py](https://intelpython.github.io/daal4py/)

请使用以下命令，安装运行 notebook 所需的所有 Python 软件包 

To install all of the Python packages required to run the notebook, use

```
pip3 install -r requirements.txt
```

### 数据集 | Dataset

本演示中使用的数据集包括 100,000 个不同的电线杆，以及反映电线杆整体健康状态的 30 多个特征。生成数据的相关说明请参见[此处](https://github.com/oneapi-src/predictive-asset-health-analytics#run-the-code-for-test-dataset-generation-training-the-model-and-prediction)的预测性资产健康状况分析存储库。

The dataset used in this demo consists of 100,000 different utility poles with over 30 features on the overall health of the utility. The instructions to generate the data can be found in the Predictive Asset Health Analytics repository [here](https://github.com/oneapi-src/predictive-asset-health-analytics#run-the-code-for-test-dataset-generation-training-the-model-and-prediction). 

### 视频演示 | Video Demo

您也可以点击[此处](https://www.intel.com/content/www/us/en/developer/videos/optimize-utility-maintenance-prediction-ai-kit.html)，观看本 notebook 的视频演示。

You may also watch a video demo of this notebook [here](https://www.intel.com/content/www/us/en/developer/videos/optimize-utility-maintenance-prediction-ai-kit.html).

## 第二部分：黑客马拉松 | Part II: Hackathon

在黑客马拉松的第二部分，您可以将数据集替换为采用类似结构的数据集，即具有二元目标变量的表格数据集。您也可以按照上述步骤生成一个新的数据集。提供的演示 notebook 可作为黑客马拉松期间的参考。

In the second part of the hackathon, you may substitute the dataset with one that follows a similar structure, a tabular dataset with a binary target variable. Alternatively, you may generate a new dataset following the steps above. The demo notebook provided can be used as a reference during the hackathon. 

祝您建模愉快！

Happy modeling!