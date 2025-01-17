# Jina-CLIP-v2

## 1. 模型介绍

[Jina-CLIP-v2](https://huggingface.co/jinaai/jina-clip-v2) 是 Jina-AI 团队推出的用于文本和图像的通用多语言多模式嵌入模型。
多模态嵌入能够通过连贯的表示方式搜索和理解不同模态的数据。它们是神经信息检索和多模态 GenAI 应用的支柱。

在[Jina-CLIP-v1](https://huggingface.co/jinaai/jina-clip-v1) 和[jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) 的基础之上，Jina-CLIP-v2进行了多项重大改进：


1. 性能提升：v2 在文本-图像和文本-文本检索任务中比 v1 性能提升了 3%。与 v1 类似，v2 的文本编码器可以用作有效的多语言长上下文密集检索器。它的性能与我们的前沿模型相当jina-embeddings-v3（目前是 MTEB 上 1B 参数下最好的多语言嵌入）。

2. 多语言支持：jina-embeddings-v3使用与文本塔相同的主干，支持 89 种语言进行多语言图像检索，与多语言图像检索任务jina-clip-v2相比，提高了 4% 。   

3. 更高的图像分辨率：v2 现在支持 512x512 输入图像分辨率，与 v1 的 224x224 相比有显著提升。更高的分辨率可以更好地处理细节图像、改进特征提取并更准确地识别细粒度视觉元素。
Matryoshka 表示：v2 允许用户将文本和图像嵌入的输出维度从 1024 截断到 64，从而减少存储和处理开销，同时保持强大的性能。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| jinaai/jina_clip_v2 |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("jinaai/jina_clip_v2")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求是3.0.0b2或develop版本**
```bash
# 提供三种 PaddlePaddle 安装命令示例，也可参考PaddleMIX主页的安装教程进行安装

# 3.0.0b2版本安装示例 (CUDA 11.8)
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Develop 版本安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# sh 脚本快速安装
sh build_paddle_env.sh
```

2）[安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **paddlenlp >= 3.0.0b3**

```bash
# 提供两种 PaddleMIX 依赖安装命令示例

# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖、paddlenlp
python -m pip install -e . --user
python -m pip install -e ppdiffusers --user
python -m pip install -r requirements.txt --user
python -m pip install paddlenlp==3.0.0b3 --user

# sh 脚本快速安装
sh build_env.sh
```

## 3 快速开始

### 推理
```bash
# 图片理解
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/jina_clip_v2/run_inference.py \
```

## 参考文献
```BibTeX
@misc{koukounas2024jinaclipv2multilingualmultimodalembeddings,
      title={jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images}, 
      author={Andreas Koukounas and Georgios Mastrapas and Bo Wang and Mohammad Kalim Akram and Sedigheh Eslami and Michael Günther and Isabelle Mohr and Saba Sturua and Scott Martens and Nan Wang and Han Xiao},
      year={2024},
      eprint={2412.08802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.08802}, 
}
```