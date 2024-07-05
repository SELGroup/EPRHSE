# 自用深度学习docker image制作
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG USERNAME

RUN apt update

RUN apt install sudo

RUN apt install -y build-essential cmake python3-dev python3-pip git 

# 常用工具
RUN apt install -y vim wget tmux

# 清华源，取消下行注释即可
# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 根据torch, docker cuda版本修改
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
RUN pip3 install torch_geometric

RUN mkdir /workspace
WORKDIR /workspace

# amp混合精度加速
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /workspace/apex
RUN pip3 install packaging
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
# RUN pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
RUN pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# 好用的泛用工具
RUN pip3 install einops tqdm torchsummary
# - einop：优雅地操作向量纬度，支持torch numpy
# - loguru: 优雅地使用日志
# - tqdm: 任务进度条
# - torchsummary: 统计模型参数、结构
# 其他工具
RUN pip3 install scikit-learn pandas scipy
# huggingface
RUN pip3 install transformers datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
# 绘图相关
RUN pip3 install seaborn scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
# 模型调参相关
RUN pip3 install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
# 性能优化相关
RUN pip3 install joblib numba -i https://pypi.tuna.tsinghua.edu.cn/simple

# 使用python3作为默认版本python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 安全性考量，避免使用root用户
RUN useradd --create-home -s /bin/bash ${USERNAME}
RUN usermod -aG sudo ${USERNAME}
RUN echo "${USERNAME}:ubuntu" | chpasswd
RUN chown -R ${USERNAME}:${USERNAME} /workspace
USER ${USERNAME}

WORKDIR /workspace
