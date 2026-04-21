FROM python:3.10-slim

# 安装系统依赖（含中文字体支持）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-noto-cjk \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# 微软雅黑字体（从 Windows 复制或下载，这里用文泉驿微米黑替代，效果接近）
# 如果你有 Windows 字体文件，可在 docker-compose 中挂载
RUN ln -sf /usr/share/fonts/truetype/wqy/wqy-microhei.ttc /usr/share/fonts/truetype/microsoft-yahei.ttc || true

# 设置工作目录
WORKDIR /workspace

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配置 Jupyter：微软雅黑 + 自动保存
RUN mkdir -p /root/.jupyter/custom
RUN echo "/* 微软雅黑字体配置 */" > /root/.jupyter/custom/custom.css && \
    echo ".CodeMirror pre, .output_area pre, .rendered_html { font-family: 'Microsoft YaHei', 'WenQuanYi Micro Hei', sans-serif !important; }" >> /root/.jupyter/custom/custom.css

# 暴露端口
EXPOSE 8888 6006

# 启动 Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]