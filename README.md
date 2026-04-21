# 裂纹图像识别系统 — 研究性专题

> 《机器学习与Python编程》课程研究性专题（测控系）  
> 组员：XXX（仓库管理员）、XXX（协作者）  
> 日期：2026年4月

---

## 一、快速开始（首次搭建环境）

### 1.1 前提条件

- Windows 10/11
- [Anaconda](https://www.anaconda.com/download)（已安装）
- [Git for Windows](https://git-scm.com/download/win)
- [VSCode](https://code.visualstudio.com/) + 推荐插件：Python、Jupyter、Ruff、GitLens
- **（推荐）NVIDIA GPU + 已安装 CUDA 驱动**：环境默认安装 PyTorch CUDA 11.8 版本，可自动加速训练；无 GPU 也能正常运行，PyTorch 会自动回退到 CPU

### 1.2 克隆仓库并创建环境

```powershell
# 1. 克隆仓库
git clone https://github.com/lijiawei255/bjtu_ml_research.git
cd bjtu_ml_research

# 2. 创建并激活 conda 环境（环境名：bjtu_ml，Python 3.10）
conda env create -f environment.yml
conda activate bjtu_ml

# ★ 如果你的电脑没有 NVIDIA GPU，需要改用 CPU 版本 PyTorch：
#   pip uninstall torch torchvision -y
#   pip install torch==2.1.1+cpu torchvision==0.16.1+cpu -f https://mirrors.aliyun.com/pytorch-wheels/cpu

# 3. 注册 Jupyter Kernel（让 VSCode/Jupyter Lab 能找到这个环境）
python -m ipykernel install --user --name=bjtu_ml --display-name "Python (bjtu_ml)"

# 4. 配置 nbstripout（自动清除 Notebook 输出，避免提交大文件）
nbstripout --install

# 5. 配置数据集路径
# 复制模板文件，然后修改为你电脑上的实际路径
copy .env.example .env
# 用 VSCode 打开 .env，把 CRACK_DATA_ROOT 改成你的数据集根目录

# 6. 启动 Jupyter Lab
jupyter lab
```

浏览器会自动打开，或者在地址栏输入 `http://localhost:8888`。

> **字体说明**：VSCode 编辑器保持系统默认字体。Jupyter Notebook 中画图时，运行：
> ```python
> from src.plot_config import set_chinese_font
> set_chinese_font()
> ```
>
> **数据集路径**：代码通过 `.env` 文件读取 `CRACK_DATA_ROOT`，不上传 Git。双方各自配置自己的路径。

### 1.3 VSCode 选择解释器

打开 VSCode 后，按 `Ctrl+Shift+P` → 输入 `Python: Select Interpreter` → 选择 `bjtu_ml (conda)`。

这样 VSCode 的代码补全、调试、Jupyter 内核都会使用统一的 conda 环境。

---

## 二、协作流程（Git 分支 + 合并）

本仓库采用**功能分支开发、main 分支合并**的方式协作。双方各自从 `main` 创建分支开发，完成后合并回 `main`。

> 为什么用分支？
> - `main` 始终保持可运行状态，不会有人正在改一半的代码
> - 每个人的工作隔离在自己的分支，互不干扰
> - 合并前可以 review 对方代码，避免低级错误进主分支

### 2.1 完整协作流程（双方通用）

```powershell
# ========== 1. 开工前：同步 main ==========
git checkout main
git pull origin main

# ========== 2. 创建功能分支 ==========
# 分支名规范：feature/功能描述 或 fix/bug描述
git checkout -b feature/数据预处理

# ========== 3. 写代码、测试 ==========
# ... 开发中 ...

# ========== 4. 提交到本地 ==========
git add .
git commit -m "feat: 实现数据加载与归一化"

# ========== 5. 推送分支到远程 ==========
git push origin feature/数据预处理

# ========== 6. 合并到 main（两种方式任选） ==========

# 【方式A：本地合并（推荐，最快）】
git checkout main
git pull origin main              # 确保 main 是最新的
git merge feature/数据预处理      # 合并分支
# 如果有冲突，解决后 git add . && git commit -m "merge: 合并数据预处理"
git push origin main              # 推送合并后的 main

# 【方式B：GitHub 网页合并（想偷懒时用）】
# 打开 https://github.com/lijiawei255/bjtu_ml_research
# 点击上方 "branches" 或页面提示的 "Compare & pull request"
# 点击 "Merge" 按钮合并，然后本地执行：
# git checkout main && git pull origin main

# ========== 7. 清理已合并的分支 ==========
git branch -d feature/数据预处理       # 删除本地分支
git push origin --delete feature/数据预处理  # 删除远程分支（可选）
```

### 2.2 关键规则

| 规则 | 说明 |
|------|------|
| **main 只合并不直接改** | 不要在 `main` 分支上直接写代码，永远从 `main` 切出新分支 |
| **分支名见名知意** | `feature/数据增强`、`fix/路径错误`、`docs/更新报告` |
| **合并前 pull main** | 合并前先 `git pull origin main`，避免覆盖他人已合并的代码 |
| **小步快跑** | 一个功能一个分支，不要攒一个星期再合并，尽量 1-2 天合并一次 |
| **合并完删分支** | 合并后删除功能分支，保持分支列表整洁 |

### 2.3 冲突处理

如果合并时提示冲突：

```powershell
git checkout main
git pull origin main
git merge feature/你的分支

# Git 会提示冲突文件，例如：
# CONFLICT (content): Merge conflict in src/data_utils.py

# 打开冲突文件，搜索 "<<<<<<< HEAD"，手动保留正确版本，删除标记行
# 解决后：
git add .
git commit -m "merge: 解决 data_utils.py 冲突"
git push origin main
```

**减少冲突的技巧**：
- 开工前 `git pull origin main` 同步最新代码
- 不要同时修改同一个文件（微信说一声"我在改 xxx.py"）
- 通用函数抽到 `src/` 独立 `.py` 中，减少 Notebook 直接冲突

### 2.4 提交信息规范（建议遵守）

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat:` | 新功能 | `feat: 添加 CNN 模型定义` |
| `fix:` | Bug 修复 | `fix: 修复数据读取路径错误` |
| `docs:` | 文档更新 | `docs: 更新分工说明` |
| `refactor:` | 代码重构 | `refactor: 提取公共函数到 utils` |
| `env:` | 环境配置 | `env: 添加 opencv 依赖` |

---

## 三、工程目录结构

```
bjtu_ml_research/
├── .vscode/
│   └── settings.json               # 团队共享 VSCode 配置
├── data/                           # 数据集（被 .gitignore 忽略，不上传）
│   ├── Negative/                   # 20000 张无裂纹图像（本地存放）
│   └── Positive/                   # 20000 张有裂纹图像（本地存放）
├── docs/                           # 文档、讨论记录、分工说明
│   ├── 分工说明.md
│   ├── 目录结构说明.md
│   └── 讨论记录/                   # 每次讨论新建一个 md 文件
├── notebooks/                      # Jupyter Notebook（研究报告主体）
│   ├── 01_数据探索与预处理.ipynb
│   ├── 02_传统机器学习.ipynb
│   ├── 03_深度学习.ipynb
│   └── 04_综合展示系统.ipynb       # 最终演示 + 报告整合
├── src/                            # 纯 Python 模块（供 notebooks 调用）
│   ├── data_utils.py               # 数据读取、划分、预处理
│   ├── models/
│   │   ├── traditional.py          # 决策树、SVM
│   │   ├── unsupervised.py         # K-Means
│   │   └── cnn.py                  # 卷积神经网络
│   ├── plot_config.py              # Matplotlib 中文字体配置
│   ├── training/
│   │   ├── losses.py               # 损失函数
│   │   └── optimizers.py           # 参数搜索/调优
│   └── evaluation/
│       └── metrics.py              # 评价指标
├── reports/                        # 导出的 PDF/HTML 报告（保留目录，不上传大文件）
├── videos/                         # 操作演示视频（保留目录，不上传大文件）
├── outputs/                        # 实验输出（图片、日志等）
├── .gitignore                      # 数据与临时文件排除规则
├── environment.yml                 # Anaconda 环境定义（团队共享）
├── pyproject.toml                  # Ruff 格式化规则配置
├── requirements.txt                # pip 依赖清单（备用）
└── README.md                       # 本文件
```

---

## 四、环境说明

### 4.1 数据集路径配置（.env）

项目使用 `.env` 文件管理数据集路径，**不上传 Git**（已加入 `.gitignore`）。

```powershell
# 1. 复制模板
copy .env.example .env

# 2. 编辑 .env，填入你的实际路径
# CRACK_DATA_ROOT=D:/课程数据/裂纹图像
```

代码中通过 `src.config.DATA_ROOT` 读取：

```python
from src.config import DATA_ROOT, POSITIVE_DIR, NEGATIVE_DIR
print(DATA_ROOT)  # 显示你配置的路径
```

**为什么这样做？**
- 双方电脑路径不同，不能写死在代码里
- `.env` 不上传 Git，各自独立配置
- 协作者拿到代码后只需要改一行 `.env` 就能跑

### 4.2 Jupyter Notebook 输出自动清理（nbstripout）

Notebook 的 Cell 输出（图片、打印日志）提交到 Git 会导致：
- 仓库体积暴涨
- `git diff` 无法阅读
- 两人同时改一个 notebook 时极易冲突

**解决方案**：`nbstripout` 会在每次提交时自动清除所有 Notebook 的输出。

```powershell
# 初始化时运行一次（已包含在 environment.yml 中）
nbstripout --install
```

此后你正常 `git add *.ipynb` → `git commit` → `git push`，nbstripout 会在后台自动清理输出。

> 最终提交报告前，如果需要保留输出展示给老师，可以临时禁用：
> ```powershell
> git add --no-verify notebooks/04_综合展示系统.ipynb
> ```

### 4.3 讨论记录（GitHub Issues）

团队讨论统一使用 **GitHub Issues**，不再用本地 markdown 文件记录。

**地址**：[https://github.com/lijiawei255/bjtu_ml_research/issues](https://github.com/lijiawei255/bjtu_ml_research/issues)

**使用规范**：
- 创建 Issue 时选择对应标签：`讨论`、`数据`、`模型`、`环境`、`报告`
- 标题格式：`[标签] 简短描述`，如 `[模型] CNN 输入尺寸选 128 还是 256`
- 讨论结论明确后关闭 Issue
- 重要的代码修改直接 push，提交信息中引用 Issue 编号，如 `fix #3`

### 4.4 为什么用 `environment.yml`？

`environment.yml` 是 Anaconda 的原生环境定义文件，比 `requirements.txt` 更适合团队：

- **明确指定 Python 版本**（3.10），避免双方 Python 版本不一致
- **区分 conda 包和 pip 包**：NumPy、Pandas 等用 conda 安装更稳定；PyTorch CUDA 版本通过 pip + `--extra-index-url` 安装，确保获取正确的 GPU 轮子
- **PyTorch CUDA 版**：通过 pip 指定 `+cu118` 后缀和阿里云镜像，确保安装 CUDA 11.8 版本；代码中自动检测 GPU 可用性，无 GPU 时回退到 CPU
- **跨平台一致**：conda 会自动处理 Windows 上的二进制依赖

### 4.2 环境更新

如果后续需要新增依赖：

```powershell
conda activate bjtu_ml

# conda 包：直接安装
conda install 新包名

# pip 包：pip install
pip install 新包名
```

**然后手动编辑 `environment.yml`**，将新增包添加到对应的 conda 或 pip 段。

> ⚠️ **不要使用 `conda env export > environment.yml`**！这会覆盖精心配置的文件，导致 PyTorch CUDA 的 `--extra-index-url` 和版本后缀 `+cu118` 丢失，协作者拉取后将无法正确安装 GPU 版本。

```powershell
# 编辑 environment.yml 后，提交到 Git
git add environment.yml
git commit -m "env: 添加 xxx 依赖"
git push origin main
```

协作者收到后执行：
```powershell
conda env update -f environment.yml --prune
```

### 4.5 如果 conda 安装太慢

可以配置清华 Anaconda 镜像：

```powershell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
```

然后重新 `conda env create -f environment.yml`。

### 4.6 VSCode 配置与 AI 插件推荐

本仓库已共享 `.vscode/settings.json`，包含：

- 自动格式化（Ruff）
- Jupyter 自动保存
- 文件排除规则（`__pycache__`、`.ipynb_checkpoints`、`.venv` 等）

**推荐安装的 VSCode 插件**（双方保持一致）：

| 插件名 | 用途 |
|--------|------|
| Python (Microsoft) | Python 语言支持 |
| Jupyter (Microsoft) | Notebook 编辑与运行 |
| Ruff (Astral Software) | 代码自动格式化 + Linter |
| GitLens | Git 历史查看 |
| GitHub Copilot / 通义灵码 / CodeGeeX | AI 辅助编程（自选其一） |

---

## 五、向老师提交物的准备

### 5.1 提交内容清单

| 提交物 | 存放位置 | 说明 |
|--------|----------|------|
| 研究报告 | `notebooks/04_综合展示系统.ipynb` | Jupyter Notebook 格式，含背景、分工、讨论记录、代码、实验结果 |
| 操作视频 | `videos/` | 录制综合展示系统的操作过程（不上传 Git，单独提交） |
| 源代码 | `src/` + `notebooks/` | 已通过 GitHub 仓库提交，老师可查看完整历史 |
| 数据集 | 不提交 | 数据保密，仅本地存放 |

### 5.2 导出 Notebook 为 PDF/HTML（备用）

若老师要求 PDF：

```bash
# 在 Jupyter Lab 中，打开 Notebook
# File -> Save and Export Notebook As -> PDF
# 或
jupyter nbconvert --to pdf notebooks/04_综合展示系统.ipynb
```

### 5.3 视频录制建议

- 工具：Windows 自带 `Xbox Game Bar`（Win+G）或 `OBS Studio`
- 内容：
  1. 打开 Jupyter Lab，启动 `04_综合展示系统.ipynb`
  2. 演示数据处理方式的选择与切换
  3. 演示模型选择（决策树 / SVM / CNN）
  4. 演示超参数调整与实时结果查看
  5. 演示测试集验证与指标输出
- 时长：5-10 分钟

---

## 六、常见问题

**Q1：协作者如何同步管理员最新更新？**
```powershell
git checkout main
git pull origin main
```

**Q2：合并分支时冲突了怎么办？**
```powershell
# 1. 确保你在 main 分支，且已拉取最新代码
git checkout main
git pull origin main

# 2. 合并你的功能分支
git merge feature/你的分支

# 3. Git 会提示冲突文件，打开文件手动保留正确版本，删除 <<<<<<< 标记
# 4. 解决后提交
git add .
git commit -m "merge: 解决冲突"
git push origin main
```

**Q3：Jupyter Notebook 里的输出要不要提交到 Git？**
- 日常开发：建议 **清除输出** 后提交（减少 diff 噪音和仓库体积）。
- 最终提交前：可以保留关键输出作为报告展示。

**Q4：模型权重文件太大，能传 Git 吗？**
- **不能**。`.gitignore` 已忽略 `*.pth`、`*.pkl` 等。如需临时共享，用微信/网盘；正式代码中提供训练脚本即可。

**Q5：AI 插件生成的代码可以直接用吗？**
- 可以用，但务必理解代码逻辑，在 conda 环境中验证通过后再提交。提交信息中注明 "部分代码由 AI 辅助生成"。

**Q6：没有 GPU，CNN 训练会不会很慢？**
- 本项目默认安装 PyTorch CUDA 11.8 版本。如果你有 NVIDIA GPU，训练会自动使用 GPU 加速；如果没有 GPU，PyTorch 会自动回退到 CPU 运行。
- 无 GPU 时建议先缩小图像尺寸（如 128x128 或 64x64），数据量也可以先用子集（如 2000 张）快速验证代码正确性，最后再跑全量。
- 也可以切换为 CPU 版 PyTorch（更轻量）：`pip install torch==2.1.1+cpu torchvision==0.16.1+cpu -f https://mirrors.aliyun.com/pytorch-wheels/cpu`

**Q7：协作者的 conda 环境和我不一样怎么办？**
- 如果 `environment.yml` 更新过，协作者运行 `conda env update -f environment.yml --prune` 即可同步。
- 如果还有差异，双方同时 `conda list` 对比，把差异包补充到 `environment.yml` 里统一提交。

---

## 七、检查清单（管理员 & 协作者首次协作前对照）

- [ ] GitHub 仓库已创建，协作者已添加为 Collaborator
- [ ] 本地 `git clone` 成功，能正常拉取和推送
- [ ] `conda env create -f environment.yml` 成功，无报错
- [ ] `conda activate bjtu_ml` 后，能 `import torch, sklearn, cv2, ipywidgets`
- [ ] 已复制 `.env.example` 为 `.env` 并配置好数据集路径
- [ ] 运行 `src/config.py` 或 `src/data_utils.py` 能正确找到数据集
- [ ] `nbstripout --install` 已执行
- [ ] VSCode 解释器已选择为 `bjtu_ml (conda)`
- [ ] Jupyter Lab 能正常启动，`http://localhost:8888` 可访问
- [ ] `from src.plot_config import set_chinese_font; set_chinese_font()` 后 matplotlib 中文正常
- [ ] 管理员成功创建一个 feature 分支、开发、合并到 main
- [ ] 协作者成功创建一个 feature 分支、开发、合并到 main
- [ ] 双方各创建一个 GitHub Issue 测试讨论流程
- [ ] 双方 VSCode 插件基本一致，AI 辅助编程已配置

---

> 祝合作顺利！有问题在 `docs/讨论记录/` 中记录，或通过微信/QQ 及时沟通。
