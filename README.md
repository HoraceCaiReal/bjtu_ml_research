# 裂纹图像识别系统 — 研究性专题

> 《机器学习与Python编程》课程研究性专题（测控系）  
> 组员：XXX（仓库管理员）、XXX（协作者）  
> 日期：2026年4月

---

## 一、快速开始（管理员首次搭建）

### 1.1 前提条件

- Windows 10/11
- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)（已安装并启动）
- [Git for Windows](https://git-scm.com/download/win)
- [VSCode](https://code.visualstudio.com/) + 推荐插件：Python、Jupyter、GitLens

### 1.2 克隆与启动

```powershell
# 1. 克隆仓库（管理员直接克隆自己的仓库，协作者见第二节）
git clone https://github.com/<你的用户名>/bjtu_ml_research.git
cd bjtu_ml_research

# 2. 准备数据（数据保密，不上传 Git）
# 将老师提供的数据集放入本地目录，例如：
# C:/Work/bjtu_crack_data/
#   ├── Negative/
#   └── Positive/
# 然后修改 docker/docker-compose.yml 中的数据挂载路径为你自己的实际路径

# 3. 启动 Docker 环境
cd docker
docker compose up --build

# 4. 浏览器访问 Jupyter Lab
# http://localhost:8888
# 无需密码，直接进入
```

> **字体说明**：本工程已统一配置为**微软雅黑**。Jupyter Lab 界面、Matplotlib 图表中文均正常显示。

---

## 二、协作流程（Git 直接提交）

本仓库采用**直接向主仓库提交**的方式协作。管理员和协作者都拥有写权限，双方直接 `git push` 到 `main` 分支。由于只有两人，关键是**提交前沟通好、避免同时改同一文件**。

### 2.1 管理员操作

#### 步骤 1：创建 GitHub 仓库并邀请协作者

1. 在 GitHub 新建仓库 `bjtu_ml_research`，设为 **Public** 或 **Private**。
2. `Settings -> Access -> Collaborators -> Add people`，邀请协作者的 GitHub 账号。
3. 本地初始化并推送：

```powershell
git init
git add .
git commit -m "init: 工程骨架 + Docker环境"
git branch -M main
git remote add origin https://github.com/<你的用户名>/bjtu_ml_research.git
git push -u origin main
```

#### 步骤 2：日常开发流程

```powershell
# 1. 提交前先拉取最新代码（避免覆盖协作者的改动）
git pull origin main

# 2. 写代码，在 Docker 里跑通
# ...

# 3. 提交（提交信息写清楚）
git add .
git commit -m "feat: 添加 SVM 基线实验"

# 4. 推送到 GitHub
git push origin main
```

> **关键规则**：
> - **每次 push 前先 `git pull`**，如果有冲突，先解决冲突再 push
> - **不要上传数据文件或模型权重**（`.gitignore` 已帮你拦截）
> - **Jupyter Notebook 建议清除输出后再提交**（减少仓库体积和冲突）

### 2.2 协作者操作

#### 步骤 1：直接克隆主仓库

协作者**不需要 Fork**，直接克隆管理员的主仓库：

```powershell
git clone https://github.com/<管理员用户名>/bjtu_ml_research.git
cd bjtu_ml_research
```

> 因为管理员已在 GitHub 上把你添加为 Collaborator，你有直接 push 权限。

#### 步骤 2：日常开发流程

```powershell
# 1. 每次开工前，先拉取最新代码
git pull origin main

# 2. 写代码，Docker 中验证
# ...

# 3. 提交并推送
git add .
git commit -m "feat: 实现数据预处理模块"
git push origin main
```

### 2.3 避免冲突的黄金法则

| 场景 | 做法 |
|------|------|
| 开工前 | `git pull origin main`，确保本地是最新的 |
| 两人同时改同一个文件 | **避免**。开工前微信说一声"我在改 xxx.py" |
| push 被拒（remote has changes）| 先 `git pull origin main`，解决冲突后再 push |
| 不小心覆盖了对方代码 | `git log` 查看历史，`git revert` 回退 |

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
│   └── settings.json               # 团队共享 VSCode 配置（字体、格式化等）
├── data/                           # 数据集（被 .gitignore 忽略，不上传）
│   ├── Negative/                   # 20000 张无裂纹图像（本地存放）
│   └── Positive/                   # 20000 张有裂纹图像（本地存放）
├── docker/
│   ├── Dockerfile                  # 环境定义（PyTorch CPU + 所有依赖）
│   └── docker-compose.yml          # 一键启动 Jupyter Lab
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
│   ├── training/
│   │   ├── losses.py               # 损失函数
│   │   └── optimizers.py           # 参数搜索/调优
│   └── evaluation/
│       └── metrics.py              # 评价指标
├── reports/                        # 导出的 PDF/HTML 报告（保留目录，不上传大文件）
├── videos/                         # 操作演示视频（保留目录，不上传大文件）
├── outputs/                        # 实验输出（图片、日志等）
├── .gitignore                      # 数据与临时文件排除规则
├── pyproject.toml                  # Ruff 格式化规则配置
├── requirements.txt                # Python 依赖清单
└── README.md                       # 本文件
```

---

## 四、环境说明

### 4.1 Docker 环境特点

- **基础镜像**：`python:3.10-slim`
- **PyTorch**：CPU 版本（`torch==2.2.0+cpu`），无需 GPU
- **中文字体**：已安装 `fonts-noto-cjk`、`fonts-wqy-microhei`，并挂载 Windows 微软雅黑字体
- **Jupyter Lab**：端口 `8888`，无密码，开箱即用
- **TensorBoard**：端口 `6006`（备用）

### 4.2 修改数据路径

`docker/docker-compose.yml` 第 12 行：

```yaml
- C:/Work/bjtu_crack_data:/workspace/data:ro
```

请将 `C:/Work/bjtu_crack_data` 替换为你电脑上**实际的数据集存放路径**（包含 `Negative/` 和 `Positive/` 的目录）。修改后重启 Docker：

```powershell
cd docker
docker compose down
docker compose up --build
```

### 4.3 VSCode 配置与 AI 插件推荐

本仓库已共享 `.vscode/settings.json`，包含：

- 自动格式化（Ruff）
- Jupyter 自动保存
- 文件排除规则（`__pycache__`、`.ipynb_checkpoints` 等）

**字体说明**：VSCode 编辑器字体保持系统默认，不做强制。只有在 Jupyter Notebook 中画图时，需要在 Notebook 开头调用 `from src.plot_config import set_chinese_font; set_chinese_font()` 即可让 matplotlib 使用微软雅黑。

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
# 在 Docker Jupyter Lab 中，打开 Notebook
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
git pull origin main
```

**Q2：两人同时 push 冲突了怎么办？**
```powershell
# 1. 先拉取对方代码
git pull origin main

# 2. Git 会提示冲突文件，打开文件手动保留正确版本
# 3. 解决后提交
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
- 可以用，但务必理解代码逻辑，在 Docker 中验证通过后再提交。提交信息中注明 "部分代码由 AI 辅助生成"。

**Q6：没有 GPU，CNN 训练会不会很慢？**
- 本任务使用 CPU 版 PyTorch，但图像尺寸建议先缩小（如 128x128 或 64x64），数据量也可以先用子集（如 2000 张）快速验证代码正确性，最后再跑全量。

---

## 七、检查清单（管理员 & 协作者首次协作前对照）

- [ ] GitHub 仓库已创建，协作者已添加为 Collaborator
- [ ] 本地 `git clone` 成功，能正常拉取和推送
- [ ] Docker Desktop 已启动，`docker compose up` 能打开 Jupyter Lab
- [ ] `http://localhost:8888` 能访问
- [ ] 在 Notebook 中运行 `import torch; print(torch.__version__)` 显示 CPU 版本
- [ ] 能成功 `import sklearn`, `import cv2`, `import ipywidgets`
- [ ] 管理员成功推送一次代码到 main
- [ ] 协作者成功拉取并推送一次代码到 main
- [ ] 双方 VSCode 插件基本一致，AI 辅助编程已配置

---

> 祝合作顺利！有问题在 `docs/讨论记录/` 中记录，或通过微信/QQ 及时沟通。
