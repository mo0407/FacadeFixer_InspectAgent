# FacadeFixer InspectAgent

建筑外立面**纯图像主动巡检**工作流。从一张原始图片开始：建立全图锚点，由视觉大模型选择需要放大的区域，最多执行两次原图局部检查，再回映坐标、合并重复候选并逐项复核。

**三图需求核对：当前已具备通用主干，尚未完整复现全部实验约定。**具体差距包括独立 baseline 批评测、0.25 与 Env 的指定语义、8×5 网格、EfficientViT、从空历史生成初始失败材料、受控阈值动作及逐条规则消融。完整对照、验收条件和三张原图见 [三图需求对照与验收](docs/requirements-three-figures.md)。

本项目的 GitHub 仓库为 [FacadeFixer_InspectAgent](https://github.com/mo0407/FacadeFixer_InspectAgent)，本机开发目录 `E:\ai\低空\FacadeFixer_IMAGE` 与该仓库是同一项目。代码从 `FacadeFixer_VIDEO` 复制后独立改造，原视频目录继续保留；原视频版说明见 [video-baseline-readme.md](docs/video-baseline-readme.md)。

本机另保留未改造源码归档 `baseline-source-20261118.zip`、来源与文件校验清单 `backup-manifest.json` 及验证记录 `validation-results.json`。这些本机溯源文件不上传至仓库，无法从远端下载；源码归档包含复制时的 81 个源文件与清单，不含模型权重、密钥或运行缓存。

## 图像工作流

```mermaid
flowchart LR
    I[原始图片] --> D[专家模型全图检测]
    D --> A[Box / Ghost / Env / Grid 锚点]
    P[冻结的离线规则手册 P0] --> G[VLM 选择检查区域或结束]
    A --> G
    G -->|区域 ID + 扩展倍数| C[从原始图片裁剪]
    C --> L[允许的局部专家模型复跑]
    L -->|局部图 + 检测框| G
    G -->|结束 / 达到最多两次局部检查| M[原图坐标回映 + 同类框去重]
    M --> V[VLM 最终逐项复核]
    V --> R[保留 / 拒绝 / 待定 + 可追溯结果]
    R --> H[失败记录与有证据反馈]
    H --> F[离线复盘与规则提案]
    F --> T[独立参考集成对验证]
    T -->|通过门槛且正向改善| N[发布下一版 P0]
    N --> P
```

同一轮内使用同一份 P0，不在检测途中修改规则。Generator 负责本轮视觉检查；Reflector、Curator 在离线阶段整理有证据的失败与规则候选。候选通过独立参考集的真实成对重放、质量门槛和正向改善检验后，才能发布为下一轮默认加载的 P0；mock 或注入的测试执行器不能晋升正式规则。

已增加 YOLOv9 分割与 Ultralytics 自定义检测/分割的注册适配器；CUBIT 权重与训练数据需自行提供，尚未进行真实 CUBIT 权重验收。内置示例继续使用 `yolo11m` 与 `yolo12m`。完整步骤见 [图像检查与离线规则闭环](docs/complete-image-workflow.md)。

## 快速开始

需要 Python 3.11 或更高版本。克隆仓库后安装并激活环境：

```powershell
git clone https://github.com/mo0407/FacadeFixer_InspectAgent.git
cd FacadeFixer_InspectAgent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\Activate.ps1
```

先运行不联网、不调用真实检测模型的合成演示：

```powershell
facade-image --workspace workspace image-demo --output-dir examples/generated_image_demo
```

演示中的图片、候选框与模型回答均为测试数据，用于检查裁剪、坐标回映、去重与报告链路，不表示真实建筑的检测效果。

也可复用本机已有 Python 环境，避免改动原视频环境的安装包：

```powershell
Set-Location "E:\ai\低空\FacadeFixer_IMAGE"
$env:PYTHONPATH = (Resolve-Path .\src).Path
& "E:\ai\低空\facade_mcp_langchain\.venv\Scripts\python.exe" -m facade_mcp.cli --workspace workspace image-demo --output-dir examples/generated_image_demo
```

## 处理自己的图片

1. 在 [image-inspection.example.json](examples/image-inspection.example.json) 中填写图片路径，以及要使用的专家模型。
2. 配好本地模型环境。当前已有 FacadeFixer 源码、权重与独立模型 Python 环境可通过环境变量指定。
3. 配置支持图片输入的 VLM；远程执行需要配置文件中的 `api_enabled=true`，并在命令中显式加入 `--allow-api`。

```powershell
$env:FACADE_SOURCE_REPO = "E:\ai\低空\项目代码\FacadeFixer-main"
$env:FACADE_SOURCE_PYTHON = "E:\ai\低空\项目代码\FacadeFixer-main\.venv\Scripts\python.exe"
$env:FACADE_SOURCE_DEVICE = "cpu"

facade-image --workspace workspace image-inspect --config examples/image-inspection.example.json --model-config examples/image-models.mock.json --playbook examples/image-playbook.json
```

这里的 `image-models.mock.json` 只替代 VLM 决策，`image-inspect` 仍会调用配置的本地专家模型。要完整离线检查基础链路，请使用 `image-demo`。

真实 VLM 可从 [image-models.remote.example.json](examples/image-models.remote.example.json) 开始配置，格式沿用 [三角色模型网关](docs/model-gateway-v1.md)：连接可选择 `chat_completions` 或 `responses`，图像连接必须设 `supports_images=true`。端点和密钥通过环境变量读取，配置文件不保存密钥。使用远程 VLM 会向该服务商发送本轮图片证据、候选信息和规则手册。

远程示例将三个角色连接到 `gpt-6.1-sol` Responses，使用 `reasoning_effort="medium"` 和 `expected_model` 响应模型校验，默认关闭 API。配置声明不证明供应商视觉能力，可以显式运行 `vision-check` 验证图像读取与结构化动作；本次开发未进行付费远程视觉测试。

## 注册自己的专家

参考 [专家注册示例](examples/experts.registry.example.json)，通过服务器环境变量 `FACADE_EXPERTS_CONFIG` 指向注册 JSON。权重、源码和解释器路径仅允许位于配置文件父目录或 `FACADE_EXPERTS_ALLOWED_ROOTS` 指定的服务器目录内。MCP 调用者选择 profile，不能在工具参数中传入任意权重或可执行文件。

每个专家声明 `class_map`、`class_codes`、全图/局部资格和任务类型。流程按能力筛选并锁定本轮实际权重哈希；局部发现会形成新的可检查锚点。注册分割专家可输出原尺寸 mask 证据。B 鼓包已补充业务映射，支持它的模型可执行通用分割；OpenCV 基线明确拒绝 B。

## 作为 MCP 使用

安装后可启动 `facade-image-mcp`，或直接执行 `python -m facade_mcp.image_server`，以 stdio 方式接入支持 MCP 的客户端。高层工具 `facade_image_inspect` 接收 `request` 对象、`model_config_path`、可选的 `playbook_path` 和默认关闭的 `allow_api`。本地专家检测继续通过已有 model MCP 调用。

可用 `facade-image --workspace workspace mcp-config --include-image --output mcp.local.json` 导出包含图像入口的客户端配置。这个文件包含本机路径，保存在本地使用。

模型配置和规则文件须位于允许的输入目录内。使用该工具不需要网页或 HTTP 服务；旧 Web 工作台仍在副本中保留。

## 结果如何理解

- 全图锚点保留原图坐标与来源；局部检测框通过裁剪偏移回到原图，不从缩略图反推坐标。
- 最终复核区分 `keep`、`reject`、`defer`。`keep` 是模型保留的候选，仍需人工验收。
- 每个候选展示原图定位和原像素局部图；原始证据、规则快照、模型调用与局部检查记录随任务保存，可用于复盘。
- 可以提供与原图等尺寸的二值墙面范围 mask，将门、窗、外楼梯和背景排除。未提供范围 mask 时，不应把检测结果视为已经完成人工范围复核。
- 没有独立人工真值时，不输出准确率、召回率或“已达标”的结论。合成演示只验证程序行为。

参数、输入输出约定及当前限制见 [图像工作流说明](docs/image-workflow.md)。主流程以框级主动核验为核心，注册分割专家可提供 mask 证据；原有视频、交互分割、正式报告模板与 Web 模块保留供独立使用。

## 失败记录与同因统计

运行失败、已确认的质量问题和待复核观测分别归档到本地失败记忆库，观测不计入失败数量，mock 视觉输出不进入观测统计。可以查看哪些原因反复出现、涉及哪些任务，并用有证据的人工反馈补充漏检、误报、框偏差等质量问题。没有真值或可靠复核依据时，不把可疑现象直接算成漏检或误报；Reflector 的原因解释保留为推测。

```powershell
facade-image --workspace workspace failure-summary --output failure-dashboard.html
facade-image --workspace workspace failure-feedback --config examples/image-feedback.example.json
```

反馈示例中的任务 ID 和证据文件须替换为本地真实产物。重复原因采用固定代码聚合，同一事件幂等提交不重复计数；规则不会因此自动更新，也不会触发付费重跑。详见 [失败记忆说明](docs/failure-memory.md)。

## 离线验证与规则发布

```powershell
facade-image --workspace workspace rule-latest
facade-image --workspace workspace rule-propose --model-config examples/image-models.remote.example.json --allow-api
```

提案不等于发布。准备独立人工参考集后，可用 `rule-validate` 对指定候选做真实成对重放，或显式运行 `rule-cycle` 完成一次提案与验证。格式见 [验证数据集示例](examples/rule-validation.example.json)，其中图片和标注仅为占位格式，不能直接用作真值。命令参数和发布门槛见 [完整流程](docs/complete-image-workflow.md)。`image-inspect` 默认加载已验证发布的最新版 P0，显式 `--playbook` 可覆盖；单轮执行继续冻结该快照。

## 测试

```powershell
python -m pytest -q
```
