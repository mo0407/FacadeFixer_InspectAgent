# 部署方法

当前按 Windows 单机部署设计。网页监听本机 127.0.0.1，允许 localhost 主机访问，任务、上传素材和版本记录持久化到 workspace。视频 CLI、MCP 和调优看板可分别使用。

## 1. 下载和安装

需要 Git、Python 3.11 或以上版本。使用 PowerShell 执行：

```powershell
git clone https://github.com/mo0407/FacadeFixer_InspectAgent.git
Set-Location FacadeFixer_InspectAgent
.\deploy\setup.ps1
```

脚本创建项目自己的 `.venv`，安装 Web、PDF 和测试依赖，首次复制 `.env.example` 到 `.env`。已有 `.env` 会保留。如果脚本被执行策略拦截，可以直接使用等价命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,web,pdf]"
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
```

## 2. 本地检测和 Mask 模型

项目服务环境与模型运行环境分别配置。准备已有的 FacadeFixer-main 源码及权重，按该源码项目的依赖要求安装模型环境。最低 YOLO12m + MobileSAM 组合需要 torch、torchvision、ultralytics，以及 OpenCV/Pillow 等该模型环境依赖。是否支持 GPU 以模型解释器探测结果为准。

必须具备下列源码结构：

```text
FacadeFixer-main/
  backend/
  model_runner/
  model_weights/
    detection/yolo12m_building.pt
    mobile_sam.pt
```

其他模型的路径见 [本地模型说明](LOCAL_MODEL_MCP.md)。在本仓库 `.env` 中填写实际绝对路径，例如：

```dotenv
FACADE_SOURCE_REPO=D:/models/FacadeFixer-main
FACADE_SOURCE_PYTHON=D:/models/FacadeFixer-main/.venv/Scripts/python.exe
FACADE_SOURCE_DEVICE=cpu
FACADE_SOURCE_TIMEOUT_SECONDS=600
FACADE_WORKSPACE=./workspace
FACADE_WEB_WORKSPACE=./workspace/web
FACADE_ALLOWED_INPUT_ROOTS=D:/videos;D:/annotations
```

检查模型配置：

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.cli models
```

检查输出包含权重、依赖及运行时状态。可用性探测后，再用实际样本验证模型加载。权重需要自己放入对应目录。

若使用 OpenCV Mask，还需要单独准备 [FacadeFixer_mask_api](https://github.com/mo0407/FacadeFixer_mask_api)，在 `.env` 配置 `FACADE_MASK_REPO` 的绝对路径。Report MCP 的内置草稿渲染可直接使用；源报告格式与限制详见报告及 Web 文档。

## 3. 启动网页

```powershell
.\deploy\start-web.ps1
# 修改端口
.\deploy\start-web.ps1 -Port 8781
```

打开 [本地页面](http://127.0.0.1:8780)。窗口需持续运行，Ctrl+C 结束。等价启动命令：

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.web_app --port 8780
```

在网页新建任务、上传图片或视频、填写建筑信息并生成报告。报告支持编辑与版本记录。当前网页有自己的抽帧设置，CLI 中的近墙区间和三角色调优配置单独使用。完整自动调优尚需执行适配器和可验收的参考集。

## 4. 三角色 API 配置

在 `.env` 本机填写：

```dotenv
FACADE_APINEBULA_RESPONSES_ENDPOINT=https://apinebula.ai/v1/responses
FACADE_APINEBULA_API_KEY=在本机填写
```

示例配置将方案生成、失败复盘、规则整理均设置为 `gpt-6.1-sol` 的 `standard` 模式。服务商令牌需有对应模型的调用权限。

先做本地结构校验：

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.cli model-config-check --config examples/model-gateway.apinebula.example.json
```

再执行一次评审。下面的示例请求含虚构评价数据，仅验证三角色链路。此命令会发送请求文本并消耗 API 额度：

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.cli --workspace workspace/reviews review-cycle --config examples/model-gateway.apinebula.example.json --request examples/review-request.example.json --allow-api
```

检查新生成的 `review_*` 目录中的 `state.json` 和 `trace.json`。缺少真实指标或验收阈值时状态保持待验收。调用真实项目时，将 `--request` 换成项目评审输入。

如果密钥放在独立环境文件，运行前设置：

```powershell
$env:FACADE_CONFIG_FILE=(Resolve-Path .env.apinebula.local).Path
```

项目加载配置时优先保留进程中已有环境变量。更新环境文件后，在新的 PowerShell 窗口运行，或清除相应的旧进程变量。

## 5. 视频流程、看板与 MCP

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.cli --workspace workspace --allow-root "D:\videos" video-prepare --video "D:\videos\facade.mp4" --building-id building-01 --sample-fps 2 --keyframe-range 12 18 --keyframe-fps 4 --max-frames 2000
```

输出包含原帧清单与待复核墙面范围。后续通过 `video_apply_wall_scope` 和 `video-detect` 继续检测与报告，见 [视频工作流](video-preparation-v1.md)。

已有调优运行的看板可单独生成：

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.cli optimization-dashboard --state workspace/run/state.json --output workspace/run/index.html
```

直接打开生成的 HTML 查看。看板需要真实运行产生的 `state.json`。以下合成演练可创建一份测试状态，输出目录须不存在：

```powershell
..venv\Scripts\python.exe examples/optimization_mask_demo.py --output workspace/mask-loop-demo
```

配置 MCP 客户端：

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.cli mcp-config --output mcp.local.json
```

将配置导入支持 stdio MCP 的客户端，客户端按需启动各服务。单独启动 HTTP MCP 服务主要用于客户端接入，页面入口为 `facade_mcp.web_app`。

## 6. 验证、维护和排查

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

| 情况 | 处理方法 |
|---|---|
| 页面无法打开 | 检查启动终端是否仍运行，以及实际端口。换端口后使用对应地址 |
| 未找到源码或解释器 | 修改 `FACADE_SOURCE_REPO`、`FACADE_SOURCE_PYTHON` 为本机绝对路径 |
| 权重或依赖缺失 | 按 `models` 输出补齐模型环境及权重 |
| CUDA 不可用 | 用指定的模型解释器检查 `torch.cuda.is_available()`，当前先配置 `cpu` |
| 输入路径被拒绝 | 将素材或标注所在目录加入 `FACADE_ALLOWED_INPUT_ROOTS`，Windows 多路径用分号 |
| API 返回 get_channel_failed | 用同一令牌检查服务商模型列表、分组与模型权限 |
| 配置没有生效 | 确认当前工作目录、`FACADE_CONFIG_FILE` 和进程已有环境变量 |
| 提示词修改无法自由理解 | Web 文字助手另需 `FACADE_LLM_MODEL/BASE_URL/API_KEY` 的 Chat Completions 配置，并在页面启用在线助手 |

备份 `workspace` 与本机配置可保留任务、证据和版本。更新代码后重新运行安装命令，重启服务。当前部署适合本机单用户操作；公网或多人访问还需要独立增加身份认证、访问权限和共享任务调度。
