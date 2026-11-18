# FacadeFixer VIDEO

建筑外立面视频处理、缺陷检测与分割、报告草稿、三角色评审及 ACE 风格上下文管理。

## 工作流

```mermaid
flowchart LR
    V[原始视频] --> F[2 fps 抽帧 / 近墙区间 4 fps]
    F --> S[墙面范围复核]
    S --> D[本地检测]
    D --> M[原图分割]
    M --> R[Report MCP 草稿]
    R --> E[参考集评价]
    E --> G[Generator]
    G --> A[Reflector]
    A --> C[Curator 候选增量]
    C --> P[独立验证与审核]
    P --> B[已采用规则快照]
    B --> G
```

已实现视频分批抽帧、范围复核、本地检测与分割接口、报告草稿、指标评价工具、有界调优控制器及规则增量提案。完整自动视频调优回调、跨帧实体缺陷归并、规则持久化合并与自动晋升仍待实现。

## 安装

完整步骤见 [部署方法](deployment.md)，包含 Windows 安装脚本、模型环境、Web 启动、三角色 API、视频处理、MCP 接入与排查。

Python 3.11 或更高版本，在仓库目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,web,pdf]"
Copy-Item .env.example .env
```

本地模型另需已有的 FacadeFixer-main 源码、权重和支持 PyTorch/Ultralytics 的运行环境。配置 `FACADE_SOURCE_REPO`、`FACADE_SOURCE_PYTHON` 与 `FACADE_SOURCE_DEVICE`。默认使用 CPU。OpenCV 分割另需配置 FacadeFixer_mask_api 源码路径。

## 视频准备与检测

```powershell
facade-workflow --workspace workspace --allow-root "D:\videos" video-prepare --video "D:\videos\facade.mp4" --building-id building-01 --sample-fps 2 --keyframe-range 12 18 --keyframe-fps 4
```

复核每帧墙面范围后，通过 `video_apply_wall_scope` 输出检测输入。将返回的 manifest URI 填入检测配置：

```json
{
  "wall_inputs": "artifact:实际产物ID",
  "project": {"name": "外墙视频巡检"},
  "detector_profile": "yolo12m",
  "segmenter_profile": "mobile_sam",
  "prompt_mode": "bbox",
  "keyframe_inference_size": 1024,
  "max_frames": 100
}
```

```powershell
facade-workflow --workspace workspace video-detect --config video-detect.json
```

报告按帧输出候选。单帧候选数与物理缺陷数量分别记录，范围边界候选保留在审计清单中。

## 三角色与 ACE 上下文

```powershell
facade-workflow --workspace workspace/reviews review-cycle --config examples/model-gateway.mock.json --request examples/review-request.example.json
```

三个角色可分别配置 `mock`、OpenAI 兼容 `chat_completions` 或 `responses` 端点。模型密钥通过配置指定的环境变量读取。远程调用同时需要 `api_enabled=true` 和 `--allow-api`。

使用 Responses 的 `standard` 模式，可将连接设置为 `"adapter": "responses"`、`"reasoning_mode": "standard"`。服务商模型权限与可用模式需另行确认。

用 `--playbook rules.json` 传入版本匹配的规则快照。Generator 引用规则 ID，Reflector 提交带证据的条目反馈，Curator 生成新增、修订、停用提案。程序校验引用、去除精确重复，输出 `playbook_snapshot.json`、`context_delta.json` 和 `trajectory.json`。提案经过独立验证与审核后才能纳入已采用规则。

## 看板与 MCP

```powershell
facade-workflow optimization-dashboard --state workspace/run/state.json --output workspace/run/index.html
facade-workflow mcp-config --output mcp.local.json
facade-web
```

看板展示轮次、报告链接、指标改善、模型参数、规则库与三角色流程图。MCP 客户端配置提供 video、model、mask、report、survey 五个 stdio 服务。

## 目录

- `src/facade_mcp/video_service.py` 和 `video_detection_workflow.py` 管理视频与检测链路。
- `model_gateway.py`、`review_cycle.py` 和 `optimization.py` 管理多模型调用及调优。
- `playbook.py` 管理规则结构、反馈及候选增量。
- `optimization_dashboard.py` 生成独立 HTML 看板。
- `tests/` 包含模型边界、MCP、报告、评价和上下文验证测试。

使用方法与当前限制详见 [视频工作流](video-preparation-v1.md)、[三角色网关](model-gateway-v1.md)、[调优控制器](optimization-controller-v1.md)。

## 验证与参考

```powershell
python -m pytest -q
```

采用 C、S、T、R、D、V、L、E、B 九类业务定义，排除门、窗和外楼梯。旧运行时的 P/S 合并及 B 类支持仍需迁移验证，详见 [评价标准](evaluation-standard-v1.md)。AI 标注保留待人工复核状态。缺少合格真值时准确率指标保持待验收。

本仓库从 FacadeFixer MCP 工作区整理，保留图片、报告与 Web 服务依赖。ACE 上下文架构参考 [ace-agent/ace](https://github.com/ace-agent/ace)，本仓库实现了适用于外墙工作流的规则数据结构与候选增量校验。
