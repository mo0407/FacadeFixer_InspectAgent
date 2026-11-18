# 三角色模型网关：第一阶段使用说明

2026-11-18 更新。本副本已实现配置检查、一次有边界的 Generator → Reflector → Curator 文本评审，以及供图像主动巡检使用的多模态输入。

## 已实现与边界

- 三个角色分别配置连接、模型名称、环境变量密钥、超时及最大输出 token。
- 支持 `mock`、`chat_completions` 和 `responses` 适配器。网关调用可通过 `image_paths` 提供 PNG/JPEG 图片证据，不同供应商可配置自己的兼容端点。
- 使用本地 Pydantic schema 验证模型 JSON，拒绝未知字段，不静默接受非法结果。
- 从调用方提供的指标值进行确定性阈值判断：没有门槛、值、证据引用或已验证参考，结果保持 pending。
- 输出计划、反思、候选经验、角色调用元数据、状态检查点。Curator 不能晋升正式经验。
- API 需要配置 `api_enabled=true` 和命令行 `--allow-api` 同时开启。演练不联网，不收费。

当前 `review-cycle` 仍是文本评审命令：不运行视频工具，不计算真实召回率，不读取图片，不自动执行改进提案，不生成分析报告。`reference_verified` 是可信调用方的声明，不是本命令完成的独立标注审计；实际测量评价器及人工验收权限后续接入。图像主动巡检通过同一网关传入 `image_paths`，连接必须显式设置 `supports_images=true`。

原有 `run`、`agent`、Video/Model/Mask/Report MCP 不被替换。新增图像入口和当前范围见 [图像工作流](image-workflow.md)；视频自动重跑及经验晋升仍按架构文档开发。

## 本地演练

在仓库目录及已安装项目的 Python 环境中运行：

```powershell
python -m facade_mcp.cli model-config-check --config examples/model-gateway.mock.json
python -m facade_mcp.cli --workspace workspace/reviews review-cycle --config examples/model-gateway.mock.json --request examples/review-request.example.json
```

若项目尚未安装，可在当前进程设置 `PYTHONPATH` 为本仓库的 `src` 目录。命令的全局参数 `--workspace` 放在子命令之前。

每次生成新的 `review_<uuid>` 目录，不覆盖已有运行：

| 文件 | 内容 |
|---|---|
| request.json | 用户任务、证据摘要及调用方提供的指标 |
| config.json | 角色/连接配置，不含环境变量密钥值 |
| plan.json | Generator 的结构化计划 |
| reflection.json | Reflector 的指标引用、证据、假设及单项改动 |
| candidate_experience.json | 待验证经验，promoted 始终为 false |
| trace.json | 模型、角色、调用结果、usage、耗时；未知费用为 null |
| state.json | 执行状态与验收状态，明确是否执行实验或生成报告 |

演练预期 `status=review_completed`，但 `acceptance_status=pending_review`。完成评审不等于检测质量达标。模拟模型输出仅验证程序流程，不作为经验有效性的证据。

## 接入不同供应商

文本评审可复制 `examples/model-gateway.remote.json`，为三个角色分别填写实际模型名称。图像主动巡检可复制 `examples/image-models.remote.example.json`，其中 Generator 为远程图像连接，Reflector、Curator 保留 mock 离线占位。各 `endpoint_env` 指向完整的 HTTPS API 端点，不是仅填写域名；具体端点与模型能力按供应商文档配置。

在本地环境设置配置中引用的 endpoint/key 环境变量。不要把 API key 写进 JSON、代码库或聊天。准备好请求文本的外发范围后，设置 `api_enabled=true`，并用 `--allow-api` 执行。

适配器发送 messages 与 max_tokens，要求返回 choices[0].message.content 中的 JSON 字符串，finish_reason 必须是 stop。不支持此协议或输出约定的模型应新增适配器，不依靠静默回退伪装成功。配置检查只校验结构和引用，不探测远程能力或环境变量凭据。

`responses` 适配器发送 `input`、`max_output_tokens` 和 `store: false`。配置 `reasoning_mode: "standard"` 时发送 `reasoning.mode: "standard"`。只接受状态为 `completed` 的 JSON 文本结果。

## 多模态输入

网关 `invoke(..., image_paths=[...])` 读取调用方已准备的本地图片快照。仅允许有效的 PNG/JPEG，每次最多 8 张，每张最多 10 MiB，总计最多 30 MiB。`supports_images=true` 的门控对 mock 同样生效；该配置声明不等于远程视觉能力已经验证。

- `chat_completions` 将文本和 `image_url` 图片内容放入消息内容中。
- `responses` 使用 `input_text` 和 `input_image` 内容块。
- 审计中的图片记录只保留 SHA-256、尺寸、字节数和媒体类型，不保存图片路径或 base64；密钥只从环境变量读取。

图像工作流使用网关执行有界的检查决策和最终复核；`review-cycle` 不会因为连接配置为支持图片而自动读取图片。

没有自动重试或自动跨供应商切换。网络超时记为 request_outcome_unknown，因为服务端可能已执行/计费。失败时保存脱敏轨迹与失败状态，先核对供应商记录再重试。

## 验证

`python -m pytest tests/test_model_gateway.py tests/test_image_gateway.py -q`

覆盖：验收门槛缺失、未验证参考、失败/通过判断、非有限指标、未知证据引用、模拟三角色运行、经验禁止晋升、API 显式开关、兼容协议响应解析、密钥不进入轨迹、超时不重试、未知输出字段拒绝。

多模态测试还覆盖图片校验、数量/大小限制、能力门控、两种协议的图片载荷及 trace 脱敏。测试使用模拟 HTTP，不证明真实供应商兼容性；真实远端视觉能力尚未验证，需在用户配置的实际模型上另行验收。
# 后续进展

## ACE 上下文增量管理

参考本地 ACE 官方源码快照中的三角色分工、条目 ID 和增量更新方式，评审现支持传入实际规则内容：

```powershell
python -m facade_mcp.cli --workspace workspace/reviews review-cycle --config examples/model-gateway.mock.json --request examples/review-request.example.json --playbook 已采用规则.json
```

规则快照包含 `version` 和 `rules`。每个条目包含稳定的 `id`、`condition`、`recommendation`，可附 `limitations`、`evidence_ids`、`helpful`、`harmful`。版本须匹配请求的 `playbook_version`。省略参数时使用同版本的空规则快照，兼容旧命令。

三个角色均读取本轮冻结快照。Generator 通过结构化字段 `used_rule_ids` 声明实际引用，Reflector 使用 `rule_feedback` 记录条目 ID、helpful/harmful/neutral、证据及原因。程序校验条目是否实际被引用、证据是否存在。反馈计数属于待验证提案，保留独立验收后的采纳入口。

Curator 可提议 add/revise/deprecate，后两者要求 `target_rule_id`。程序对 add 按规范化条件与建议做精确去重，给每条提案编号并保存父快照哈希。语义相近但文字不同的规则仍需审核。

新增产物 `playbook_snapshot.json` 保存完整上下文，`context_delta.json` 保存候选增量及规则反馈，`trajectory.json` 串联计划、评价、引用和反思。原有输出文件继续保留。调优控制器通过 `playbook=` 接收同样的快照，整次运行的版本冻结。

本阶段产物供后续独立验证与审核使用，已采用规则库的持久化合并和自动晋升仍待实现。执行日志使用结构化引用，避免依赖提示词文本中的 ID 正则提取。测试覆盖上下文注入、重复规则、错误目标、引用范围和版本冲突。

已增加[像素评价与有界调优控制器](optimization-controller-v1.md)，支持独立计算配对 mask 指标，以及通过显式执行适配器重测。视频自动执行仍未接入。
