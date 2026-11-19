# 失败记忆与重复原因统计

失败记忆把“程序运行出错”“有依据的质量问题”“需要复核的观测”和“复盘提出的原因推测”分开记录。它先回答发生了什么、证据在哪里、同类问题出现几次，再为后续优化提供材料。原因出现次数多，不等于解释已被证实，也不等于某条改进规则已经有效。

## 保存在哪里

统一数据库位于 `workspace/failure_memory.sqlite3`，其中 `workspace` 是运行命令的 `--workspace` 目录。同一工作区的不同任务汇入同一个数据库；其他工作区不会自动并入。

每个任务目录另保存 `failure_summary.json`，便于单独检查和归档该任务。它是写入时的本任务快照，不是全局统计的实时视图；其他任务后续新增事件，不会回写旧任务的 JSON。CLI/MCP 查询则从当前 SQLite 数据库现查。执行失败时保存 `failure.json`，记录明确的运行失败状态与安全上下文。成功完成图像处理也可能包含人工指出的质量问题；任务运行状态和检测质量判断是两回事。

人工反馈后更新本任务的最新汇总，并另存 `feedback-事件ID.json`。完成时登记为不可变 artifact 的快照使用 `failure_summary-at-completion.json`，不会因后续反馈改写。并发反馈的汇总发布与 SQLite 写事务协调，避免旧快照覆盖新计数。配置或请求文件在正常任务启动前解析失败，也会生成一条预检查失败任务。

运行错误可带有阶段、专家 profile、VLM 模型、P0 版本与输入哈希等上下文，用于定位问题。记录采用受控原因代码与安全字段，不将密钥、图片 base64 或任意远端错误正文作为失败原因保存。原始图像证据仍位于对应任务目录。

## 哪些记录进入失败统计

| 记录类别 | 依据 | 是否进入失败数量 |
| --- | --- | --- |
| 运行失败 | 执行框架实际捕获的模型、工具、输入或输出校验等错误 | 是 |
| 人工确认的质量问题 | 可追溯的复核者说明及同任务证据 | 是 |
| 指标未达目标 | 已验证参考与实测指标，`review-cycle` 判定为 fail | 是，原因代码为 `metric_below_target` |
| 指标 pending | 缺目标、缺实测值或参考未验证 | 否 |
| 待复核观测 | 程序发现不确定性或执行了保守约束 | 否，单独汇总 |
| Reflector 解释 | 模型复盘提出的可能原因 | 作为 `hypothesis` 保留，不当作已确认的根因 |

`review-cycle` 也记录执行时实际发生的 API 超时、无效 JSON 或 schema 校验错误等运行失败。Reflector 的解释保存在评审任务 `failure_summary.json` 的 `insights` 中，带 `cause_status="hypothesis"`、`confirmed=false`；这些解释不额外增加数据库失败事件数。

例如，“专家返回了空候选集”是可观察的结果；只有经过人工复核或合格参考证明图片中确有目标缺陷，才能把它记录为漏检。“图片缩小后可能丢失细节”可以是复盘假设，不能仅凭这句话就证明漏检的原因。

## 质量问题的固定代码

人工反馈选择以下代码，配合具体说明与证据。不要用自由文本替代原因代码，否则同一问题容易被拆成许多无法比较的名称。

| `reason_code` | 含义 | 反馈应说明 |
| --- | --- | --- |
| `missed_defect` | 漏检 | 哪处可确认缺陷没有出现在结果中，以及定位证据 |
| `false_positive` | 误报 | 哪个候选实际为非缺陷，以及判断依据 |
| `wrong_class` | 类型错误 | 当前类型、复核类型及可见证据 |
| `bbox_inaccurate` | 检测框不准确 | 位置偏移、覆盖不足或包含过多非缺陷区域 |
| `mask_inaccurate` | mask 不准确 | 分割边缘、漏分或多分情况及对应产物 |
| `scope_error` | 范围错误 | 门窗、外楼梯、背景或目标墙面范围的处理错误 |
| `report_error` | 报告错误 | 图片、文字、编号、引用或内容不一致的位置 |
| `duplicate_defect` | 缺陷重复 | 被重复表达的候选及其对应关系 |

图像主动检查以框级核验为核心，注册分割专家可附带 mask 证据；`mask_inaccurate` 仅用于实际存在 mask 产物的反馈。代码表中的一种问题未出现，并不表示该能力已经通过验收。

没有人工真值时，不能只根据置信度低或模型意见不同就断言漏检、误报或类型错误。人工反馈是带来源的质量结论，不会自动建立整张图的完整真值，也不会自动生成准确率或召回率。

## 观测单列

| 观测代码 | 含义 |
| --- | --- |
| `unknown_class` | 专家类别无法映射到当前业务定义 |
| `scope_uncertain` | 缺陷或墙面范围仍不明确 |
| `evidence_downsampled` | 候选证据为适配输入而被缩小 |
| `harness_override` | 执行框架根据范围、类别或证据约束覆盖了模型决定 |
| `candidate_rejected` | VLM 在核验中排除了某个候选，该决定不等于经人工真值确认的误报 |

这些记录可以帮助发现需要人工检查的地方，但不计入失败总数。例如，因证据缩小而强制 `defer` 是一个可追溯的保守动作，不等于已经发生漏检；如果之后人工确认结果确实漏掉了缺陷，可另行提交质量反馈。

mock 视觉输出不作为图像观测写入统计，包括合成演示及 mock VLM 的待定/排除决定。此时任务快照中的 `mock_observations_omitted=true` 表示已省略这类模拟观测，而不是证明真实图像没有问题。

## 同因计数与幂等

汇总以固定 `reason_code` 聚合，而不是对整段错误文案做字符串匹配。查看汇总时区分：

- **事件数量**：该原因下的独立事件数。
- **涉及任务数**：这些事件分布在多少个不同任务中。
- **重复次数**：同一原因的事件数量减一，即 `count - 1`；第一次出现不算重复。

一个任务中可以有多个不同事件，同一个事件也可能被重新保存。`occurrence_key` 是任务内的稳定事件标识，例如 `review-001:candidate-0001`。重提同一事件时沿用该标识，程序返回首次保存的记录，不覆盖原说明，也不增加事件数；若同一标识改用另一原因代码，会明确报错。确有新的独立问题才使用新的标识。不要通过反复生成新标识把同一问题重复提交。

结构化汇总中，`reasons` 的每一项包含 `count`、`affected_jobs`、`repeat_count`、首次/最近出现时间和建议动作。总览 `totals` 包含：

| 字段 | 口径 |
| --- | --- |
| `runtime_failures` | 运行失败事件数 |
| `quality_failures` | 有依据的质量声明事件数 |
| `observations` | 单列的观测事件数 |
| `events` | 上述三类事件总数 |
| `affected_jobs` | 三类事件涉及的任务去重数，包含仅有观测的任务 |

因此，`totals.affected_jobs` 不能直接解释为“失败任务数”。某一原因项的 `affected_jobs` 才是出现该 `reason_code` 的任务数。`events` 列出事件明细，`latest_event` 为当前查询范围内最近一条事件，没有事件时为 `null`。

“同因”在这里表示同一个受控原因类别，不能单凭次数推出相同的物理根因。阶段、模型、参数、P0 和证据用于进一步区分原因出现的条件；要证明某个改动有效，仍需在独立参考上对照验证。

## CLI 用法

在已安装项目并激活环境后，查看整个工作区：

```powershell
facade-image --workspace workspace failure-summary
```

限定任务并生成 HTML 汇总：

```powershell
facade-image --workspace workspace failure-summary --job-id image_inspection_实际任务ID --output failure-dashboard.html
```

HTML 供查看汇总、事件与证据引用；SQLite 和各任务的 JSON 保留结构化记录。统计只覆盖已写入该工作区失败记忆的记录，不自动补录旧运行目录。

人工反馈模板见 `examples/image-feedback.example.json`：

```json
{
  "job_id": "image_inspection_00000000000000000000000000000000",
  "reason_code": "bbox_inaccurate",
  "occurrence_key": "review-001:candidate-0001",
  "summary": "人工复核说明：该框未覆盖缺陷右侧边缘。",
  "evidence_refs": ["candidate-0001-evidence.png"],
  "reviewer": "reviewer-name",
  "candidate_id": "candidate-0001"
}
```

示例中的全零任务 ID 只是满足格式的占位值，不对应真实任务。替换为真实任务、复核说明和证据后提交：

```powershell
facade-image --workspace workspace failure-feedback --config examples/image-feedback.example.json
```

反馈字段约束：

| 字段 | 要求 |
| --- | --- |
| `job_id` | 已完成或失败的图像任务，格式为 `image_inspection_` 加 32 位小写十六进制字符 |
| `reason_code` | 上文列出的 8 类人工质量代码，不接受运行代码或 `metric_below_target` |
| `occurrence_key` | 1～160 字符的任务内稳定标识 |
| `summary` | 1～1000 字符的复核结论及依据 |
| `evidence_refs` | 1～30 个该任务目录下的现存文件相对路径 |
| `reviewer` | 1～100 字符的复核者标识 |
| `candidate_id` | 可省略；提供时须为真实存在的 `candidate-0001` 形式的候选 ID |

任务必须已经存在，正在运行的任务和非图像任务不接受此人工反馈。`evidence_refs` 不接受绝对路径、越界路径或带 `#` 的片段，也不能引用可变的 `failure_summary.json`、`failure.json`、`job.json` 作为质量证据。不读取任意外部文件，不用不存在的文件名充当证据。程序保存所引证据的哈希和复核者声明，`confirmed` 表示调用方确认，未经过独立审计。

提供候选 ID 时，应指向实际复核对象。缺陷漏检可能没有现成候选 ID，可省略该字段，在反馈说明和任务内定位证据中明确位置。反馈成功返回 `event` 与更新后的本任务 `job_summary`，并明确 `playbook_updated=false`、`automatic_retry=false`；幂等重提时 `event.inserted=false`。

## MCP 用法

图像 MCP 在原有检查入口之外提供：

| 工具 | 入参 | 用途 |
| --- | --- | --- |
| `facade_image_failure_summary` | 可选 `job_id`、默认关闭的 `output_html` | 获取工作区或单任务汇总，可选择输出 HTML |
| `facade_image_record_feedback` | `feedback` 对象 | 提交与 CLI 相同结构的人工质量反馈 |

工具通过当前 MCP 服务的工作区配置访问数据库和任务产物；反馈证据受同任务目录约束。

`output_html=false` 时仅返回当前数据库汇总；设置为 `true` 时，返回结果中增加 `dashboard`。全局 HTML 保存到工作区的 `failure-dashboard.html`，指定任务则保存到工作区的 `failure-dashboard-任务ID.html`；任务可以是图像检查或三角色评审任务。

## 与复盘和规则的关系

失败记录可作为 Reflector 的证据输入，模型提出的解释标为 `hypothesis`。Curator 可以据此提出规则候选，但重复次数不会直接把解释变成事实，也不会把规则候选自动变成“已验证有效的规则”。

本模块不自动改写 P0、不自动训练模型、不自动支付或启动下一轮重跑。要采用新的规则，仍需独立验证、显式审核与版本更新；当前运行继续使用冻结的 P0。

可显式运行图像版离线规则生命周期，读取有证据失败、提出候选并在独立参考集上成对验证。只有满足发布约束和正向改善的真实执行才可发布下一版规则；失败计数或模型解释本身不会触发晋升，详见 [完整流程](complete-image-workflow.md)。
