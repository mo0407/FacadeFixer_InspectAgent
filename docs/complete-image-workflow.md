# 单图主动巡检与规则验证

本说明描述当前已提供的接口。Baseline 的具体实验约定、初始 P0 的自动 Generate→Match 启动、逐条规则开关和失败后自动修订仍有待补项，详见 [三图需求对照与验收](requirements-three-figures.md)。下述成对验证是整本 parent/candidate P0 对比，不等同于多条规则的逐条消融。

当前流程处理一张静态 PNG/JPEG：保存原图快照，调用本地专家，建立可引用锚点，允许 VLM 最多选择两次原像素局部检查，再逐候选核验并生成报告草稿。坐标使用未按 EXIF 旋转的存储像素，框为零起点、右下边界不包含的 `xyxy`。`keep/reject/defer`、模型置信度和草稿均需要人工复核，不代表测得的准确率。

## 场景与局部检查

`context_mode` 默认 `auto`。有 `scope_mask_path` 时，输入必须是与原图同尺寸、非空、仅含 0/255 的灰度 PNG；系统从其墙面连通域生成 Env 锚点，并用 mask 检查候选的范围。没有 scope mask 时，VLM 返回 `SceneContext`，给出目标墙面、门窗、外楼梯、背景等归一化矩形及说明，再映射回原图。Env 是环境上下文框，不是缺陷真值 mask；VLM 的场景判断也不会生成权威墙面 mask。还可用 `context_regions` 补充手工框，或选择 `manual/none` 控制自动提取。

专家能力从模型 MCP 的 `facade_models_capabilities` 发现，包括 `class_codes`、`full_image`、`local_eligible` 和实际权重 SHA-256。`expert_class_filters` 只能缩小声明类别。局部动作必须引用已有锚点和该锚点的合格专家；已知类别按支持类别筛选，范围外锚点不可局部检查。至少一个专家需支持整图检测。

局部 `scale=1/2/3` 表示扩展锚点周围的上下文，裁剪直接取自原像素，不把预览放大作为检测输入。局部新发现会生成新锚点，并刷新后续可选锚点及证据图。每轮最多两次，重复的“锚点、专家、scale”组合会被拒绝。专家 checkpoint 身份在整图与局部调用间锁定；漂移会失败，不能在一轮中混用新旧权重。

冻结校验还覆盖类别映射、backend/task 和关键实现文件的定义指纹，以及实际运行报告的库版本等实现信息；集成模型按成员名称逐一对应权重哈希。离线配对重放锁定全部已配置专家的定义库存，允许新规则选择此前未使用的局部专家，同时核对每个实际使用专家的实现身份。云端模型仅能核对服务商返回的模型 ID，不能验证服务商内部的物理权重。

## 注册自己的分割专家

复制 [专家注册格式](../examples/experts.registry.example.json) 后，在本机填入已安装的 source checkout、Python runtime、训练 checkpoint 和实际类别映射。`yolo9e_cubit` 只是示例 profile 名，不证明本机已有 CUBIT 权重或该权重通过验收。通用预训练模型也不能仅靠改名变成建筑缺陷专家。

示例含 YOLOv9-seg 和 Ultralytics-seg 两种 backend。YOLOv9 要求 `task=segment` 且配置 `repo_path`；Ultralytics 支持 `detect/segment`。`class_map` 的键必须替换为 checkpoint 的真实 label 或类别 ID 字符串，值只能是 `C/S/T/R/D/V/L/E/B`，且 `class_codes` 集合必须等于映射值集合。示例中的占位 label 和类别组合只演示格式，不是训练类别声明。

注册文件由服务端环境指定，调用方只能选择命名 profile，不能通过检测 MCP 自定义 weights、Python 或命令。下面的设置应提供给启动模型 MCP 的本机进程；已有 MCP 进程需重启才能加载新注册文件。

```powershell
Copy-Item .\examples\experts.registry.example.json .\examples\experts.registry.local.json
# 编辑 local 文件，替换所有占位路径和 label 后再启用。
$env:FACADE_EXPERTS_CONFIG = (Resolve-Path .\examples\experts.registry.local.json).Path
$env:FACADE_EXPERTS_ALLOWED_ROOTS = "E:\models;E:\runtimes;E:\sources"
```

示例内的所有路径均位于配置父目录下；相对路径按该父目录解析。外部绝对路径必须处于 `FACADE_EXPERTS_ALLOWED_ROOTS` 的分号分隔白名单中。可选 `checkpoint_sha256` 应填本机真实文件的 64 位小写 SHA-256，用于固定版本；示例未填写伪 hash。能力查询只检查文件及 hash，不加载模型、不下载资产；`ready=true` 与 `model_load_verified=false` 并存，不代表推理成功。

`python_path` 可指向虚拟环境内的 Python 符号链接，启动时保留该入口，以便 Python 找到虚拟环境。注册时链接入口和解析后的目标均须位于上述服务器允许根内；若基础解释器位于其他目录，也应将其目录加入白名单。

自定义模型运行时代码可在 profile 中用 `implementation_files` 声明，例如 `{"backbone": "server-assets/runtime/ultralytics/nn/backbone/efficientViT.py"}`。键只能使用字母、数字、点、下划线或连字符，文件路径遵循同样的服务器目录白名单。声明文件缺失时专家不可用；文件的 SHA-256 会进入专家定义指纹，检查前后不一致时任务会失败。此配置只记录并检查声明的文件，并不自动导入它们，也不证明 Python 实际导入的文件与声明一致；仍须在独立运行环境核实实际加载路径和模型推理。

分割专家返回的每实例 mask 以该次输入图的原尺寸二值 PNG 保存，并校验路径、尺寸、像素值和 hash，附带 RLE、面积和 artifact。局部输入的 mask 尺寸对应局部原像素裁剪，候选框另行平移回全图。结构化巡检结果保留专家 mask 的 artifact 引用，报告主输出仍是候选框与核验决定；需要生成或点修最终 mask 时使用 `facade_models_segment` / `facade_models_refine`。`B` 表示鼓包，是业务类别；OpenCV mask baseline 明确不支持 `B`，应选择适用的模型分割器并复核结果。

## 模型配置与一次运行

从项目根目录执行，使用已安装项目依赖的 Python。示例环境路径可按本机替换：

```powershell
$python = "E:\ai\低空\facade_mcp_langchain\.venv\Scripts\python.exe"
$env:PYTHONPATH = "src"
& $python -m facade_mcp.cli model-config-check --config .\examples\image-models.remote.example.json
& $python -m facade_mcp.cli --workspace .\workspace image-demo --output-dir .\examples\generated_image_demo
```

[远端配置示例](../examples/image-models.remote.example.json) 使用 Responses、`gpt-6.1-sol`、`reasoning_effort=medium`、`expected_model=gpt-6.1-sol`；generator、reflector、curator 共用 `image_inspector` 连接。endpoint/key 通过 `FACADE_IMAGE_ENDPOINT` / `FACADE_IMAGE_API_KEY` 环境变量读取，JSON 只保存变量名。当前示例 `api_enabled=false`。`model-config-check` 仅校验配置；`image-demo` 使用合成数据和 mock，不证明真实检测或视觉能力。

实际运行需复制配置并按本机服务设置 endpoint/model，显式将副本的 `api_enabled` 设为 `true`，同时传 `--allow-api`。这会向配置的服务发送概览、局部或候选证据图。`expected_model` 检查服务实际返回的模型身份，并在一轮中固定它；不一致会拒绝结果。

```powershell
& $python -m facade_mcp.cli --workspace .\workspace image-inspect --config .\examples\image-inspection.example.json --model-config .\examples\image-models.local.json --allow-api
```

先替换请求中的照片路径和 `detector_profiles`。CLI 按请求 JSON 父目录解析 `image_path/scope_mask_path`；MCP 请传允许根内的明确路径或 artifact URI。结果在 workspace 的任务目录中，包含 source、anchors、局部证据、trajectory、专家调用、模型 trace、P0 快照、`result.json` 和 `report.html`。零候选不证明不存在缺陷。

## 失败记忆

运行异常保存阶段和稳定原因码；人工质量反馈通过 [反馈格式](../examples/image-feedback.example.json) 指定真实 job、问题类型及该 job 内已有证据文件。重复提交相同 occurrence key 幂等。候选被模型排除等已实现的观察项单列，不自动成为质量失败；待定或低分同样不能直接证明漏检、误报。人工声明也不等同独立审计。查询汇总可看原因、次数、证据及 HTML 看板：

```powershell
& $python -m facade_mcp.cli --workspace .\workspace failure-feedback --config .\examples\image-feedback.local.json
& $python -m facade_mcp.cli --workspace .\workspace failure-summary --output .\workspace\failure-dashboard.html
```

## 离线规则候选、配对验证与发布

“离线”指巡检任务之外的独立规则生命周期，不表示禁用已授权的远端模型调用。`rule-propose` 从有证据的非 mock 历史失败提炼假设、受控改动及引用，仅生成候选 P0，不采用它。候选、父版本、证据和元数据有 hash 绑定；应使用命令返回的 `candidate_playbook_path`，不能手写候选文件绕过注册。

[验证集格式](../examples/rule-validation.example.json) 有 3 个占位 case、5 个格式示意框；路径、类别和框都必须替换为独立人工参考。它没有提供真实 GT，`independent_reference_confirmed=false` 会阻止发布。不能只把该标记改成 true 就把示例框当作参考。实际参考图片不能与提案的历史失败来源重合，也不能重复；bbox 必须位于各自原图内。示例阈值同样只是配置示意，应按任务要求事先确定。

验证在同一批独立图片上分别运行父 P0 和候选 P0，保持请求、模型配置、专家权重及实际 VLM 身份一致。指标按同类 bbox 的 IoU 阈值做一对一匹配，计算 precision/recall/F1，不是像素 mask 指标。发布要求满足所有目标阈值、各指标不超允许退化，并有正向 F1 改善；默认 `min_improvement=0.01`、至少 3 张图和 5 个参考缺陷。正式发布还需真实完成的图像任务、非 mock 模型 trace 和 MCP 专家调用证据。mock 提案、模拟 HTTP、注入 runner 均不能晋升。

```powershell
# 可重复 --job-id 选择历史失败；省略时使用合格历史事件。
& $python -m facade_mcp.cli --workspace .\workspace rule-propose --model-config .\examples\image-models.local.json --job-id image_inspection_REPLACE_WITH_REAL_ID --allow-api
& $python -m facade_mcp.cli --workspace .\workspace rule-validate --model-config .\examples\image-models.local.json --candidate "REPLACE_WITH_RETURNED_CANDIDATE_PATH" --dataset .\examples\rule-validation.local.json --allow-api
& $python -m facade_mcp.cli --workspace .\workspace rule-latest
# 一次有界的提案 + 验证尝试；不会持续循环或保证发布。
& $python -m facade_mcp.cli --workspace .\workspace rule-cycle --model-config .\examples\image-models.local.json --dataset .\examples\rule-validation.local.json --allow-api
```

通过后以事务发布不可变版本；若父版本已过期则拒绝晋升。未显式指定 `--playbook` 的下一次巡检自动加载已验证 latest，无发布版本时使用空 P0。一次巡检开始即深拷贝并冻结其版本和 hash；本轮失败、反馈或同时发布的新规则不会修改正在运行的 P0。

## 视觉连接显式真验

`supports_images=true` 是声明。要检查真实连接，使用已启用 API 的本机配置显式运行：

```powershell
& $python -m facade_mcp.cli --workspace .\workspace vision-check --model-config .\examples\image-models.local.json --output .\workspace\vision-validation.json --allow-api
```

该命令用合成编号色块检验 generator 的颜色、数字、结构化动作和实际模型身份。只有真实远端请求通过才返回 `capability_verified=true`；未启用 API 会 `not_run`，mock 与注入 HTTP 不能认证真实视觉能力。即使通过，也仅证明这一次合成探针，不证明建筑缺陷准确率。本说明不声称远端视觉或真实 CUBIT 权重已验收。

## 对应 MCP 入口

用 `python -m facade_mcp.cli mcp-config --include-image --output mcp.local.json` 导出包含 image 服务的客户端配置；本地专家推理仍由 model MCP 执行。MCP 远端调用同样要求模型配置 `api_enabled=true` 和工具参数 `allow_api=true`。

| CLI | Image MCP 工具与主要参数 |
| --- | --- |
| `image-inspect` | `facade_image_inspect(request, model_config_path, playbook_path?, allow_api)` |
| `failure-summary` | `facade_image_failure_summary(job_id?, output_html)` |
| `failure-feedback` | `facade_image_record_feedback(feedback)` |
| `rule-propose` | `facade_image_rule_propose(model_config_path, playbook_path?, job_ids?, allow_api)` |
| `rule-validate` | `facade_image_rule_validate(candidate_path, dataset_path, model_config_path, allow_api)` |
| `rule-cycle` | `facade_image_rule_cycle(dataset_path, model_config_path, job_ids?, allow_api)` |
| `rule-latest` | `facade_image_rules_latest()` |
| `vision-check` | `facade_image_validate_vision(model_config_path, allow_api)` |

CLI 的 `rule-propose/rule-cycle` 还可指定 `--playbook`；MCP `rule-cycle` 使用当前 latest 父版本，没有该参数。模型 MCP 可用 `facade_models_capabilities` 查询能力和 hash、`facade_models_detect` 运行命名专家、`facade_models_segment/refine` 生成或修订 mask、`facade_models_get_job/get_artifact` 追溯结果。
