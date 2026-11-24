# 图 1 与 P0 实验记录和运行说明

本说明记录一次实验的边界和仓库中的入口，不表示已经训练模型或完成全部四图实验。原始数据、权重、私有配置、调用凭据、实验输出和恢复过程中的工作目录均未随 Git 仓库分发。取得并核对这些材料后，才能实际运行；仅凭仓库无法重现这次最终恢复版本或其结果。

## 实验状态

- 图 1 使用 CUBIT 原 val 子集中的 203 张图片，分别处理 YOLO11m、YOLOv9-e 和 EfficientViT 的专家结果，并进行 VLM 判决。这些数据及模型权重的训练和选模来源仍需核实。图 1 的入口是推理流程，不执行训练。
- 203 张图的结果参与参考标注匹配。一张图片 `SP0_1018` 存在零面积标注，尚待人工审核，因此 P0 候选构建按部分参考复核处理，排除了该病例；其余 202 张产生的 4720 条待处理差异事件被划分为 875 批。差异事件并不等同于已证实的误报或漏报，类别映射和标注完整性仍需审核。
- 875 批已处理，得到 2082 条规则候选，状态为 `partial_reference_review`。它们是待审核提案，尚未验证、注册或推广为正式 P0，不能声称检测质量已改善。此次过程复用了早期批次并历经提示修订和多次恢复，细节依赖私有运行记录；不要把该结果简化为一次统一提示的独立运行。
- 这 203 张图及其标注已用于 P0 开发，不能作为该 P0 的独立最终测试集。需要另备与模型训练、选模和 P0 开发隔离的数据，并检查图片、近重复、建筑和拍摄序列是否重叠，才能评估泛化能力。图 2 的 P0 对照、图 4 的无 Playbook 对照及逐条规则验证需另行安排。

## 准备与入口

安装项目和所需模型依赖，按 [模型接入说明](../model_assets/README.md) 自行准备专家注册文件、代码和权重。使用真实视觉服务时，准备支持图片输入的模型配置和环境变量；不要将凭据、真实配置或结果加入版本库。图 1 批处理和 P0 v2 入口目前依赖 POSIX 文件锁。

以下命令中的尖括号为需要替换的占位符，不是仓库自带的文件。可先查看完整参数：

```bash
python -m facade_mcp.figure1_runner --help
python -m facade_mcp.baseline_p0_cli --help
python -m facade_mcp.baseline_p0_cli match --help
python -m facade_mcp.baseline_p0_cli curate --help
python -m facade_mcp.p0_v2_cli --help
python -m facade_mcp.p0_v2_cli prepare --help
python -m facade_mcp.p0_v2_cli run --help
```

### 图 1：专家与 VLM

先运行或恢复专家阶段。`--output` 必须位于 `--workspace` 之下，图片目录仅扫描其直接包含的 JPEG/PNG，数量必须与 `--expected-count` 一致：

```bash
python -m facade_mcp.figure1_runner \
  --images-dir '<图片目录>' --expected-count 203 \
  --workspace '<工作目录>' --output '<工作目录>/figure1-run' \
  --source-repo '<模型源码目录>' --source-python '<模型环境的 Python>' \
  --experts-config '<专家注册 JSON>' --experts-only
```

使用相同输入、输出目录和专家配置，将 `--experts-only` 换成 `--complete --model-config '<模型配置 JSON>' --allow-api`，进行真实 VLM 阶段。真实服务会收到图像证据并可能产生费用，模型配置须明确启用 API、指定预期模型且支持图像。失败任务需显式传入 `--retry-failed`；如先前请求结果未知，仅在核对该情况后才考虑额外传入 `--acknowledge-unknown`，避免误认为重试不会重复计费。

### 标注匹配与候选生成

匹配需要已完成的图 1 汇总、对应多边形标注，以及经过人工核对的类别 ID 映射。以下映射仅示范 JSON 语法，必须替换为实际数据的类别映射：

```bash
python -m facade_mcp.baseline_p0_cli --workspace '<工作目录>' \
  --allow-root '<输入允许目录>' match \
  --summary '<图 1 汇总 JSON>' --labels '<标注目录>' \
  --class-map '{"0":"C","1":"S"}' --expected-count 203
```

仅在核实指定类别标注完整后使用 `--reference-complete`；`--assume-reference-complete` 只表达未经独立审核的条件假设。存在标注问题时，不能把不完整标注推断出的差异直接当成可靠的误报或漏报。

```bash
python -m facade_mcp.baseline_p0_cli --workspace '<工作目录>' \
  --allow-root '<输入允许目录>' curate \
  --match-batch '<匹配结果 JSON>' --model-config '<模型配置 JSON>' \
  --expected-count 203 --allow-api
```

`curate` 对匹配差异进行复盘和候选整理，不启动专家推理或训练，也不发布规则。仅在理解和接受参考数据存在问题时显式使用 `--allow-incomplete`；其输出仍然需要参考审核和独立验证。可用 `--max-calls` 限制本次调用，其他参数和续跑约束以 `--help` 为准。

### P0 v2 历史修订

`facade_mcp.p0_v2_cli` 的 `prepare` / `run` 用于带父清单、SHA-256 和修订 ID 校验的候选修订流程；它绑定特定历史实验的输入与已验证前缀，不是通用的零起点训练入口。`prepare` 输出修订 ID；`run` 需提供同一父清单与哈希、模型配置、预期修订 ID 和显式 `--allow-api`。最终第七次恢复还依赖工作目录中的专用恢复脚本、私有中间文件及调用记录，这些材料不属于 Git 发布内容，不能仅凭上述命令重建最终候选。

发布可用于后续检测的 P0 前，需先审核标注、类别映射和候选规则，并在独立参考集上进行真实成对验证；逐条规则消融及四图后续实验需要按实际需求补全和确认。
