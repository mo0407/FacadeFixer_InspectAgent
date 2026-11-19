# 图像主动巡检工作流

本版输入是一张静态图片，不进行视频抽帧、跨帧跟踪或外立面拼接。它实现图中的“视觉锚定 → 主动局部检查 → 最终核验”，原视频模块保留在独立副本中，便于后续复用。

## 1. 运行边界

源代码保留在本机 `E:\ai\低空\FacadeFixer_VIDEO`，图像改造位于 `E:\ai\低空\FacadeFixer_IMAGE`，对应远程项目 FacadeFixer_InspectAgent。复制来源提交及原始文件的 SHA-256 记录在本机保留的 `backup-manifest.json`，该溯源文件不随仓库发布。副本不携带原仓库的 Git 历史、忽略文件、模型权重、运行结果和密钥；需要通过环境变量连接现有本地模型环境。

同一张图的最多两次局部检查属于一次运行。一次运行中模型身份、专家权重哈希、P0 规则快照与检查预算冻结。`image-inspect` 默认加载已验证发布的最新版规则，显式 `--playbook` 可覆盖。复盘和规则整理发生在离线阶段；提案需经过独立参考集成对验证与正向改善检验才可发布，不能悄悄改变当前轮的上下文。完整用法见 [图像检查与离线规则闭环](complete-image-workflow.md)。

## 2. 每个环节的输入与输出

| 环节 | 输入 | 输出 | 谁决定 |
| --- | --- | --- | --- |
| 输入快照 | 原始图片、配置、可选墙面范围 mask、P0 | 可追溯的图片与规则快照 | Python 校验路径、文件和配置 |
| 全图专家检测 | 原图、专家模型列表、阈值、推理尺寸 | 全图候选框、类别、分数及来源 | 专家模型 |
| 视觉锚定 | 原图、候选框、网格与上下文区域 | 带编号的全图预览和原图坐标索引 | Python 生成 |
| 检查决策 | 全图预览、锚点 ID、既有证据、P0、剩余预算 | 检查哪个锚点、上下文扩展倍数、允许的专家；或结束 | Generator VLM |
| 局部执行 | 合法动作、原图、锚点坐标 | 原图裁剪、局部专家框、裁剪偏移与执行记录 | Python 校验并调用 MCP |
| 坐标回映与去重 | 全图及局部候选、裁剪偏移 | 原图坐标下的合并候选与来源 | Python |
| 最终复核 | 合并候选、预览和已有局部证据、同一份 P0 | 每个候选的 `keep` / `reject` / `defer` 及理由 | Generator VLM |
| 结果归档 | 原图证据、候选、判断、模型参数、检查历史 | JSON 结果、可阅读报告及审计材料 | Python |

VLM 选择工具动作，Python 执行框架控制工具许可和预算。框坐标来自专家输出及确定性的几何映射，最终复核不允许凭空创造候选 ID。

## 3. 锚点代表什么

| 锚点 | 含义 | 用途 |
| --- | --- | --- |
| Box | 分数达到 `high_threshold` 的候选 | 快速定位较强的专家响应 |
| Ghost | 达到 `low_threshold`、未达到高阈值的候选 | 复核容易漏掉或误判的弱响应 |
| Env | 显式提供的环境或上下文区域 | 为局部理解提供可追溯的区域提示 |
| Grid | 与候选无关的网格搜索区域 | 让 VLM 有机会主动检查专家没有出框的位置 |

低分候选只是检查线索，不是已确认缺陷。网格也不代表已经发现缺陷。局部检查从原图裁剪，扩展倍数扩大上下文范围，不是把缩略图放大后当成新证据。

## 4. 请求参数

示例见 `examples/image-inspection.example.json`。输入为单帧 PNG 或 JPEG。图片路径相对于配置文件所在目录解析，因此示例中的 `images/facade.jpg` 指 `examples/images/facade.jpg`。也可以填写绝对路径；CLI 的 `--allow-root` 用于允许额外输入目录。

| 字段 | 示例值 | 作用 |
| --- | --- | --- |
| `image_path` | `images/facade.jpg` | 待检查的原始图片 |
| `detector_profiles` | `["yolo11m", "yolo12m"]` | 全图专家及局部检查可选模型白名单 |
| `low_threshold` | `0.1` | 全图/局部专家候选最低分数 |
| `high_threshold` | `0.5` | 区分 Box 与 Ghost 的阈值 |
| `max_inspections` | `2` | 单张图局部检查上限，最多为 2，可提前结束 |
| `grid_rows` / `grid_cols` | `3` / `3` | 无候选区域的全图搜索网格 |
| `merge_iou` | `0.5` | 同类重复框的合并阈值 |
| `inference_size` | `960` | 支持该参数的专家模型推理尺寸 |
| `scope_mask_path` | `images/facade-wall.png` | 可选，原图等尺寸的二值墙面范围 mask |
| `context_regions` | `[]` | 可选的 Env 区域列表；每项包括唯一 `id`、`bbox_xyxy` 和 `description` |
| `context_mode` | `auto` | 自动从范围 mask 连通域或 VLM 场景判断建立 Env；另可选 `manual` / `none` |
| `expert_class_filters` | `{}` | 按 profile 收窄允许类别，不能扩大专家已声明能力 |
| `max_candidates` | `200` | 候选观察数量上限；超出时明确失败，不静默丢弃候选 |

已提供 YOLOv9 分割和 Ultralytics 自定义专家的服务器注册适配器，见 `examples/experts.registry.example.json`。图中的 CUBIT 数据与权重并不随代码提供，也未在本次开发中验收。模型名称须与实际注册 profile 一致；`facade_models_capabilities` 读取类别、资格和真实权重哈希，不加载模型。`python -m facade_mcp.cli models` 另可探测旧模型环境的依赖配置。

`inference_size` 必须为 32 的倍数，当前仅向支持该选项的专家传入。`context_regions` 的 ID 采用 `env-` 前缀，例如 `{"id":"env-wall-top","bbox_xyxy":[100,100,700,400],"description":"目标墙面顶部"}`；坐标必须在原图范围内。`auto` 模式在有范围 mask 时按连通域建立墙面 Env；否则把图片交给 VLM 区分墙面、门窗、外楼梯、背景与上下文。自动 Env 仅为上下文线索，不是像素 mask 或人工真值。

## 5. 坐标、范围与去重

框使用原图像素的 `xyxy` 约定。局部框通过加上裁剪左上角偏移回映到原图，并校验边界。预览图可以缩小显示，但不会替代几何计算中的原始坐标。坐标对应文件存储的栅格方向，程序不自动套用 EXIF 旋转；若需旋转，请先生成方向统一的图片，再绘制对应范围 mask。

同类且重叠达到阈值的候选可归入一个复核项。保留来源记录以区分“多个专家或局部检查都看到同一处”和“两个不同物理缺陷”。单图去重不等于跨图实体关联。

业务范围是目标建筑可见墙面，排除门、窗、外楼梯和背景建筑。建议提供经过复核、与原图等尺寸的灰度二值 PNG 范围 mask：255 为允许检查的墙面，0 为排除区域，且至少包含一部分墙面。未提供 mask 时，报告应说明范围未经人工复核；VLM 的文字判断不能替代可靠的几何范围。

图像工作流在输出边界采用 C、S、T、R、D、V、L、E、B 九类业务定义：将旧 M 归入 C、P 归入 S，旧 L 中明确标为白华/泛碱的记录归入 E，并保留源模型信息。这不改变旧模型的训练类别或 checkpoint。B 类虽然可在业务结果中表达，实际能否识别仍取决于所选权重。未知类型保留为 `unknown`，不能因最终 VLM 判断而自动变成已确认类型。详见 `evaluation-standard-v1.md`。

最终复核另记录 `scope_decision`：`wall` 表示目标建筑墙面，`excluded` 表示门窗、外楼梯或背景等排除范围，`uncertain` 表示不确定。范围 mask 的覆盖状态、VLM 的范围判断和最终保留状态分别记录。未知类别、部分/完全落在范围外，或视觉范围判断不明确的候选不直接升级为保留项。

## 6. VLM 与规则配置

`examples/image-models.mock.json` 满足原有网关的三个角色结构，但图像主动检查本轮仅调用 `generator`。其中 `supports_images=true` 是图片输入的必要条件；这只是本地能力声明，不代表已经验证某个远程服务实际支持视觉输入。

远程连接沿用 `chat_completions` 或 `responses`。设置模型名、`endpoint_env`、`api_key_env`，通过环境变量提供完整 API 端点及密钥。配置中的 `api_enabled` 和 CLI 的 `--allow-api` 都开启后才允许远程调用。图片证据、候选和 P0 会随请求发往配置的服务商；审计记录不写入图片 base64 或密钥。

可复制 `examples/image-models.remote.example.json`，设置 `FACADE_IMAGE_ENDPOINT` 和 `FACADE_IMAGE_API_KEY`。示例将三个角色映射至同一个 `gpt-6.1-sol` Responses 连接，使用 `reasoning_effort="medium"`；`expected_model` 校验服务返回的模型身份，可根据供应商真实响应调整。默认 `api_enabled=false`，不会因填写端点而自动允许外发。图像在线阶段使用 Generator，离线规则阶段使用 Reflector 与 Curator。配置声明不证明视觉可用，需显式执行 `vision-check`；本次未进行付费远端视觉验证。

`examples/image-playbook.json` 是可读示例规则，具有稳定 ID 和版本。示例中的经验计数为 0，不把人工编写的初始规则包装成已经验证有效的沉淀经验。将规则文件作为 `--playbook` 传入后，本轮保存快照；下轮采用新规则需显式更换版本。

## 7. 运行与验证

先执行合成演示：

```powershell
facade-image --workspace workspace image-demo --output-dir examples/generated_image_demo
```

再运行本地真实专家与 mock VLM：

```powershell
facade-image --workspace workspace image-inspect --config examples/image-inspection.example.json --model-config examples/image-models.mock.json --playbook examples/image-playbook.json
```

mock VLM 不会连接 API，但 `image-inspect` 仍会运行本地专家。远程 VLM 应改用自己的模型配置，并加入 `--allow-api`。

也可以以 stdio MCP 接入：

```powershell
python -m facade_mcp.image_server
```

工具 `facade_image_inspect` 的入参：

```json
{
  "request": {
    "image_path": "E:/inputs/facade.jpg",
    "detector_profiles": ["yolo11m", "yolo12m"],
    "max_inspections": 2
  },
  "model_config_path": "E:/configs/image-models.json",
  "playbook_path": "E:/configs/image-playbook.json",
  "allow_api": false
}
```

MCP 调用中的路径应填允许输入目录下的绝对路径或有效 artifact URI。此工具是图像流程的高层入口，专家检测仍由 model MCP 执行，不需要新建网页或 HTTP 端口。

任务目录保存以下主要产物：

| 文件 | 内容 |
| --- | --- |
| `report.html` | 可读的检查报告，每个候选包括定位图和局部图 |
| `result.json` | 原图坐标下的候选、合并来源及最终判断 |
| `anchors.json` | 全图锚点及坐标索引 |
| `expert_capabilities.json` | 本轮服务器专家能力与权重身份快照 |
| `scene_context.json` | 自动 Env 上下文；不代表人工范围或像素 mask |
| `observations.json` | 各模型/局部检查的原始候选观察 |
| `trajectory.json` / `inspections.json` | 决策过程及局部执行记录 |
| `playbook.json` | 本轮冻结的 P0 规则快照 |
| `model-config.json` | 本轮连接与角色配置，不含环境变量密钥值 |
| `model_trace.json` | 模型、调用状态、用量与图片证据摘要 |
| `final-review-NNN.json` | 每一批候选的结构化最终复核响应 |
| `input.*` / `source.png` | 原始文件快照和供检测使用的统一栅格图 |

最终复核每批最多 4 个候选，附带全图背景及局部证据，单次最多 7 张图。局部图保留原像素供查看；若为适配模型输入而将超过 2048 边长的局部证据缩图，该候选会被强制标为 `defer`，避免把缩图复核当作细节已确认。

一次验收应分别检查：输入证据是否完整、范围是否正确、坐标回映是否正确、同类去重是否误合并、最终判断是否有图像依据、预算是否遵守，以及完整轨迹能否复现。没有独立人工真值时，保留这些程序与证据检查结果，不计算准确率或召回率。

## 8. 当前阶段的限制

- 主流程以单张图的框级核验为核心，注册分割专家可提供原尺寸 mask 证据；交互分割和指定正式报告模板仍通过原有独立模块使用。
- 合成演示验证实现链路，不证明真实场景的召回或误报改善。
- 两次局部检查是明确的预算上限，不保证覆盖所有可疑区域；证据不足时输出待定。
- 本轮不训练模型、不在线更新 P0。离线验证发布要求独立人工参考、真实成对重放、模型/权重身份一致、达到门槛且有正向改善；mock、模拟 HTTP 或注入执行器不准晋升正式 P0。
