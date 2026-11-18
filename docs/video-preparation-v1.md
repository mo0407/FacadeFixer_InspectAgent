# 视频工作流准备与墙面范围复核

视频准备默认每秒 2 帧，近墙关键时段默认每秒 4 帧。MCP 每批最多 200 帧，最后合并原帧和墙面范围复核清单。

## 运行

在仓库目录使用项目 Python 环境：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m facade_mcp.cli --workspace "E:\ai\低空\workflow-evaluations\video-run" --allow-root "E:\ai\低空\video\building" video-prepare --video "E:\ai\低空\video\building\w4.mp4" --building-id "building-01" --sample-fps 2 --max-frames 2000
```

该命令会真实生成原尺寸 PNG，可能占用较多磁盘；`max-frames` 为明确抽帧预算，超预算在抽帧前失败，不静默少抽。当前逐批独立读取并校验源视频，长视频重复哈希有时间开销；没有自动断点恢复或缓存复用。

人工查看视频后，可追加 `--keyframe-range 12 18 --keyframe-range 42 47 --keyframe-fps 4`。时间单位为秒，区间左闭右开。相同解码帧去重，实际密度受视频原帧率限制。关键帧优先级为 2，普通帧为 1。该值决定处理顺序，且会随范围清单传递至检测报告数据。

## 两个新增 MCP 工具

| 工具 | 输入 | 输出 |
|---|---|---|
| `video_prepare_workflow` | 视频路径、建筑 ID、模态、抽样帧率、帧数预算 | 原帧清单、批次清单、待复核范围清单 |
| `video_apply_wall_scope` | 修改后的范围清单路径 | 原坐标墙面检测输入、原图引用、墙面 mask 快照、逐帧审核记录 |

抽帧使用原视频最近帧，记录实际解码帧号与时间。统一清单仍为 `facade-frames/v1`，可供已有拼接工具使用。全局工具轨迹写入编排任务的 `tool_calls.jsonl`。

### 完成范围复核

将生成的 `scope-review.json` 另存为新文件后编辑，保持 `frame_manifest`、`source_sha256` 和所有 `frame_id` 不变，不直接覆盖已注册 artifact。每帧填写：

```json
{
  "frame_id": "保留实际帧ID",
  "status": "included",
  "reviewer": "复核人员标识",
  "reason": "目标建筑墙面；已排除门窗、外楼梯与背景",
  "wall_mask": "E:/人工标注/当前帧-wall.png"
}
```

`included` 要求原尺寸、单通道、非空二值 mask，白色为检测墙面。`excluded` 不需要 mask，但必须记录原因和复核者。清单需要逐帧完整，不能遗漏、重复或保留 pending。路径须位于服务允许的输入根目录内。

工具保留完整原图尺寸，仅将范围外像素置黑。因此检测坐标仍可直接对应原图；输出附带独立原图引用。黑色边界可能产生检测伪影，后续仍需过滤边界候选。最终分割和报告证据应使用原图。工具验证数据结构和完整性，不宣称已经视觉验证人工 mask 正确，也不对复核者做身份认证。

## 当前边界

范围复核通过后，将 `video_apply_wall_scope` 返回的 `manifest.uri` 填到以下配置的 `wall_inputs`。使用同一个 `--workspace` 以解析产物引用。

```json
{
  "wall_inputs": "artifact:替换为实际产物标识",
  "project": {"name": "建筑外立面巡检"},
  "detector_profile": "yolo12m",
  "segmenter_profile": "mobile_sam",
  "prompt_mode": "bbox",
  "keyframe_inference_size": 1024,
  "threshold": 0.25,
  "max_frames": 100
}
```

```powershell
python -m facade_mcp.cli --workspace "E:\ai\低空\workflow-evaluations\video-run" video-detect --config video-detect.json
```

先用 `models` 检查本地模型权重与运行环境。`prompt_mode` 可用 `bbox`、`point`、`text`，具体由模型支持的提示方式决定。检测用墙面范围图，分割用原图。框内像素全部在墙面范围内才进入分割，跨边界和未知类别记录在 `candidate-audit.json`。最后通过 Report MCP 生成并校验 HTML 草稿。

近墙关键帧优先检测。YOLO 默认以 1024 推理尺寸处理关键帧，普通帧保留模型原默认值。置信度阈值相同。其他检测器可设 `keyframe_inference_size: null`，仅保留密集抽帧与优先处理。近墙时段目前由操作者标记，1024 推理的质量收益需用同一参考集实测。

这份草稿按帧列出候选，跨帧归并、立面总图及调优自动回跑仍待接入。运行时仍沿用现有模型类别，业务分类迁移另行验证。

调优控制器见 [评价与调优控制器](optimization-controller-v1.md)。接入时应缓存这一阶段产物；只有抽样参数或范围发生变化才重新准备，单纯改变分割模型不应重新抽帧。

已用 201 帧合成视频测试跨批次合并，用合成小视频验证 CLI 经真实 stdio MCP 的调用路径。合成视频只验证工程流程，不评价真实建筑识别效果。
