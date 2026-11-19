# CUBIT-InSeg 论文模型资料

下载日期：2026-11-20。论文：[From instance segmentation to physical quantification: High-resolution UAV-based dataset for façade defect assessment](https://doi.org/10.1016/j.autcon.2026.106980)。作者[数据仓库](https://github.com/BenyunZhao/CUBIT-InSeg)指向[基准资源](https://benyunzhao.github.io/CUBIT-Dataset/)，其中的共享目录提供下列训练结果和参数。

| 本地文件 | 作者来源 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `yolov9e-seg-cubit-best.pt` | [YOLOv9-e best.pt](https://drive.google.com/file/d/1WjQZ81dusnEdTk7KUmpmAi8VqKHqny7u/view) | 121,949,610 字节 | `0353f35c1f136fa4c5a3b1411341355758c6971fbbcb01198cea4b9abc518603` |
| `efficientvit-m5-seg-cubit-best.pt` | [EfficientViT-M5 best.pt](https://drive.google.com/file/d/1Lwb9BrWy7ORN7yPoEarIT_04LLnS7Ppr/view) | 98,531,202 字节 | `cfbc562cdca0766d06a2c7d328977ceda0a78f25b38bca0ccc67e11baeb6be00` |
| `yolov9e-args.yaml` | [YOLOv9-e 训练参数](https://drive.google.com/file/d/1bS072-yJ5Vvkx13e4kEt1IY998WuJfhU/view) | 1,601 字节 | `5580c8f1702667ccf9a1a12d75ad2e61146129b73875d392aacbc72d168f10a1` |
| `efficientvit-m5-args.yaml` | [EfficientViT-M5 训练参数](https://drive.google.com/file/d/1KSwb9wtTuNmUVHFFFM_aRIP68_M1bG9C/view) | 1,701 字节 | `2f3cd0d572add511539db0dfdc8684ebe73da07b3bec0b1e4daf1dd4cba4d810` |

`efficientvit-m5-cubit-test.log` 和 `yolov9e-cubit-test.log` 是作者公开的 [EfficientViT-M5](https://drive.google.com/file/d/1YlVqKEijicu-Bu0sWqCYV1caJfUT2r2l/view) 和 [YOLOv9-e](https://drive.google.com/file/d/1VgzgWf5ghE3c6YJO9Cu-XBrmq5Vot7vs/view) 测试日志，并非本项目测试结果。两份作者训练参数都写明实例分割、300 轮、图像尺寸 640、`pretrained: null`。论文表述其基准实验从头训练；这些 `best.pt` 是训练结果，不是通用初始化权重。

完整 ZIP 结构与 CRC 已检查，SHA-256 是本次本机计算值，未与作者发布的校验和比对。安全静态检查 checkpoint 内容可见 `crack`、`spalling` 类名。2026-11-20 已使用隔离兼容运行器，通过项目 `ModelService.detect` 对一张 4500×3000 原图完成两个模型的 GPU 前向：每个模型均返回 17 个候选，类别映射成功，且由项目校验原图尺寸 Mask；阈值 0.6，输入尺寸参数 640。此结果仅证明单图接口可运行，不是准确率评估，也未证明兼容实现与论文作者实现数值一致。

两个 checkpoint 含作者自定义 Ultralytics 模块：YOLOv9-e 引用 `ultralytics.nn.extra_modules.block`，EfficientViT-M5 引用 `ultralytics.nn.backbone.efficientViT`。论文数据仓库及上述模型目录中未找到完整作者实现，本项目分别使用实验性独立兼容层，见 `../yolov9e_runtime`、`../efficientvit_runtime`。两者使用受限的 PyTorch weights-only 方式加载；模型源码、运行库及原作者实现仍须对照验证。PyTorch checkpoint 可能包含可执行的 pickle 数据，应确认来源并谨慎加载。

项目本地 203 张测试图来自 CUBIT 原验证集；作者训练参数显示启用验证集。作者 `best.pt` 可能用过这些图片选择模型，未核实前不得把此测试集称为对作者 checkpoint 完全独立。
