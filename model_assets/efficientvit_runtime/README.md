# EfficientViT-M5 CUBIT 推理兼容环境

此目录仅用于载入已下载的 CUBIT EfficientViT-M5 checkpoint，不提供训练。代码与论文作者的完整训练 fork 不同，兼容性依据权重中序列化的层结构和单图运行验证；不能宣称与作者原实现数值一致或已通过完整精度评测。

## 来源

- `efficientvit_compat.py` 改编自 Microsoft Cream 的 [EfficientViT downstream 实现](https://github.com/microsoft/Cream/blob/74fb8c5ebfc3a3dbfb397df27d74b2e4322d68a0/EfficientViT/downstream/efficientvit.py)，MIT 许可证见 [LICENSE.microsoft.txt](LICENSE.microsoft.txt)。去除 MMDetection/MMCV 注册导入和定制 `EfficientViT.train`，将三尺度输出转为列表；文件仅用于反序列化并执行已有 checkpoint，不支持重新初始化或训练模型。
- `launch.py` 根据已下载 checkpoint 的层索引将三个骨干输出映射到 YOLO 层位置 2、3、4；基于该 checkpoint 的结构验证，并非论文作者原始路由代码。
- `python-worker` 用该环境启动项目的 `registered_expert_worker.py`。它不会改变全局 Ultralytics 安装。
- 作者权重、原训练参数和许可证信息见[模型资料](../cubit-inseg-paper-2026/README.md)。

## 创建环境

已测试的基础环境使用 Python 3.10，并包含 CUDA PyTorch 2.12.0+cu130、timm 1.0.27、NumPy、Pillow 和 OpenCV。隔离 venv 继承这些大体积依赖，仅额外安装 Ultralytics 8.3.9 和 dill 0.3.9：

```sh
FACADE_MODEL_BASE_PYTHON=/path/to/base/python ./model_assets/efficientvit_runtime/setup.sh
```

脚本会在本目录新建 `.venv`，遇到已存在的环境会停止，不会覆盖。基底环境不是自包含的；换机器时需先准备相应依赖，再设置 `FACADE_MODEL_BASE_PYTHON`，并确认 CUDA、依赖版本和模型输出。需要 `torch.serialization.safe_globals` 和 `get_unsafe_globals_in_checkpoint`。以上版本组合在 2026-11-20 本机通过测试，不代表任意新版依赖都兼容。

## 权重和安全边界

启动器只接受项目注册专家 worker 和 SHA-256 为 `cfbc562cdca0766d06a2c7d328977ceda0a78f25b38bca0ccc67e11baeb6be00` 的对应本地 checkpoint。加载使用 `weights_only=True`、固定类白名单和只允许恢复 `set` 的 dill 类型替代函数，不启用普通 unrestricted pickle 加载。项目仍应只从可信来源取得权重，验证文件哈希并保护运行目录。

将 `python-worker` 配为注册专家的 `python_path`，`backend=ultralytics`、`task=segment`，类别 `crack → C`、`spalling → S`。`implementation_files` 至少应包含 `python-worker`、`launch.py`、`efficientvit_compat.py`；外部基底环境及包版本也需记录。专家的 ready 状态只检查文件，不等于真实推理通过。

## 单图验证记录

2026-11-20，本机 RTX 3090、网络隔离的 bwrap 环境，通过项目 `ModelService.detect` 对一张 4500×3000 原图以输入尺寸 640、阈值 0.25 执行：`status=completed`，输出 20 个缺陷和 20 个原图尺寸 Mask，0 个未映射、0 条警告，用时约 8.27 秒。该测试不使用标签，不证明缺陷准确率或与作者结果逐像素一致；本地 203 张测试图可能参与作者的原验证集模型选择。
