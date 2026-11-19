# CUBIT 论文模型接入

这里提供两个已下载 CUBIT checkpoint 的**实验性单图推理兼容代码**。它们不是作者原始 fork，尚未完成与作者实现的数值一致性或准确率验证，也不提供论文模型的训练代码。

- [YOLOv9-e 运行器、依赖及重建说明](yolov9e_runtime/README.md)
- [EfficientViT-M5 运行器、依赖及重建说明](efficientvit_runtime/README.md)
- [权重、作者参数、来源及校验记录](cubit-inseg-paper-2026/README.md)
- [专家注册示例](cubit-experts.example.json)，包括权重哈希、类别映射和关键实现文件哈希声明。

本机同时有忽略 Git 的 `cubit-experts.local.json`，作为已通过单图验证的配置。它不会自动被当前服务加载。设置 `FACADE_EXPERTS_CONFIG` 指向此配置并重新启动使用它的模型服务，才会注册 `cubit_yolov9e` 和 `cubit_efficientvit_m5`；上线或重启前应先确认其他正在运行的任务。`ready=true` 只表示所声明的文件和哈希通过检查，不意味着加载或准确率得到验证。

跨机器复制时，`*.pt`、`vendor/`、`.venv/`、`*.local.json` 及作者 `*.log` 被 Git 忽略，不会随 Git 自动传送；应根据需要单独安全传送权重、获取日志，按两个运行器 README 在目标机重建环境，再检查示例配置中的路径、Python 版本及源码哈希清单。示例中的安装路径针对 Python 3.10；其他 Python 版本需要改对应的 `implementation_files` 路径。备份的 Conda 链接或虚拟环境不能视为可直接迁移的完整环境。

现有代码一次最多接收 100 张图片，尚未实现第一张需求图的完整批量流程。项目当前会为分割结果保存原图尺寸 Mask；在剩余磁盘空间不足时不要直接全量运行。该文档不改变训练/验证/测试集划分。
