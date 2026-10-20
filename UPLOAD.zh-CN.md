# 上传到个人 GitHub

此压缩包包含代码、配置、9张论文图、结果表、完整D2合成数据集和说明文档。**D1数据集及其训练、评价、OOF和绘图缓存均不在包内**；模型权重、原始冻结副本和机器路径清单也不在包内。

## 使用 GitHub Desktop

1. 解压 ZIP，得到 `IF-BridgeThermal` 文件夹。部分源码的目录较深，建议选择较短且不与现有工作目录重名的解压位置。
2. 在 GitHub Desktop 中选择 **File → Add local repository**，选择解压后的文件夹。若提示它还不是 Git 仓库，点击 **create a repository**，在该目录创建仓库。
3. 查看 Changes，填写提交说明，例如 `Add manuscript code and figures`，然后提交。若创建仓库时已生成初始提交，则不需要重复提交。
4. 点击 **Publish repository**，选择你的个人账户和仓库名称，并确认是否勾选 **Keep this code private**，再发布。

本项目建议使用独立仓库。

若确实要加入已有仓库，可以把解压后的项目文件夹复制到该仓库的本地目录，再通过 GitHub Desktop 提交并推送。

此包包含完整合成数据，建议按以上步骤解压后通过 GitHub Desktop 提交并发布。GitHub 不会自动将 ZIP 展开为代码目录。

## 使用代码前

- 从 `README.zh-CN.md` 和 `docs/EXPERIMENTS.md` 查找实验入口。
- 图件成品可以直接使用；重新绘图需要另行提供 `docs/DATA.md` 中列出的缓存。
- 训练通过 `tools/run_experiment.py` 启动；路径、GPU和D1私有参数按 `docs/RUNTIME.md` 配置。私有配置文件不要加入版本控制。
- 作者新增代码未指定开源许可证；已保留第三方代码的原许可证。

GitHub 官方操作说明：
https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-a-repository-from-your-local-computer-to-github-desktop
