# AI 代码代理说明（仓库专用）

以下说明旨在让 AI 编程代理（Copilot / 自动化脚本）快速上手此仓库。内容基于对仓库中可发现的文件与约定的分析，仅包含可验证的、可操作的信息。

## 一眼看懂（仓库概况）
- **仓库类型**：以 Markdown 文档为主的学习资料库（知识库/笔记），目录划分为主题文件夹，如 `ai/`、`alg/`、`edge/`、`kaggle/`、`linux/` 等。
- **可执行代码**：极少量示例脚本位于 `kaggle/`（例如 `kaggle/titanic_competition_w_tensorflow_decision_forests.py`），多数为 Colab 导出内容，包含 Colab 特有依赖与 shell magic，不可直接在本地 Windows/PowerShell 环境直接运行。
- **构建/测试**：仓库中没有发现构建工具、CI 配置或测试框架（未发现 `package.json`、`requirements.txt`、`.github/workflows` 等）。因此不要假设存在自动化构建或测试流程。

## 主要约定和注意点（对代理很重要）
- 文档以 Markdown 为主，编辑请直接修改对应 `.md` 文件并保留相对链接（示例：`README.md` 中使用 `./ai/index.md` 的相对引用）。
- 有些文件名含空格（例如 `prj/Image Classification.md`），修改或重命名时请保持路径一致以免破坏链接。
- `kaggle` 下的 Python 脚本通常含有：
  - Colab 专用导入：`from google.colab import drive`、`kagglehub` 等。
  - Shell/魔法命令：`sudo mkdir -p`、`!head /kaggle/working/submission.csv` 等。
  - 建议：若需运行或修改这些脚本，先移除/替换 Colab 专有部分并添加清晰的运行说明（例如需要的 Python 包清单）。
- 不要擅自引入项目级构建或元数据（例如添加 Jekyll/Hexo 配置、YAML front-matter），除非用户同意将仓库转为网站形式。

## 可参考的关键文件（示例）
- 仓库根说明：`README.md` —— 概述与目录结构。
- AI 技术索引：`ai/index.md` —— 主题索引（导航入口）。
- Edge 比较说明：`edge/compare.md` —— 当前正在编辑的文件，示例写作风格。
- Kaggle 示例：`kaggle/titanic_competition_w_tensorflow_decision_forests.py` —— 包含 Colab 导出痕迹，示例如何处理真实代码片段。

## 对自动化修改的具体建议（代理行为规范）
- 修改文档时：只修改目标 `.md` 的内容，不移动或重命名文件，除非明确要求并同时更新所有受影响的相对链接。
- 若需要运行或改造 `kaggle` 下脚本：
  - 在 PR 中同时提交一个 `README` 或注释，说明本地复现步骤与依赖（列出 pip 包和 Python 版本）。
  - 用注释或条件检查隔离 Colab 专属代码（示例：`try: from google.colab import drive except ImportError: pass`）。
- 保持文件编码为 UTF-8，保留原有中文内容与术语表述，避免自动翻译或改写专业术语。

## 当遇到不确定的改动
- 如果改动可能影响大量相对链接或目录结构，提交变更前在 PR 描述中列出受影响的文件清单。
- 如果需要新增项目级配置（如 `requirements.txt`、CI），先在 Issue 中描述计划并征求仓库维护者同意。

---
请检查以上说明是否覆盖了你希望 AI 代理遵守的重点（例如是否允许批量重命名、是否需要自动生成依赖文件等）。收到确认后我可以根据反馈迭代此文件。
