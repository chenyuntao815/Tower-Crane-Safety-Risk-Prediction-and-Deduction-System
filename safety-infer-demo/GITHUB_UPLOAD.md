# GitHub 上传与接收者启动说明

## 包内内容

`safety-infer-demo/` 包含全部推演与预测源码、测试、两套三维页面及本地 Three.js、界面样式和主题、知识图谱 CSV、示例、`inputdata.csv`、`C_L_A.h5`、依赖清单、启动脚本、文档、`.gitignore` 和空白 `.env.example`。

`PACKAGE_MANIFEST.json` 记录每个源文件的大小和 SHA-256。真实 `.env`、Streamlit secrets、虚拟环境、缓存、编辑器设置、生成图谱 `graph.html`/`lib/`、旧预测输出和本地启动笔记均不在包内，后续也不要补上传。

## 上传

1. 解压 `safety-infer-demo-github.zip`，在自己的 GitHub 账户中新建仓库。
2. 将解压后 `safety-infer-demo` 文件夹**内部的文件和子目录**上传到仓库根目录，使 README.md 直接显示在仓库首页。
3. 确认 `.gitignore`、`.env.example` 这两个以点开头的文件也已上传。
4. ZIP 和外部校验文件用于传递整个项目，无需再作为源码提交；需要时可另外作为下载附件分享。

本次仅准备本地文件，不会创建远程仓库或推送代码。

## 最小启动：手动运行推演与三维演示

使用 Python 3.11，在项目根目录执行；无需数据库、传感器、TensorFlow 或大模型密钥：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-operations.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 `http://localhost:8000/twin/`，原工序页面位于 `/twin/legacy.html`。不要直接双击 HTML 打开，页面需要本地 API。

## 其他模块

- 综合页面：同一环境安装 `requirements.txt`，另开终端使用该环境执行 `python -m streamlit run streamlit_app.py`，访问端口 8501。
- 在线图谱：需要自己的 Neo4j 数据库；新用户须按文件标签、`id` 属性和关系类型导入 `Knowledge Graph/`，CSV 不会自动建立数据库；已有正确图谱时无需重复导入。连接信息按 README 在后端终端设置。
- 历史预测：建议独立 Python 环境安装 `requirements-prediction.txt`，执行 `python -m uvicorn prediction_api:app --host 127.0.0.1 --port 8001`；已有模型随包提供，不要为启动服务先执行原训练脚本。研究限制见 [DATA_MODEL_NOTES.md](DATA_MODEL_NOTES.md)。
- DeepSeek：复制 `.env.example` 为 `.env`，仅在本机填写密钥；`.env` 已被忽略。

完整流程见 [README.md](README.md)。本次在现有环境离线检查，没有在新机器重新联网安装依赖，也没有验证真实数据库或付费模型服务。

## 测试与重新打包

开发测试依赖位于 `requirements-dev.txt`；其中的 HTTPX 版本约束适配主依赖中较早的 FastAPI/Starlette TestClient。训练脚本的旧版 Keras 与训练专用依赖不属于服务启动依赖。

```powershell
python -m unittest -v test_operations test_operation_api test_digital_twin test_tower_graph test_llm_decision
node --experimental-vm-modules --test test_operation_inputs.mjs
python scripts/build_github_package.py
```

打包脚本仅使用 Python 标准库，按白名单生成上传目录、ZIP、SHA-256 和文件清单，并检查常见凭据格式；不会输出疑似密钥内容。
