# 塔吊安全风险预测与推演：安装与使用

请按 README.md 的命令安装依赖、配置已更新的 Neo4j，并启动图谱 API、独立 CNN-LSTM-AT 预测 API 和 Streamlit 页面。

## 图谱更新

Knowledge Graph/ 中的 5 个实体 CSV 和 4 个关系 CSV 对应当前塔吊场景。新用户需先按文件对应标签、id 属性和关系类型导入自己的 Neo4j；已有正确图谱时无需重复导入，也无需设置 scenario 属性。项目不会自动创建或改写数据库。关系 HAS_INDICATOR 的 direction 使用 positive/negative，阈值按关系属性读取。新版运行推演和原工序三维演示可使用本地 CSV，无需在线数据库。

## 推演计算

每个输入指标按关系中的三个阈值转换为 0–1 的危险度。达到预警阈值时危险度为 0.6，达到危险阈值时为 1。若只提供部分关联指标，按已提供指标权重归一。结合因素初始权重与适用阶段系数形成动态权重，再按 CAUSES、LEADS_TO 的触发概率、影响度和置信度计算风险与事故链。

这些是图谱参数下的情景估计，供方案比选与风险排查使用；现场处置应依据设备说明书、专项方案和适用管理要求。

## 常见问题

- 因素列表为空：确认 Neo4j 已启动、连接信息及数据库名正确，且图谱节点带有代码查询使用的标签和 id 属性。
- 因素无适用阶段或指标：检查 Neo4j 中的 ACTIVE_IN 和 HAS_INDICATOR 关系。
- AI 决策失败：确认启动 API 服务的进程能读取 DEEPSEEK_API_KEY，已安装 requirements.txt 中的 openai SDK，并在更新配置后重启服务。页面会显示实际错误，不再静默返回预设建议。
- CSV 预测连接失败：确认 prediction_api 服务在 8001 端口运行，并安装 requirements-prediction.txt。
