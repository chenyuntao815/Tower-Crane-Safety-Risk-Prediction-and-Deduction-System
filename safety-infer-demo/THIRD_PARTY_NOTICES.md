# 第三方资源说明

`digital_twin/vendor/package/` 包含 Three.js r180 的必要文件：`build/three.core.mjs`、`build/three.module.mjs`、`examples/jsm/controls/OrbitControls.mjs`。其 MIT 许可及版权声明完整保留于 [digital_twin/vendor/package/LICENSE](digital_twin/vendor/package/LICENSE)，代码文件原有声明亦予保留。

FastAPI、Streamlit、NumPy、Neo4j Python Driver、PyVis、OpenAI SDK、TensorFlow/Keras 等通过 requirements 文件安装，适用各项目自身许可。PyVis 运行时可能生成 `graph.html` 和 `lib/`；这些可再生文件未随本分享包分发。

本分享包没有代替作者选择或添加项目自身的统一开源许可证；第三方许可不等同于项目代码、图谱、数据及模型的统一授权。作者可在确定共享范围后另行添加适当许可及数据来源说明。
