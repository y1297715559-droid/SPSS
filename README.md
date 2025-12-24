# 问卷解析 + 维度/关系约束 + SPSS数据生成器（本地网页）

## 你能做什么
- 粘贴问卷文本：自动识别题号/题干/选项
- 配置维度：题号归属、反向题
- 关系约束：性别差异、维度相关、A→C→B 中介
- 一键生成：CSV、SPSS语法（.sps），以及可选的 .sav（若安装 pyreadstat）

## 运行方式（本地）
1. 安装 Python 3.10+  
2. 在此文件夹运行：
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. 浏览器会自动打开本地地址（通常是 http://localhost:8501 ）

## 部署成“网页链接”（你自己点几下就能完成）
- Streamlit Community Cloud：把本文件夹上传到 GitHub，然后在 Streamlit Cloud 里选择仓库部署
- Render / Railway 也可以：同理

> 注意：这里生成的是“模拟数据”，用于测试/演示/方法仿真，不可替代真实问卷数据用于研究结论。
