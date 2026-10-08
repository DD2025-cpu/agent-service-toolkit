# OilWind-Agent 手动验收指南

## 使用工具

推荐使用 VS Code 打开 `E:\F_WorkSpace\OilWind-Agent`。本项目主要通过终端启动 FastAPI 和 Streamlit，VS Code 的集成终端、JSON 查看和多文件搜索已经足够。已经熟悉 PyCharm 时也可以继续使用，运行结果没有差别。

不要把 `.env` 提交到 Git，也不要在截图、论文或聊天记录中展示其中的 API Key 和 `AUTH_SECRET`。

## 启动系统

打开两个 PowerShell 终端。

终端一：

```powershell
cd E:\F_WorkSpace\OilWind-Agent
uv sync --frozen
uv run python src\run_service.py
```

终端二：

```powershell
cd E:\F_WorkSpace\OilWind-Agent
uv run streamlit run src\streamlit_app.py
```

浏览器打开 `http://127.0.0.1:8501`。默认 Agent 应显示为 `oilwind-agent`，默认模型应为 `deepseek-v4-flash`。

如果提示端口被占用，通常表示服务已经启动。先访问页面，不要重复启动多个实例。

## 执行测试

测试数据位于 `data/manual_agent_test_cases.json`，全部为虚构数据。按照 `T01` 至 `T08` 依次复制 `prompt` 到聊天框。除 `T06` 外，每项建议新建一个聊天，避免上一项参数影响下一项。

`T06` 必须在同一个聊天中完成：先发送 `prompt`，确认 Agent 暂停并追问；再发送 `follow_up_prompt`，确认 Agent 使用前一轮的50MW容量继续计算。

每项检查三个层面：

1. Agent 是否选择了 `expected_tools` 中的业务工具。
2. 数值名称和单位是否正确，尤其要区分“公式计算IRR”和“模型预测IRR”。
3. 缺参、非法单位或工具失败时，是否停止计算且没有编造数值。

Streamlit 默认会显示工具调用过程。展开工具消息可以核对实际工具名称和结构化结果。

## 成功标志

- T01 返回公式IRR、NPV、两种回收期和LCOE。
- T02 返回模型版本和人民币口径代理数据集说明。
- T03 返回五个SHAP特征贡献，不作因果结论。
- T04 返回悲观、基准和乐观三类模拟结果。
- T05 保留两个项目标识并按公式IRR排序。
- T06 第一轮追问，补参后继续计算。
- T07 停止计算并指出单位或数值错误。
- T08 在 `outputs/reports` 下生成Markdown报告。

如果八项均满足，可记录“主Agent手动验收通过”。如果失败，请保存用户输入、工具名称、工具返回的 `status` 和最终回答，不要只记录最终页面截图。

## 自动复核

本地确定性测试：

```powershell
uv run pytest tests\oilwind_agent -q
```

真实 DeepSeek 验收会产生API费用：

```powershell
uv run python scripts\accept_oilwind_agent.py
```
