# LLMOps API

一个采用 Flask 的 LLMOps 后端服务骨架。项目按 Web 入口、配置、业务层、LLM 核心能力和可复用扩展包分层，便于后续接入模型供应商、向量数据库、OAuth 与定时任务。

## 快速开始

在 `llmops-api` 目录中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python App.py
```

服务默认运行在 `http://127.0.0.1:5000`。可通过以下接口确认服务状态：

```bash
curl http://127.0.0.1:5000/api/v1/health
```

预期响应：

```json
{"status":"ok"}
```

## 项目结构

```text
llmops-api/
├── App.py                  # 服务启动入口
├── app/                    # Flask 应用工厂与 HTTP 入口
├── config/                 # 默认配置和环境配置选择
├── internal/               # 应用内部实现
│   ├── core/               # LLM 核心能力
│   │   ├── agent/          # Agent 编排
│   │   ├── chain/          # 调用链编排
│   │   ├── prompt/         # 提示词管理
│   │   ├── model_runtime/  # 模型供应商与运行时适配
│   │   ├── moderation/     # 内容安全审核
│   │   ├── tool/           # 工具调用
│   │   └── vector_store/   # 向量数据库适配
│   ├── exception/          # 通用业务异常
│   ├── extension/          # 数据库等第三方扩展初始化
│   ├── handler/            # 请求处理器
│   ├── middleware/         # 认证、日志等中间件
│   ├── migration/          # 数据库迁移脚本
│   ├── model/              # 数据库模型
│   ├── router/             # API 路由
│   ├── schedule/           # 定时任务
│   ├── schema/             # 请求与响应的数据结构
│   ├── server/             # 应用服务构建层
│   ├── service/            # 业务服务层
│   └── task/               # 即时和延迟任务
├── pkg/                    # 可复用的扩展包（如 OAuth）
├── storage/                # 本地运行时文件，不存放敏感数据
├── test/                   # 自动化测试
├── .env.example            # 环境变量模板
└── requirements.txt        # Python 依赖
```

## 开发约定

- 真实密钥只写入本地 `.env` 或部署环境变量，绝不提交到仓库。
- 新增接口时，在 `internal/router/` 注册路由；业务规则放在 `internal/service/`。
- 模型、Embedding、RAG、Agent 等能力放入 `internal/core/` 对应模块，避免与 HTTP 层耦合。
- 数据库迁移脚本生成在 `internal/migration/versions/`，应随代码提交。
