# 🔍 Intelligent Search Engine

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/tylerelyt/test_bed)

一个基于机器学习的智能搜索引擎系统，支持CTR预测、实时排序优化和MLOps流水线。

## 🌟 特性

### 核心功能
- **🔍 智能检索**: 基于TF-IDF的倒排索引，支持中文分词
- **🧠 CTR预测**: Wide & Deep深度学习模型预测点击率，优化搜索排序
- **⚡ 实时优化**: 实时收集用户行为数据，动态调整排序策略
- **📊 数据分析**: 完整的用户行为分析和搜索效果统计

### 技术特性
- **🏗️ 微服务架构**: 数据服务、索引服务、模型服务分离
- **🚀 高性能**: 异步数据保存、智能缓存、批量操作
- **🛡️ 高可靠性**: 完善的错误处理、数据验证、健康检查
- **🔧 易扩展**: 标准化接口设计，支持插件化扩展

### MLOps支持
- **📈 实验管理**: A/B测试、模型版本控制
- **🔄 自动化流水线**: 数据收集→模型训练→部署→监控
- **📊 监控告警**: 实时性能监控、数据质量检查
- **🎯 可视化界面**: 基于Gradio的Web界面

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 内存: 至少2GB
- 存储: 至少1GB可用空间

### 安装

```bash
# 克隆项目
git clone https://github.com/tylerelyt/test_bed.git
cd test_bed

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 启动系统

```bash
# 方式1: 使用启动脚本
./quick_start.sh

# 方式2: 直接启动
python start_system.py
```

系统启动后，访问 http://localhost:7861 即可使用。

## 📖 使用指南

### 基本使用

1. **索引构建**: 系统启动时会自动构建索引，也可以手动添加文档
2. **搜索测试**: 在搜索框中输入查询词，系统会返回相关文档
3. **点击反馈**: 点击搜索结果可以记录用户行为，用于模型训练
4. **模型训练**: 收集足够数据后，可以训练CTR预测模型

### 高级功能

#### 1. 批量数据导入

```python
from src.search_engine.data_utils import import_ctr_data
result = import_ctr_data("path/to/your/data.json")
```

#### 2. 自定义排序策略

```python
from src.search_engine.service_manager import get_index_service
index_service = get_index_service()
results = index_service.search("查询词", top_k=10)
```

#### 3. 实验管理

系统支持A/B测试，可以在监控页面配置不同的排序策略进行对比。

## 🏗️ 架构设计

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI Layer      │    │  Business Layer │    │  Service Layer  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Portal    │ │───▶│ │ Search Tab  │ │───▶│ │DataService  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │Training Tab │ │───▶│ │IndexService │ │
│                 │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │Monitor Tab  │ │───▶│ │ModelService │ │
│                 │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 数据流

```
用户查询 → 索引检索 → 初排序 → CTR预测 → 重排序 → 结果展示
    ↓
用户点击 → 行为记录 → 数据存储 → 模型训练 → 模型更新
```

## 📊 性能指标

- **检索延迟**: < 100ms (10K文档)
- **并发支持**: 100+ 并发用户
- **内存使用**: < 500MB (基础配置)
- **存储效率**: 压缩率 > 70%

## 🛠️ 开发指南

### 项目结构

```
intelligent-search-engine/
├── src/                    # 源代码
│   └── search_engine/     
│       ├── data_service.py      # 数据服务
│       ├── index_service.py     # 索引服务
│       ├── model_service.py     # 模型服务
│       ├── service_manager.py   # 服务管理器
│       ├── data_utils.py        # 数据工具
│       └── portal.py           # UI入口
├── models/                # 模型文件
├── data/                  # 数据文件
├── docs/                  # 文档
├── test/                  # 测试
├── tools/                 # 工具脚本
└── requirements.txt       # 依赖
```

### 扩展开发

#### 添加新的排序算法

1. 在 `src/search_engine/ranking/` 创建新的排序模块
2. 实现 `RankingInterface` 接口
3. 在 `IndexService` 中注册新算法

#### 添加新的特征

1. 在 `CTRSampleConfig` 中定义新特征
2. 在 `DataService.record_impression` 中计算特征值
3. 更新模型训练逻辑

## 🧪 测试

```bash
# 运行单元测试
python -m pytest test/

# 运行集成测试
python test/test_integration.py

# 性能测试
python test/test_performance.py
```

## 📈 监控

系统提供多维度监控：

- **系统监控**: CPU、内存、磁盘使用率
- **业务监控**: 搜索QPS、点击率、响应时间
- **数据监控**: 数据质量、模型性能指标
- **告警机制**: 异常检测和自动告警

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [jieba](https://github.com/fxsjy/jieba) - 中文分词
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Gradio](https://gradio.app/) - Web界面框架
- [pandas](https://pandas.pydata.org/) - 数据处理

## 📞 联系方式

- 项目主页: https://github.com/tylerelyt/test_bed
- 问题反馈: https://github.com/tylerelyt/test_bed/issues
- 邮箱: tylerelyt@gmail.com

## 🔄 更新日志

### v1.0.0 (2024-07-14)
- ✨ 初始版本发布
- 🔍 基础搜索功能
- 🧠 CTR预测模型
- 📊 数据分析界面
- 🏗️ MLOps支持

---

⭐ 如果这个项目对你有帮助，请给它一个星标！ 