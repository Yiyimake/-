# redkb —— 本地红色知识库 + RAG

这个目录是「图片识别 + 红色知识库 RAG」的实现，配合上一级的 `heritage_server.py` 使用。

```
redkb/
├── kb.jsonl        史料库（一行一条，人工审核录入）
├── rag.py          检索：BM25 + 特征哈希向量，RRF 融合，可选 FAISS
├── recognize.py    CLIP 零样本图像识别（可选依赖，缺了就降级为手动选择）
├── llm.py          讲解生成：通义千问 / 讯飞星火 / 任意 OpenAI 兼容端点，缺了就本地抽取
└── README.md       本文件
```

## 一、防编造的机制（答辩可以直接讲这三点）

1. **知识库是唯一事实来源**：`kb.jsonl` 里每一条都是人工录入的史料，模型只能引用它。
2. **检索结果逐条编号**：页面把命中的史料列成 [1][2][3]，讲解里每句史实都要标引用号，
   评委可以逐条对照——指哪打哪。
3. **两种生成模式都受限**：
   - 配了大模型：system prompt 里写死「只能用给定的【史料】、史料不足就说不足、禁止编造人名地名数字」；
   - 没配大模型：`extractive` 模式**直接拼接库里的原文**，一个 token 都不生成，天然不可能编造。

## 二、知识库格式（加条目就照这个写）

`kb.jsonl` 一行一个 JSON，字段如下：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | ✅ | 唯一编号，如 `kb-031` |
| `name` | ✅ | 文物/旧址名称 |
| `type` | | 革命文物 / 革命旧址 / 重要历史事件 |
| `place` | | 所在地 |
| `era` | | 年代，如 `1935年1月` |
| `summary` | ✅ | 概述（2~4 句，**只写有出处的事实**） |
| `facts` | ✅ | 关键史实数组，一条一句，便于逐条引用 |
| `tags` | | 关键词，检索和 CLIP 提示词都会用到 |
| `interpretation` | | 讲解要点（怎么讲、讲什么最打动人） |
| `aliases` | | 别名 |
| `source_note` | ✅ | 出处备注（**必须补到具体文献**） |
| `reviewed` | | 是否已人工审核，`true/false` |

加完条目重启服务即可生效，检索、CLIP 候选、讲解全部自动跟上。

> ⚠️ 现有 30 条示例的史实是公开常见表述，但 `source_note` 还都写着「待补充权威出处」，
> `reviewed` 也全是 `false`。**正式答辩前必须逐条核对并填上出处**（《中国共产党简史》、文物档案、
> 纪念馆展陈资料等），否则「人工审核录入」这句话站不住。

## 三、现在能跑什么、什么需要额外装

| 能力 | 现状 | 怎么启用 |
| --- | --- | --- |
| 史料检索 | ✅ 开箱可用（BM25 + 向量，numpy） | 无需配置 |
| 讲解生成 | ✅ 本地抽取模式 | 无需配置 |
| 讲解生成（大模型） | ⚠️ 需 API Key | `set DASHSCOPE_API_KEY=sk-xxx`（通义千问）或 `set SPARK_API_KEY=xxx`（星火），重启服务 |
| FAISS 向量库 | ⚠️ 未安装，自动用 numpy（结果一致） | `pip install faiss-cpu` 后重启 |
| CLIP 图像识别 | ⚠️ 未安装 transformers | 见下 |

### 启用 CLIP 图像识别

```powershell
pip install transformers
set HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import ChineseCLIPModel; ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')"
```

模型会下到 `%USERPROFILE%\.cache\huggingface`。如果机器不能联网，可以先在别的机器下好，
整个目录拷过来，然后：

```powershell
set REDKB_CLIP_MODEL=D:\models\chinese-clip-vit-base-patch16
```

重启服务后，页面上的「🔍 识别这张照片」按钮就亮了，识别结果会给出前 3 个候选及置信度。

### 启用大模型讲解

```powershell
# 通义千问（阿里云百炼，比赛/学生有免费额度）
set DASHSCOPE_API_KEY=sk-xxxxxxxx

# 或讯飞星火
set SPARK_API_KEY=xxxxxxxx

# 或任何 OpenAI 兼容端点
set REDKB_LLM_BASE_URL=https://your-endpoint/v1
set REDKB_LLM_MODEL=your-model
set REDKB_LLM_API_KEY=sk-xxxxxxxx
```

配好重启服务，页面上的讲解标签会从「本地抽取」变成绿色的「大模型生成」，
右下角会显示用的哪个模型。API 调用失败会自动退回本地抽取，并如实说明原因。

## 四、接口

```
GET  /api/kb/status                       条目数、检索后端、LLM/CLIP 状态
GET  /api/kb/list                         条目清单（下拉框用）
POST /api/kb/search   {query, top_k, entry_id?}
POST /api/kb/recognize {image, top_k?}          → 候选 [{id, name, score}]
POST /api/kb/story     {entry_id? | query?, question?}
     → {entry, passages[], story{text, mode, provider, model, citations[]}}
```

## 五、调优建议

* **召回不准**：先看 `tags`/`aliases` 里有没有常用叫法（如「飞夺泸定桥」「红船」），
  这些字段对检索权重很大。
* **讲解太模板化**：这是没接大模型的必然结果；配 Key 后自然变成叙述性文本。
* **想换更好的向量**：`rag.py` 里 `embed()` 是个 40 行的特征哈希实现，替换成
  sentence-transformers / text2vec 的中文向量模型后，检索质量会明显提升，
  接口不用改（只要仍返回单位向量）。
