import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ── 加载词库 ──────────────────────────────────────────────
_WORDS_PATH = os.path.join(os.path.dirname(__file__), "words.json")

def _load_word_library() -> dict:
    try:
        with open(_WORDS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# 按词长降序排列，优先匹配长词（避免"最低价"被"最"先截断）
WORD_LIBRARY: dict = _load_word_library()
SORTED_WORDS: list[str] = sorted(WORD_LIBRARY.keys(), key=len, reverse=True)


# ── AI 兜底的 System Prompt ────────────────────────────────
AI_SYSTEM_PROMPT = """你是一位专业的小红书内容合规顾问。
以下敏感词已由词库精确匹配完成，无需重复识别：
{already_found}

你的任务：**仅识别词库未覆盖的隐晦风险表达**，包括：
- 谐音规避词（如"減féi"、"z最好"等变形写法）
- 语义过度承诺（如"完全没有任何副作用"、"彻底根除"等）
- 隐晦引流词（如"私我"、"找我拿"、"滴我"等）
- 竞品品牌名称

## 输出格式（严格 JSON，不含任何其他文字）

{{
  "sensitive_words": [
    {{
      "word": "敏感词原文（与输入文本完全一致的子串）",
      "position": [起始字符索引, 结束字符索引],
      "reason": "敏感原因（20字以内）",
      "suggestions": ["替换方案1", "替换方案2"]
    }}
  ],
  "risk_level": "high 或 medium 或 low",
  "summary": "整体风险评估（80字以内）"
}}

如无额外发现，sensitive_words 返回空数组。风险等级综合词库命中数量和AI发现结果共同判断。"""


# ── 第一层：词库精确匹配 ──────────────────────────────────
def _dict_match(text: str) -> list[dict]:
    """
    遍历词库，找出所有命中词及其位置。
    使用贪婪策略：已被标记的字符区间不会被重复命中。
    """
    hits = []
    occupied = set()  # 已命中的字符索引集合，避免重叠

    for word in SORTED_WORDS:
        search_from = 0
        while True:
            idx = text.find(word, search_from)
            if idx == -1:
                break
            span = set(range(idx, idx + len(word)))
            if not span & occupied:  # 无重叠才记录
                occupied |= span
                hits.append({
                    "word": word,
                    "position": [idx, idx + len(word)],
                    "reason": "词库收录敏感词",
                    "suggestions": WORD_LIBRARY[word],
                    "source": "dict"
                })
            search_from = idx + 1

    # 按出现位置排序
    hits.sort(key=lambda x: x["position"][0])
    return hits


# ── 第二层：AI 语义兜底 ───────────────────────────────────
def _ai_analyze(text: str, dict_hits: list[dict]) -> dict:
    already = ", ".join(f'"{h["word"]}"' for h in dict_hits) if dict_hits else "（无）"
    system = AI_SYSTEM_PROMPT.format(already_found=already)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"请分析以下小红书文案：\n\n{text}"}
        ],
        max_tokens=2048,
        temperature=0.1,
    )
    return _parse_json_response(response.choices[0].message.content or "")


# ── 合并结果 ──────────────────────────────────────────────
def _merge_results(text: str, dict_hits: list[dict], ai_result: dict) -> dict:
    ai_words = ai_result.get("sensitive_words", [])

    # 验证 AI 返回词的位置
    validated_ai = []
    dict_positions = {tuple(h["position"]) for h in dict_hits}

    for item in ai_words:
        word = item.get("word", "")
        if not word:
            continue
        position = item.get("position", [])
        # 修正位置
        if not (len(position) == 2 and
                isinstance(position[0], int) and
                isinstance(position[1], int) and
                0 <= position[0] < position[1] <= len(text) and
                text[position[0]:position[1]] == word):
            idx = text.find(word)
            if idx == -1:
                continue
            position = [idx, idx + len(word)]
            item["position"] = position

        # 去重：与词库命中位置不重叠
        span = set(range(position[0], position[1]))
        overlaps = any(
            span & set(range(h["position"][0], h["position"][1]))
            for h in dict_hits
        )
        if not overlaps:
            item["source"] = "ai"
            validated_ai.append(item)

    all_words = dict_hits + validated_ai
    all_words.sort(key=lambda x: x["position"][0])

    # 综合风险等级
    total = len(all_words)
    ai_risk = ai_result.get("risk_level", "low")
    if total >= 5 or ai_risk == "high":
        risk = "high"
    elif total >= 2 or ai_risk == "medium":
        risk = "medium"
    else:
        risk = "low"

    dict_count = len(dict_hits)
    ai_count = len(validated_ai)
    summary_parts = []
    if dict_count:
        summary_parts.append(f"词库命中 {dict_count} 个敏感词")
    if ai_count:
        summary_parts.append(f"AI 额外发现 {ai_count} 个隐晦表达")
    if not all_words:
        summary_parts.append("文案基本合规，未发现明显敏感词")

    ai_summary = ai_result.get("summary", "")
    if ai_summary and all_words:
        summary_parts.append(ai_summary)

    return {
        "sensitive_words": all_words,
        "risk_level": risk,
        "summary": "；".join(summary_parts) if summary_parts else "检测完成。"
    }


# ── 主入口 ────────────────────────────────────────────────
def analyze_text(text: str) -> dict:
    if not text or not text.strip():
        return {
            "sensitive_words": [],
            "risk_level": "low",
            "summary": "文本为空，无需检测。"
        }

    # 第一层：词库精确匹配（快速、确定）
    dict_hits = _dict_match(text)

    # 第二层：AI 语义分析（识别词库未覆盖的隐晦表达）
    ai_result = _ai_analyze(text, dict_hits)

    # 合并两层结果
    return _merge_results(text, dict_hits, ai_result)


# ── 工具函数 ──────────────────────────────────────────────
def _parse_json_response(response_text: str) -> dict:
    cleaned = response_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {
        "sensitive_words": [],
        "risk_level": "low",
        "summary": "AI 分析结果解析失败，词库命中结果仍有效。",
        "parse_error": True
    }


def get_word_library() -> dict:
    """供前端词库管理接口使用。"""
    return WORD_LIBRARY


def reload_word_library():
    """重新加载词库（用于热更新）。"""
    global WORD_LIBRARY, SORTED_WORDS
    WORD_LIBRARY = _load_word_library()
    SORTED_WORDS = sorted(WORD_LIBRARY.keys(), key=len, reverse=True)
