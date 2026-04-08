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

你的任务：**仅识别词库未覆盖的风险表达**，并区分两类：

## 违禁词（category: "banned"）
可能导致笔记被删除或账号封禁，包括：
- 医疗效果宣称（"治愈"、"根治"等）
- 虚假绝对化承诺（"100%有效"等）
- 违规引流（"私我"、"找我拿"、"加V"等）
- 违禁商品相关词
- 谐音/变形违规词（如"減féi"）

## 限流词（category: "throttle"）
可能导致流量降低或笔记被降权，包括：
- 过度营销词（"爆款"、"网红"等）
- 平台敏感话题
- 竞品品牌名称
- 诱导性互动词（"点赞"、"关注"、"转发"等变体）
- 模糊夸大词（"神奇"、"奇迹"等）

## 输出格式（严格 JSON，不含任何其他文字）

{{
  "banned_words": [
    {{
      "word": "原文中完全一致的子串",
      "position": [起始索引, 结束索引],
      "reason": "违禁原因（20字以内）",
      "suggestions": ["替换方案1", "替换方案2"]
    }}
  ],
  "throttle_words": [
    {{
      "word": "原文中完全一致的子串",
      "position": [起始索引, 结束索引],
      "reason": "限流原因（20字以内）",
      "suggestions": ["替换方案1", "替换方案2"]
    }}
  ],
  "risk_level": "high 或 medium 或 low",
  "summary": "整体风险评估（100字以内）"
}}

如无额外发现，对应数组返回空数组。风险等级综合所有命中数量判断。"""


# ── 第一层：词库精确匹配 ──────────────────────────────────
def _is_ascii_word(s: str) -> bool:
    """判断字符串是否全由 ASCII 字母或数字组成（需要词边界检测）。"""
    return bool(s) and all(c.isascii() and (c.isalpha() or c.isdigit()) for c in s)


def _has_word_boundary(text: str, idx: int, word: str) -> bool:
    """
    对纯 ASCII 字母/数字词，检查命中位置前后是否为词边界，
    避免 'q' 匹配到 'sequential' 内部。
    中文词/混合词不做边界检查，直接返回 True。
    """
    if not _is_ascii_word(word):
        return True
    end = idx + len(word)
    before_ok = (idx == 0) or not (text[idx - 1].isascii() and (text[idx - 1].isalpha() or text[idx - 1].isdigit()))
    after_ok  = (end == len(text)) or not (text[end].isascii() and (text[end].isalpha() or text[end].isdigit()))
    return before_ok and after_ok


def _dict_match(text: str) -> list[dict]:
    """
    词库精确匹配，返回命中列表。
    所有词库词默认归类为 banned（违禁词），AI 层会补充 throttle 类别。
    """
    hits = []
    occupied = set()

    for word in SORTED_WORDS:
        search_from = 0
        while True:
            idx = text.find(word, search_from)
            if idx == -1:
                break
            if _has_word_boundary(text, idx, word):
                span = set(range(idx, idx + len(word)))
                if not span & occupied:
                    occupied |= span
                    hits.append({
                        "word": word,
                        "position": [idx, idx + len(word)],
                        "reason": "词库收录敏感词",
                        "suggestions": WORD_LIBRARY[word],
                        "category": "banned",
                        "source": "dict"
                    })
            search_from = idx + 1

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
def _validate_ai_words(text: str, ai_words: list[dict], dict_hits: list[dict], category: str) -> list[dict]:
    """验证 AI 返回词的位置，去重，打上 category 和 source 标记。"""
    validated = []
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
            item["category"] = category
            item["source"] = "ai"
            validated.append(item)
    return validated


def _merge_results(text: str, dict_hits: list[dict], ai_result: dict) -> dict:
    # AI 补充的违禁词
    ai_banned = _validate_ai_words(
        text, ai_result.get("banned_words", []), dict_hits, "banned"
    )
    # AI 补充的限流词（与已验证的所有词去重）
    all_so_far = dict_hits + ai_banned
    ai_throttle = _validate_ai_words(
        text, ai_result.get("throttle_words", []), all_so_far, "throttle"
    )

    all_words = dict_hits + ai_banned + ai_throttle
    all_words.sort(key=lambda x: x["position"][0])

    banned_list    = [w for w in all_words if w.get("category") == "banned"]
    throttle_list  = [w for w in all_words if w.get("category") == "throttle"]

    # 综合风险等级
    ai_risk = ai_result.get("risk_level", "low")
    total_banned   = len(banned_list)
    total_throttle = len(throttle_list)
    if total_banned >= 3 or ai_risk == "high":
        risk = "high"
    elif total_banned >= 1 or total_throttle >= 3 or ai_risk == "medium":
        risk = "medium"
    else:
        risk = "low"

    # 摘要
    parts = []
    dict_banned_cnt  = len([w for w in dict_hits if w.get("category") == "banned"])
    ai_banned_cnt    = len(ai_banned)
    ai_throttle_cnt  = len(ai_throttle)

    if dict_banned_cnt:
        parts.append(f"词库命中 {dict_banned_cnt} 个违禁词")
    if ai_banned_cnt:
        parts.append(f"AI 额外发现 {ai_banned_cnt} 个违禁表达")
    if ai_throttle_cnt:
        parts.append(f"AI 发现 {ai_throttle_cnt} 个限流词")
    if not all_words:
        parts.append("文案基本合规，未发现明显问题")
    ai_summary = ai_result.get("summary", "")
    if ai_summary and all_words:
        parts.append(ai_summary)

    return {
        "banned_words": banned_list,
        "throttle_words": throttle_list,
        "sensitive_words": all_words,      # 兼容旧格式，保留合并列表
        "risk_level": risk,
        "summary": "；".join(parts) if parts else "检测完成。"
    }


# ── 主入口 ────────────────────────────────────────────────
def analyze_text(text: str) -> dict:
    if not text or not text.strip():
        return {
            "banned_words": [],
            "throttle_words": [],
            "sensitive_words": [],
            "risk_level": "low",
            "summary": "文本为空，无需检测。"
        }

    dict_hits  = _dict_match(text)
    ai_result  = _ai_analyze(text, dict_hits)
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
        "banned_words": [],
        "throttle_words": [],
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
