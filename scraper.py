"""
scraper.py — 小红书笔记内容抓取

小红书必须登录才能访问笔记，本模块通过用户提供的浏览器 Cookie
调用移动端 API 获取笔记标题 + 正文，不依赖 headless browser。

使用方法：
    result = fetch_note(url, cookie_str)
    # result = {"title": "...", "desc": "...", "text": "...（合并后全文）"}
"""

import re
import json
import hashlib
import time
import requests

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Origin": "https://www.xiaohongshu.com",
    "Referer": "https://www.xiaohongshu.com/",
})


def _extract_note_id(url: str) -> str | None:
    """
    从各种小红书链接格式中提取 note_id。
    支持：
      - https://www.xiaohongshu.com/explore/<id>
      - https://www.xiaohongshu.com/discovery/item/<id>
      - https://xhslink.com/a/<short>   （需先 redirect）
      - 纯 24 位 hex id
    """
    url = url.strip()
    # 直接 24 位 hex
    if re.fullmatch(r'[0-9a-f]{24}', url):
        return url
    # explore / discovery
    m = re.search(r'/(?:explore|discovery/item)/([0-9a-f]{24})', url)
    if m:
        return m.group(1)
    # xhslink short URL — follow redirect
    if 'xhslink.com' in url:
        try:
            r = requests.head(url, allow_redirects=True, timeout=10,
                              headers={"User-Agent": _SESSION.headers["User-Agent"]})
            return _extract_note_id(r.url)
        except Exception:
            pass
    return None


def fetch_note(url: str, cookie_str: str = "") -> dict:
    """
    抓取笔记内容，返回：
        {
            "title": str,
            "desc": str,
            "text": str,   # title + '\n' + desc 的合并全文，用于检测
            "note_id": str,
            "error": str | None
        }

    cookie_str：浏览器 document.cookie 字符串，登录后才能访问内容。
    """
    note_id = _extract_note_id(url)
    if not note_id:
        return {"error": "无法解析笔记 ID，请确认链接格式正确", "text": ""}

    # 构建 Cookie header
    headers = {}
    if cookie_str.strip():
        headers["Cookie"] = cookie_str.strip()

    # 移动端 note_info API
    api_url = f"https://edith.xiaohongshu.com/api/sns/h5/v1/note_info?note_id={note_id}"
    try:
        resp = _SESSION.get(api_url, headers=headers, timeout=15)
        data = resp.json()
    except Exception as e:
        return {"error": f"请求失败：{e}", "text": ""}

    code = data.get("code", -1)
    if code == -1 and not data.get("success"):
        return {
            "error": "接口返回失败，可能原因：① Cookie 未填写或已过期 ② 笔记已删除/设为私密",
            "text": ""
        }
    if code == 461:
        return {"error": "需要登录授权，请在设置中填入小红书 Cookie", "text": ""}
    if code != 0:
        return {"error": f"接口错误（code={code}）：{data.get('msg', '')}", "text": ""}

    note = data.get("data", {}).get("note_detail_map", {})
    # API 返回结构：{note_id: {note: {title, desc, ...}}}
    note_info = note.get(note_id, {}).get("note", {}) if note else {}
    if not note_info:
        # 兜底：data 直接是 note 对象
        note_info = data.get("data", {})

    title = note_info.get("title", "").strip()
    desc = note_info.get("desc", "").strip()

    if not title and not desc:
        return {
            "error": "获取到笔记数据但内容为空，可能笔记已删除或 Cookie 权限不足",
            "text": ""
        }

    text = (title + "\n" + desc).strip() if title else desc
    return {
        "title": title,
        "desc": desc,
        "text": text,
        "note_id": note_id,
        "error": None
    }
