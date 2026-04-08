import json
import os
from flask import Flask, render_template, request, jsonify
from claude_service import analyze_text, get_word_library, reload_word_library

app = Flask(__name__)
WORDS_PATH = os.path.join(os.path.dirname(__file__), "words.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "请提供要检测的文本"}), 400

    text = data["text"]
    if len(text) > 5000:
        return jsonify({"error": "文本过长，请控制在5000字以内"}), 400

    try:
        result = analyze_text(text)
        return jsonify(result)
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return jsonify({"error": "API Key 无效，请检查 .env 文件中的 DEEPSEEK_API_KEY"}), 500
        elif "timeout" in error_msg.lower():
            return jsonify({"error": "请求超时，请重试"}), 504
        else:
            return jsonify({"error": f"分析失败：{error_msg}"}), 500


@app.route("/words", methods=["GET"])
def words_list():
    """返回当前词库（供前端词库管理页使用）。"""
    return jsonify(get_word_library())


@app.route("/words", methods=["POST"])
def words_add():
    """添加自定义词条：{"word": "xx", "suggestions": ["a", "b"]}"""
    data = request.get_json()
    word = (data or {}).get("word", "").strip()
    suggestions = (data or {}).get("suggestions", [])
    if not word:
        return jsonify({"error": "word 不能为空"}), 400

    library = get_word_library()
    library[word] = suggestions

    with open(WORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)

    reload_word_library()
    return jsonify({"ok": True, "total": len(library)})


@app.route("/words/<path:word>", methods=["DELETE"])
def words_delete(word):
    """删除词条。"""
    library = get_word_library()
    if word not in library:
        return jsonify({"error": "词条不存在"}), 404

    del library[word]
    with open(WORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)

    reload_word_library()
    return jsonify({"ok": True, "total": len(library)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
