from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from flask import render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# 載入 API 金鑰
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "secret.env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("API_KEY")

# 初始化 Gemini 模型
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

# 初始化對話歷史（每次啟動都會重來，可後續支援 session）
chat_history = [
    SystemMessage(content="""你是一位語音式的急救助手。請一步一步引導使用者處理有人昏倒的緊急情況。
1. 每次只給一個步驟。
2. 請根據使用者的回應決定下一步。
3. 回答必須冷靜、清楚，並適時地鼓勵急救者給予情緒價值。
4. 回答必須簡潔明瞭，避免使用專業術語與過多冗言贅字。
5. 急救電話統一為119。
6. 如果使用者回答不清楚，請詢問更多資訊。
7. 使用繁體中文。
8. 請從「我是急救助手，請問現在發生什麼情況？」這句話開始對話。
9. 略過詢問是否安全，直接說明要檢查昏倒者的意識狀態。
"""),
    HumanMessage(content="hi")
]


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(response)

    return jsonify({"response": response.content})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

