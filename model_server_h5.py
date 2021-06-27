import numpy as np
import json
import traceback
from flask import Flask, request
from keras.models import load_model
from keras_bert import get_custom_objects
from model_train import token_dict, OurTokenizer

maxlen = 10
tokenizer = OurTokenizer(token_dict)

with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())

# custom_objects = get_custom_objects()
model = load_model("cls_cnews.h5", custom_objects=get_custom_objects())

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def hello_world():
    return 'Hello World!'


@app.route("/predict", methods=["GET", "POST"])
def predict():
    return_result = {"code": 200, "message": "success", "data": []}
    try:
        text = request.get_json()["text"]
        print('Input data', text)
        # 利用BERT进行tokenize
        X1, X2 = tokenizer.encode(first=text, max_len=maxlen)
        # 模型预测并输出预测结果
        predicted = model.predict([[X1], [X2]])
        y = np.argmax(predicted[0])
        return_result["data"] = {"text": text, "label": label_dict[str(y)]}
    except Exception:
        return_result["code"] = 400
        return_result["message"] = traceback.format_exc()

    return json.dumps(return_result, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=15000, threaded=False)
