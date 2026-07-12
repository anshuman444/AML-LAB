import io
import base64
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/visualize_regression", methods=["POST"])
def visualize_regression():
    data = request.get_json()
    x = np.array(data["x"])
    y = np.array(data["y"])
    m = data["m"]
    b = data["b"]
    y_pred = []
    for i in range(len(x)):
        y_pred.append(m * x[i] + b)
    plt.figure()
    plt.scatter(x, y, label="Data Points")
    plt.plot(x, y_pred, color="red", label="Regression Line")
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    image1_b64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    image1_data_uri = f"data:image/png;base64,{image1_b64}"
    plt.close()

    metrics = ["MSE", "MAE", "RMSE", "R2"]
    values = [data["mse"], data["mae"], data["rmse"], data["r2"]]
    plt.figure(figsize=(6,4))
    plt.scatter(metrics, values)
    for i in range(4):
        plt.text(metrics[i], values[i], round(values[i], 4))
    plt.title("Model Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.grid(True)
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    image2_b64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    image2_data_uri = f"data:image/png;base64,{image2_b64}"
    plt.close()

    return jsonify({
        "image1": image1_data_uri,
        "image2": image2_data_uri
    })

if __name__ == "__main__":
    app.run(debug=True)
