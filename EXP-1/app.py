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
    image1_path = "static/regression_plot.png"
    plt.savefig(image1_path)
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
    image2_path = "static/metrics_plot.png"
    plt.savefig(image2_path)
    plt.close()

    return jsonify({
        "image1": image1_path,
        "image2": image2_path
    })

if __name__ == "__main__":
    app.run(debug=True)
