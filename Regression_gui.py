import tkinter as tk
from tkinter import scrolledtext
import threading
import os


# Colors
BG      = "#0a0a1a"
BG2     = "#0d1b2a"
CARD    = "#111827"
BORDER  = "#1e2d3d"
ACCENT  = "#00f5ff"
PURPLE  = "#7b2ff7"
TEXT    = "#ccd6f6"
SUBTEXT = "#8892b0"
SUCCESS = "#64ffda"
GOLD    = "#FFD700"
RED     = "#ff6b6b"


# Models List
MODELS = [
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "Decision Tree",
    "Neural Network",
    "Ridge Regression",
    "Lasso Regression",
    "Linear Regression",
    "Elastic Net",
    "Polynomial Regression",  # ✅
    "SVR",                    # ✅
    "Voting Ensemble",
]


MODEL_ICONS = {
    "Random Forest":     "🌲",
    "Gradient Boosting": "📈",
    "XGBoost":           "⚡",
    "Decision Tree":     "🌳",
    "Neural Network":    "🧠",
    "Ridge Regression":  "📊",
    "Polynomial Regression": "🔢",
    "SVR": "🎯",
    "Lasso Regression":  "📉",
    "Linear Regression": "📐",
    "Elastic Net":       "🔗",
    "Voting Ensemble":   "🔀",
}


# Run
def run_model(model_name, output_callback):
    try:
        import joblib
        import numpy as np
        from sklearn.metrics import r2_score, mean_squared_error

        #Check file
        if not os.path.exists("all_predictions.pkl"):
            output_callback("❌  'all_predictions.pkl' not found!\n\n")
            output_callback("👉  Add this to the END of main_final.py and run it:\n\n")
            output_callback("    all_predictions = {\n")
            output_callback("        'Random Forest': (y_test_pred_log, y_train_pred_rf, y_test_log, y_train_log),\n")
            output_callback("        ... etc\n")
            output_callback("    }\n")
            output_callback("    joblib.dump(all_predictions, 'all_predictions.pkl')\n")
            return

        output_callback("⏳ Loading predictions...\n")
        all_preds = joblib.load("all_predictions.pkl")

        if model_name not in all_preds:
            output_callback(f"❌  '{model_name}' not found in predictions file!\n")
            output_callback(f"Available: {list(all_preds.keys())}\n")
            return

        y_test_pred, y_train_pred, y_test_log, y_train_log = all_preds[model_name]

        output_callback("✅ Predictions loaded\n")
        output_callback("⏳ Calculating metrics...\n")

        #Metrics
        test_r2  = r2_score(y_test_log,  y_test_pred)
        train_r2 = r2_score(y_train_log, y_train_pred)
        rmse     = np.sqrt(mean_squared_error(y_test_log, y_test_pred))
        diff     = train_r2 - test_r2
        status   = "⚠️  Overfitting!" if diff > 0.10 else "✅  Balanced Model"

        icon = MODEL_ICONS.get(model_name, "🤖")

        result = (
            f"\n{'═'*44}\n"
            f"  {icon}  MODEL RESULTS\n"
            f"{'═'*44}\n"
            f"  Model      :  {model_name}\n"
            f"  Test  R²   :  {test_r2*100:.2f}%\n"
            f"  Train R²   :  {train_r2*100:.2f}%\n"
            f"  Difference :  {diff*100:.1f}%\n"
            f"  RMSE (log) :  {rmse:.4f}\n"
            f"  Status     :  {status}\n"
            f"{'═'*44}\n"
        )

        output_callback(result)
        output_callback("DONE ✅\n")

    except Exception as e:
        import traceback
        output_callback(f"\n❌ Error:\n{traceback.format_exc()}\n")



# GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Game Popularity — Model Runner")
        self.root.geometry("720x680")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self._build()

    def _build(self):
        #Header
        tk.Label(self.root, text="🎮  GAME POPULARITY PREDICTOR",
                 bg=BG, fg=ACCENT, font=("Consolas", 15, "bold")).pack(pady=(18, 2))
        tk.Label(self.root, text="Select a model  →  Run  →  See results",
                 bg=BG, fg=SUBTEXT, font=("Consolas", 9)).pack()
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)

        #Model Selection
        sel_frame = tk.LabelFrame(
            self.root, text="  ⚡ SELECT MODEL  ",
            bg=CARD, fg=ACCENT, font=("Consolas", 9, "bold"),
            relief="flat", highlightthickness=1, highlightbackground=BORDER
        )
        sel_frame.pack(fill="x", padx=20, pady=4)

        self.model_var = tk.StringVar(value=MODELS[0])
        inner = tk.Frame(sel_frame, bg=CARD)
        inner.pack(padx=10, pady=8, fill="x")

        tree_models   = ["Random Forest", "Gradient Boosting", "XGBoost",
                         "Decision Tree", "Voting Ensemble"]
        linear_models = ["Ridge Regression", "Lasso Regression",
                         "Linear Regression", "Elastic Net"]

        for i, name in enumerate(MODELS):
            icon  = MODEL_ICONS.get(name, "🤖")
            if name == "Neural Network":
                color = GOLD
            elif name in tree_models:
                color = ACCENT
            else:
                color = SUCCESS

            tk.Radiobutton(
                inner,
                text=f"{icon}  {name}",
                variable=self.model_var, value=name,
                bg=CARD, fg=color, selectcolor=BG2,
                activebackground=CARD, activeforeground=ACCENT,
                font=("Consolas", 11), anchor="w"
            ).grid(row=i // 2, column=i % 2, sticky="w", padx=16, pady=4)

        #Legend
        legend = tk.Frame(self.root, bg=BG)
        legend.pack(fill="x", padx=24, pady=(2, 0))
        tk.Label(legend, text="🔵 Tree Models",   bg=BG, fg=ACCENT,  font=("Consolas", 8)).pack(side="left", padx=6)
        tk.Label(legend, text="🟡 Neural Network",bg=BG, fg=GOLD,    font=("Consolas", 8)).pack(side="left", padx=6)
        tk.Label(legend, text="🟢 Linear Models", bg=BG, fg=SUCCESS, font=("Consolas", 8)).pack(side="left", padx=6)

        #Status
        self.status_var = tk.StringVar(value="Ready — select a model and press RUN")
        tk.Label(self.root, textvariable=self.status_var,
                 bg=BG, fg=SUBTEXT, font=("Consolas", 8),
                 anchor="w").pack(fill="x", padx=22, pady=(6, 0))

        #Run Button
        self.run_btn = tk.Button(
            self.root, text="▶   RUN MODEL",
            command=self._on_run,
            bg=PURPLE, fg="white",
            activebackground=ACCENT, activeforeground=BG,
            relief="flat", font=("Consolas", 12, "bold"),
            padx=20, pady=10, cursor="hand2"
        )
        self.run_btn.pack(fill="x", padx=20, pady=8)

        #Console
        tk.Label(self.root, text="📋  OUTPUT",
                 bg=BG, fg=ACCENT, font=("Consolas", 9, "bold"),
                 anchor="w").pack(fill="x", padx=20)

        self.console = scrolledtext.ScrolledText(
            self.root, bg="#060d14", fg=SUCCESS,
            insertbackground=ACCENT, font=("Consolas", 10),
            relief="flat", height=12, padx=12, pady=10,
            highlightthickness=1, highlightbackground=BORDER
        )
        self.console.pack(fill="both", expand=True, padx=20, pady=(4, 4))

        #Footer
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=20, pady=4)
        tk.Label(self.root,
                 text="Online Games Popularity Prediction  ·  ML Project",
                 bg=BG, fg=SUBTEXT, font=("Consolas", 8)).pack(pady=4)

    def _log(self, text):
        self.console.insert("end", text)
        self.console.see("end")
        self.root.update_idletasks()

    def _on_run(self):
        self.console.delete("1.0", "end")
        self.run_btn.config(text="⏳  Running...", state="disabled", bg=SUBTEXT)
        self.status_var.set(f"Running {self.model_var.get()}...")

        selected = self.model_var.get()

        def callback(text):
            self.root.after(0, lambda t=text: self._log(t))

        def worker():
            run_model(selected, callback)
            self.root.after(0, lambda: self.run_btn.config(
                text="▶   RUN MODEL", state="normal", bg=PURPLE))
            self.root.after(0, lambda: self.status_var.set("Done ✅"))

        threading.Thread(target=worker, daemon=True).start()



# Main
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
