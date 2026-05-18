import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (same as regression GUI)
# ─────────────────────────────────────────────
BG       = "#1a1a2e"
PANEL    = "#16213e"
ACCENT   = "#00ff88"
ACCENT2  = "#00ccff"
BTN_BG   = "#0f3460"
TXT      = "#e0e0e0"
DIM      = "#888888"
RED      = "#ff4444"
GOLD     = "#ffd700"
FONT_MON = ("Courier New", 10)

# ─────────────────────────────────────────────
#  MODELS AVAILABLE
# ─────────────────────────────────────────────
MODELS = {
    "Random Forest":      "random_forest_model.pkl",
    "XGBoost":            "xgb_random_search.pkl",
    "Neural Network":     "mlp_model.pkl",
    "LightGBM":           "lgbm_model.pkl",
    "CatBoost":           "final_catboost_model.pkl",
    "Gradient Boosting":  "gradient_boosting_model.pkl",
    "Decision Tree":      "decision_tree_model.pkl",
    "SVM":                "svm_model.pkl",
    "Logistic Regression":"logistic_model.pkl",
    "KNN":                "knn_model.pkl",
}

MODEL_ICONS = {
    "Random Forest":      "🌲",
    "XGBoost":            "⚡",
    "Neural Network":     "🧠",
    "LightGBM":           "💡",
    "CatBoost":           "🐱",
    "Gradient Boosting":  "🚀",
    "Decision Tree":      "🌿",
    "SVM":                "🔷",
    "Logistic Regression":"📊",
    "KNN":                "📍",
}

CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}


# ─────────────────────────────────────────────
#  HELPER – load artefacts silently
# ─────────────────────────────────────────────
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class ClassificationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Popularity — Classification Runner")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # centre on screen
        w, h = 720, 780
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self.selected_model = tk.StringVar(value="Random Forest")
        self._build_ui()

    # ── UI ──────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(18, 6))

        tk.Label(hdr, text="🎮  GAME POPULARITY PREDICTOR",
                 font=("Courier New", 17, "bold"),
                 fg=ACCENT, bg=BG).pack()
        tk.Label(hdr, text="Classification Edition  •  Select a model → Run → See results",
                 font=("Courier New", 9), fg=DIM, bg=BG).pack(pady=(2, 0))

        self._divider()

        # ── Model selector ───────────────────────
        sel_frame = tk.LabelFrame(self.root, text="  ⚙  SELECT MODEL  ",
                                  font=("Courier New", 10, "bold"),
                                  fg=ACCENT, bg=PANEL,
                                  bd=1, relief="flat",
                                  highlightbackground=ACCENT,
                                  highlightthickness=1)
        sel_frame.pack(fill="x", padx=20, pady=6)

        names = list(MODELS.keys())
        for i, name in enumerate(names):
            row, col = divmod(i, 2)
            icon = MODEL_ICONS.get(name, "•")
            rb = tk.Radiobutton(
                sel_frame,
                text=f"  {icon}  {name}",
                variable=self.selected_model,
                value=name,
                font=("Courier New", 10),
                fg=TXT, bg=PANEL,
                selectcolor=BTN_BG,
                activebackground=PANEL,
                activeforeground=ACCENT,
                indicatoron=True,
                cursor="hand2",
            )
            rb.grid(row=row, column=col, sticky="w", padx=20, pady=5)

        sel_frame.columnconfigure(0, weight=1)
        sel_frame.columnconfigure(1, weight=1)

        self._divider()

        # ── Run button ───────────────────────────
        self.run_btn = tk.Button(
            self.root,
            text="▶   RUN MODEL",
            font=("Courier New", 13, "bold"),
            fg=BG, bg=ACCENT,
            activebackground="#00cc66",
            activeforeground=BG,
            bd=0, cursor="hand2",
            pady=12,
            command=self._on_run,
        )
        self.run_btn.pack(fill="x", padx=20, pady=6)

        self._divider()

        # ── Output panel ─────────────────────────
        out_frame = tk.LabelFrame(self.root, text="  📋  OUTPUT  ",
                                  font=("Courier New", 10, "bold"),
                                  fg=ACCENT2, bg=PANEL,
                                  bd=1, relief="flat",
                                  highlightbackground=ACCENT2,
                                  highlightthickness=1)
        out_frame.pack(fill="both", expand=True, padx=20, pady=6)

        self.output = scrolledtext.ScrolledText(
            out_frame,
            font=FONT_MON,
            bg="#0a0a18", fg=ACCENT,
            insertbackground=ACCENT,
            bd=0, relief="flat",
            wrap="word",
            state="disabled",
            height=18,
        )
        self.output.pack(fill="both", expand=True, padx=6, pady=6)

        # colour tags
        self.output.tag_config("header",  foreground=ACCENT,  font=("Courier New", 10, "bold"))
        self.output.tag_config("info",    foreground=TXT)
        self.output.tag_config("key",     foreground=ACCENT2, font=("Courier New", 10, "bold"))
        self.output.tag_config("val",     foreground=GOLD)
        self.output.tag_config("good",    foreground=ACCENT)
        self.output.tag_config("warn",    foreground=RED)
        self.output.tag_config("dim",     foreground=DIM)
        self.output.tag_config("divider", foreground="#333355")

        # status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Courier New", 8), fg=DIM, bg=BG).pack(pady=(0, 8))

    def _divider(self):
        tk.Frame(self.root, bg="#2a2a4a", height=1).pack(fill="x", padx=20, pady=2)

    # ── Output helpers ───────────────────────────
    def _write(self, text, tag="info"):
        self.output.config(state="normal")
        self.output.insert("end", text, tag)
        self.output.see("end")
        self.output.config(state="disabled")

    def _clear(self):
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        self.output.config(state="disabled")

    def _line(self, char="─", n=54, tag="divider"):
        self._write(char * n + "\n", tag)

    # ── Run logic ────────────────────────────────
    def _on_run(self):
        self.run_btn.config(state="disabled", text="⏳  Running...")
        self._clear()
        thread = threading.Thread(target=self._run_model, daemon=True)
        thread.start()

    def _run_model(self):
        model_name = self.selected_model.get()
        pkl_file   = MODELS[model_name]

        self._write(f"Loading dataset and preprocessing...\n", "dim")
        self.status_var.set("Loading data…")

        try:
            # ── load artefacts ─────────────────────
            train_medians   = safe_load("train_medians.pkl")
            robust_scaler   = safe_load("robust_scaler.pkl")
            std_scaler      = safe_load("standard_scaler.pkl")
            svm_scaler      = safe_load("svm_scaler.pkl")
            final_features  = safe_load("final_features_list.pkl")
            target_mapping  = safe_load("target_mapping.pkl")
            target_fill     = safe_load("target_fill_value.pkl")
            date_mode       = safe_load("date_mode.pkl")

            self._write("Preprocessing done\n", "good")

            # ── load model ─────────────────────────
            self._write(f"Training {model_name} (this may take a minute)...\n", "dim")
            self.status_var.set(f"Loading {model_name}…")
            model = safe_load(pkl_file)

            if model is None:
                raise FileNotFoundError(f"Could not load '{pkl_file}'. Make sure the file exists.")

            self._write(f"{model_name} model loaded ✔\n", "good")
            self._line()

            # ── load test data ─────────────────────
            self.status_var.set("Loading test data…")
            test_df = safe_load("X_test_processed.pkl")
            y_test  = safe_load("y_test.pkl")

            if test_df is None or y_test is None:
                # fallback: use train CSV if processed test set not saved
                raise FileNotFoundError(
                    "X_test_processed.pkl or y_test.pkl not found.\n"
                    "Please run your training script first and save:\n"
                    "  joblib.dump(X_test, 'X_test_processed.pkl')\n"
                    "  joblib.dump(y_test, 'y_test.pkl')"
                )

            # ── predict ────────────────────────────
            self.status_var.set("Predicting…")

            if model_name == "XGBoost":
                if isinstance(model, xgb.Booster):
                    dtest = xgb.DMatrix(test_df)
                    raw   = model.predict(dtest)
                    preds = np.argmax(raw, axis=1) if raw.ndim == 2 else raw.astype(int)
                else:
                    preds = model.predict(test_df)
            elif model_name == "KNN":
                from sklearn.decomposition import PCA
                pca   = safe_load("knn_pca.pkl") or PCA(n_components=min(80, test_df.shape[1]))
                X_knn = pca.transform(test_df) if hasattr(pca, "components_") else test_df
                preds = model.predict(X_knn)
            elif model_name == "SVM":
                X_svm = svm_scaler.transform(test_df) if svm_scaler else test_df
                preds = model.predict(X_svm)
            elif model_name in ("CatBoost",):
                X_cat = test_df.select_dtypes(include=[np.number])
                preds = model.predict(X_cat).flatten()
            else:
                preds = model.predict(test_df)

            # ── metrics ────────────────────────────
            acc    = accuracy_score(y_test, preds)
            f1_w   = f1_score(y_test, preds, average="weighted")
            f1_mac = f1_score(y_test, preds, average="macro")
            report = classification_report(y_test, preds,
                                           target_names=["Low", "Medium", "High"],
                                           digits=4)
            cm     = confusion_matrix(y_test, preds)

            # ── per-class accuracy ──────────────────
            per_class = cm.diagonal() / cm.sum(axis=1)

            # ── print results ──────────────────────
            self._write(f"\nModel        : ", "key"); self._write(f"{model_name}\n", "val")
            self._write(f"Test Accuracy: ", "key"); self._write(f"{acc*100:.2f}%\n", "val")
            self._write(f"F1 (weighted): ", "key"); self._write(f"{f1_w:.4f}\n", "val")
            self._write(f"F1 (macro)   : ", "key"); self._write(f"{f1_mac:.4f}\n", "val")

            self._line()
            self._write("Per-Class Accuracy\n", "header")
            for i, label in CLASS_LABELS.items():
                bar = "█" * int(per_class[i] * 20)
                self._write(f"  {label:<8}: ", "key")
                self._write(f"{per_class[i]*100:5.1f}%  {bar}\n", "val")

            self._line()
            self._write("Classification Report\n", "header")
            self._write(report + "\n", "info")

            self._line()
            self._write("Confusion Matrix\n", "header")
            header = "         " + "  ".join(f"{l:^8}" for l in ["Low","Med","High"])
            self._write(header + "\n", "dim")
            for i, row in enumerate(cm):
                row_str = f"  {CLASS_LABELS[i]:<7} " + "  ".join(f"{v:^8}" for v in row)
                self._write(row_str + "\n", "info")

            self._line()

            # ── status ─────────────────────────────
            if acc >= 0.75:
                status_tag, status_txt = "good",  "✅  Balanced Model"
            elif acc >= 0.60:
                status_tag, status_txt = "warn",  "⚠️  Moderate Model"
            else:
                status_tag, status_txt = "warn",  "❌  Needs Improvement"

            self._write(f"Status       : ", "key")
            self._write(status_txt + "\n", status_tag)
            self._line()
            self._write("DONE ✔\n", "good")
            self.status_var.set("Done ✔")

        except FileNotFoundError as e:
            self._write(f"\n⚠️  File Not Found:\n{e}\n", "warn")
            self.status_var.set("Error – file not found")
        except Exception as e:
            self._write(f"\n❌  Error:\n{str(e)}\n", "warn")
            self.status_var.set("Error – see output")

        finally:
            self.run_btn.config(state="normal", text="▶   RUN MODEL")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = ClassificationGUI(root)
    root.mainloop()