import sys
from pathlib import Path

target = Path(r"d:\3rd sem M.tech\agent_workspace\agent\tools\harness_template.py")
content = target.read_text(encoding="utf-8")

new_step = """
_write_results(result)
print(f"[harness] DONE — val_f1_macro = {val_f1:.4f}")

# ── Step 9: Generate Visualizations ───────────────────────────────────────────
try:
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_val)
    else:
        y_pred_proba = None
    
    from agent.tools.harness_template import generate_plots
    class_names = ["No", "Low", "Medium", "High"]
    generate_plots(result, y_val, y_pred, y_pred_proba, _EXP_DIR, class_names, model=model)
except Exception as e:
    print(f"[harness] Visualization failed: {e}")
'''
"""

if "Step 9: Generate Visualizations" not in content:
    content = content.replace(
        "_write_results(result)\nprint(f\"[harness] DONE — val_f1_macro = {val_f1:.4f}\")\n'''",
        new_step
    )

generate_plots_code = """
def generate_plots(results, y_val, y_pred, y_pred_proba, exp_dir, class_names, model=None):
    import os
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    exp_dir = Path(exp_dir)
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    exp_name = results.get("exp_id", "Unknown")
    f1_score = results.get("val_f1_macro", 0.0)
    
    # 1. confusion_matrix.png
    try:
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6), dpi=150)
        
        # User requested: "show both raw counts and normalized"
        cm_norm = confusion_matrix(y_val, y_pred, normalize='true')
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\\n({cm_norm[i, j]:.1%})"
        
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix - {exp_name}\\nF1: {f1_score:.4f}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Failed to generate confusion_matrix.png: {e}")
        plt.close()

    # 2. per_class_metrics.png
    try:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2, 3], zero_division=0)
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.figure(figsize=(10, 6), dpi=150)
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1')
        
        plt.ylabel('Score')
        plt.title(f'Per-Class Metrics - {exp_name}')
        plt.xticks(x, class_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'per_class_metrics.png')
        plt.close()
    except Exception as e:
        print(f"Failed to generate per_class_metrics.png: {e}")
        plt.close()

    # 3. prediction_distribution.png
    try:
        actual_counts = [sum(y_val == i) for i in range(len(class_names))]
        pred_counts = [sum(y_pred == i) for i in range(len(class_names))]
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.figure(figsize=(8, 6), dpi=150)
        plt.bar(x - width/2, actual_counts, width, label='Actual')
        plt.bar(x + width/2, pred_counts, width, label='Predicted')
        
        plt.ylabel('Count')
        plt.title(f'Actual vs Predicted Distribution - {exp_name}')
        plt.xticks(x, class_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'prediction_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Failed to generate prediction_distribution.png: {e}")
        plt.close()

    # 4. f1_leaderboard_trend.png
    try:
        lb_path = Path("master_log/leaderboard.json")
        if lb_path.exists():
            with open(lb_path, "r") as f:
                lb = json.load(f)
            exps = lb.get("experiments", [])
            exps_sorted = sorted(exps, key=lambda e: e.get("timestamp", ""))
            
            f1s = [e.get("val_f1_macro", 0.0) for e in exps_sorted]
            exp_names = [e.get("exp_id", "") for e in exps_sorted]
            
            plt.figure(figsize=(10, 6), dpi=150)
            plt.plot(range(len(f1s)), f1s, marker='.', linestyle='-', color='gray')
            
            if exp_name in exp_names:
                idx = exp_names.index(exp_name)
                plt.plot(idx, f1s[idx], 'ro', markersize=8, label='Current')
            
            plt.title('Leaderboard Val F1 Macro Trend')
            plt.xlabel('Experiment Index (Chronological)')
            plt.ylabel('F1 Macro')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / 'f1_leaderboard_trend.png')
            plt.close()
    except Exception as e:
        print(f"Failed to generate f1_leaderboard_trend.png: {e}")
        plt.close()

    # 5. family_comparison.png
    try:
        lb_path = Path("master_log/leaderboard.json")
        if lb_path.exists():
            with open(lb_path, "r") as f:
                lb = json.load(f)
            exps = lb.get("experiments", [])
            
            fam_best = {}
            for e in exps:
                fam = e.get("architecture_family", "unknown")
                f1_s = e.get("val_f1_macro", 0.0)
                if fam not in fam_best or f1_s > fam_best[fam]:
                    fam_best[fam] = f1_s
                    
            if exp_name != "Unknown" and results.get("architecture_family"):
                fam = results.get("architecture_family")
                if fam not in fam_best or f1_score > fam_best[fam]:
                    fam_best[fam] = f1_score
            
            fams = list(fam_best.keys())
            scores = list(fam_best.values())
            
            sorted_pairs = sorted(zip(scores, fams))
            if sorted_pairs:
                scores, fams = zip(*sorted_pairs)
                
                plt.figure(figsize=(10, 6), dpi=150)
                colors = ['red' if f == results.get("architecture_family") else 'skyblue' for f in fams]
                plt.barh(fams, scores, color=colors)
                plt.xlabel('Best Val F1 Macro')
                plt.title('Performance by Inferred Type')
                plt.tight_layout()
                plt.savefig(plots_dir / 'family_comparison.png')
                plt.close()
    except Exception as e:
        print(f"Failed to generate family_comparison.png: {e}")
        plt.close()

    # 6. feature_importance.png
    try:
        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]
            
            plt.figure(figsize=(10, 8), dpi=150)
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_importance.png')
            plt.close()
        elif model is not None and hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:
                importances = np.mean(np.abs(coefs), axis=0)
            else:
                importances = np.abs(coefs)
            
            indices = np.argsort(importances)[-20:]
            plt.figure(figsize=(10, 8), dpi=150)
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
            plt.xlabel('Mean Absolute Coefficient')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_importance.png')
            plt.close()
    except Exception as e:
        print(f"Failed to generate feature_importance.png: {e}")
        plt.close()

    # 7. experiment_summary.png
    try:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
        
        # 7.1 Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        cm_norm = confusion_matrix(y_val, y_pred, normalize='true')
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\\n({cm_norm[i, j]:.1%})"
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axs[0, 0])
        axs[0, 0].set_title(f"Confusion Matrix\\nF1: {f1_score:.4f}")
        axs[0, 0].set_ylabel("Actual")
        axs[0, 0].set_xlabel("Predicted")
        
        # 7.2 Per-Class
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2, 3], zero_division=0)
        x = np.arange(len(class_names))
        width = 0.25
        axs[0, 1].bar(x - width, precision, width, label='Precision')
        axs[0, 1].bar(x, recall, width, label='Recall')
        axs[0, 1].bar(x + width, f1, width, label='F1')
        axs[0, 1].set_title('Per-Class Metrics')
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(class_names)
        axs[0, 1].legend()
        
        # 7.3 Distribution
        actual_counts = [sum(y_val == i) for i in range(len(class_names))]
        pred_counts = [sum(y_pred == i) for i in range(len(class_names))]
        axs[1, 0].bar(x - width/2, actual_counts, 0.35, label='Actual')
        axs[1, 0].bar(x + width/2, pred_counts, 0.35, label='Predicted')
        axs[1, 0].set_title('Actual vs Predicted')
        axs[1, 0].set_xticks(x)
        axs[1, 0].set_xticklabels(class_names)
        axs[1, 0].legend()
        
        # 7.4 Trend
        lb_path = Path("master_log/leaderboard.json")
        if lb_path.exists():
            with open(lb_path, "r") as f:
                lb = json.load(f)
            exps = lb.get("experiments", [])
            exps_sorted = sorted(exps, key=lambda e: e.get("timestamp", ""))
            f1s = [e.get("val_f1_macro", 0.0) for e in exps_sorted]
            exp_names = [e.get("exp_id", "") for e in exps_sorted]
            
            axs[1, 1].plot(range(len(f1s)), f1s, marker='.', linestyle='-', color='gray')
            if exp_name in exp_names:
                idx = exp_names.index(exp_name)
                axs[1, 1].plot(idx, f1s[idx], 'ro', markersize=8, label='Current')
            axs[1, 1].set_title('Leaderboard Val F1 Macro Trend')
            axs[1, 1].legend()
            
        fig.suptitle(f'Experiment Summary: {exp_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / 'experiment_summary.png')
        plt.close()
    except Exception as e:
        print(f"Failed to generate experiment_summary.png: {e}")
        plt.close()
"""

if "def generate_plots(" not in content:
    content += "\n" + generate_plots_code

target.write_text(content, encoding="utf-8")
print("Patch applied.")
