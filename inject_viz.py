from pathlib import Path

viz_code = """
    # ── Visualization ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json as _json
        from pathlib import Path as _Path
        from sklearn.metrics import confusion_matrix as _cm
        from sklearn.metrics import precision_recall_fscore_support as _prf

        _class_names = ['No', 'Low', 'Medium', 'High']
        _plots_dir = _Path(__file__).parent / 'plots'
        _plots_dir.mkdir(exist_ok=True)

        # 1. Confusion matrix
        try:
            _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
            _cm_raw = _cm(y_val, y_pred)
            _cm_norm = _cm_raw.astype(float) / (_cm_raw.sum(axis=1, keepdims=True) + 1e-9)
            sns.heatmap(_cm_raw, annot=True, fmt='d', ax=_axes[0],
                       xticklabels=_class_names, yticklabels=_class_names, cmap='Blues')
            _axes[0].set_title('Confusion Matrix (raw)')
            sns.heatmap(_cm_norm, annot=True, fmt='.2f', ax=_axes[1],
                       xticklabels=_class_names, yticklabels=_class_names, cmap='Blues')
            _axes[1].set_title(f'Confusion Matrix (normalized) | F1={{val_f1:.4f}}')
            plt.tight_layout()
            plt.savefig(_plots_dir / 'confusion_matrix.png', dpi=150)
            plt.close()
        except Exception as _e:
            print(f'[viz] confusion_matrix failed: {{_e}}')

        # 2. Per-class metrics
        try:
            _p, _r, _f, _ = _prf(y_val, y_pred, labels=[0,1,2,3],
                                   average=None, zero_division=0)
            _x = range(len(_class_names))
            _fig, _ax = plt.subplots(figsize=(10, 5))
            _w = 0.25
            _ax.bar([i-_w for i in _x], _p, _w, label='Precision')
            _ax.bar([i    for i in _x], _r, _w, label='Recall')
            _ax.bar([i+_w for i in _x], _f, _w, label='F1')
            _ax.set_xticks(list(_x)); _ax.set_xticklabels(_class_names)
            _ax.set_ylim(0, 1); _ax.legend(); _ax.set_title('Per-Class Metrics')
            plt.tight_layout()
            plt.savefig(_plots_dir / 'per_class_metrics.png', dpi=150)
            plt.close()
        except Exception as _e:
            print(f'[viz] per_class_metrics failed: {{_e}}')

        # 3. Prediction distribution
        try:
            import numpy as _np
            _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
            _actual_counts = [_np.sum(y_val == i) for i in range(4)]
            _pred_counts   = [_np.sum(y_pred == i) for i in range(4)]
            _axes[0].bar(_class_names, _actual_counts, color='steelblue')
            _axes[0].set_title('Actual Distribution')
            _axes[1].bar(_class_names, _pred_counts, color='coral')
            _axes[1].set_title('Predicted Distribution')
            plt.tight_layout()
            plt.savefig(_plots_dir / 'prediction_distribution.png', dpi=150)
            plt.close()
        except Exception as _e:
            print(f'[viz] prediction_distribution failed: {{_e}}')

        # 4. Leaderboard trend
        try:
            _lb_path = _Path(__file__).parent.parent.parent / 'master_log' / 'leaderboard.json'
            if _lb_path.exists():
                _lb = _json.load(open(_lb_path))
                _exps = _lb.get('experiments', [])
                _f1s  = [e.get('val_f1_macro', 0) for e in _exps]
                _fig, _ax = plt.subplots(figsize=(12, 4))
                _ax.plot(_f1s, marker='o', linewidth=1.5, markersize=4)
                if _f1s:
                    _ax.plot(len(_f1s)-1, _f1s[-1], 'ro', markersize=10,
                            label=f'Current: {{_f1s[-1]:.4f}}')
                _ax.set_xlabel('Experiment'); _ax.set_ylabel('F1 Macro')
                _ax.set_title('F1 Trend Across All Experiments')
                _ax.legend(); plt.tight_layout()
                plt.savefig(_plots_dir / 'f1_leaderboard_trend.png', dpi=150)
                plt.close()
        except Exception as _e:
            print(f'[viz] f1_leaderboard_trend failed: {{_e}}')

        # 5. Family comparison
        try:
            if _lb_path.exists():
                _lb = _json.load(open(_lb_path))
                _fam_best = {{}}
                for _e in _lb.get('experiments', []):
                    _fam = _e.get('inferred_type', _e.get('architecture_family', 'other'))
                    _f1v = _e.get('val_f1_macro', 0)
                    if _fam not in _fam_best or _f1v > _fam_best[_fam]:
                        _fam_best[_fam] = _f1v
                _fams = list(_fam_best.keys())
                _vals = [_fam_best[f] for f in _fams]
                _fig, _ax = plt.subplots(figsize=(10, 5))
                _colors = ['red' if f == result.get('inferred_type','') 
                          else 'steelblue' for f in _fams]
                _ax.barh(_fams, _vals, color=_colors)
                _ax.set_xlabel('Best F1 Macro')
                _ax.set_title('Best F1 per Architecture Family')
                plt.tight_layout()
                plt.savefig(_plots_dir / 'family_comparison.png', dpi=150)
                plt.close()
        except Exception as _e:
            print(f'[viz] family_comparison failed: {{_e}}')

        # 6. Feature importance (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                import numpy as _np
                _imp = model.feature_importances_
                _idx = _np.argsort(_imp)[-20:]
                _fig, _ax = plt.subplots(figsize=(10, 8))
                _ax.barh(range(len(_idx)), _imp[_idx])
                _ax.set_title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(_plots_dir / 'feature_importance.png', dpi=150)
                plt.close()
        except Exception as _e:
            print(f'[viz] feature_importance failed: {{_e}}')

        # 7. Summary 2x2
        try:
            _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
            # confusion matrix
            _cm_raw = _cm(y_val, y_pred)
            sns.heatmap(_cm_raw, annot=True, fmt='d', ax=_axes[0,0],
                       xticklabels=_class_names, yticklabels=_class_names, cmap='Blues')
            _axes[0,0].set_title('Confusion Matrix')
            # per class metrics
            _p, _r, _f, _ = _prf(y_val, y_pred, labels=[0,1,2,3],
                                   average=None, zero_division=0)
            _x = range(4); _w = 0.25
            _axes[0,1].bar([i-_w for i in _x], _p, _w, label='P')
            _axes[0,1].bar([i    for i in _x], _r, _w, label='R')
            _axes[0,1].bar([i+_w for i in _x], _f, _w, label='F1')
            _axes[0,1].set_xticks(list(_x))
            _axes[0,1].set_xticklabels(_class_names)
            _axes[0,1].legend(); _axes[0,1].set_title('Per-Class Metrics')
            # prediction distribution
            _actual_counts = [_np.sum(y_val == i) for i in range(4)]
            _pred_counts   = [_np.sum(y_pred == i) for i in range(4)]
            _x2 = range(4); _w2 = 0.35
            _axes[1,0].bar([i-_w2/2 for i in _x2], _actual_counts, _w2, label='Actual')
            _axes[1,0].bar([i+_w2/2 for i in _x2], _pred_counts,   _w2, label='Predicted')
            _axes[1,0].set_xticks(list(_x2))
            _axes[1,0].set_xticklabels(_class_names)
            _axes[1,0].legend(); _axes[1,0].set_title('Prediction Distribution')
            # leaderboard trend
            if _lb_path.exists():
                _lb  = _json.load(open(_lb_path))
                _f1s = [e.get('val_f1_macro', 0) for e in _lb.get('experiments', [])]
                _axes[1,1].plot(_f1s, marker='o', linewidth=1.5, markersize=3)
                if _f1s:
                    _axes[1,1].plot(len(_f1s)-1, _f1s[-1], 'ro', markersize=8)
                _axes[1,1].set_title('F1 Trend')
            _fig.suptitle(f'Experiment Summary | F1={{val_f1:.4f}}', fontsize=14)
            plt.tight_layout()
            plt.savefig(_plots_dir / 'experiment_summary.png', dpi=150)
            plt.close()
        except Exception as _e:
            print(f'[viz] experiment_summary failed: {{_e}}')

        print(f'[viz] plots saved to {{_plots_dir}}')

    except Exception as _e:
        print(f'[viz] visualization skipped: {{_e}}')
"""

# Dedent the block because the outer try has 4 spaces inside HARNESS_CODE, which is 0-indented
import textwrap
viz_code = textwrap.dedent(viz_code)

path = Path('agent/tools/harness_template.py')
text = path.read_text(encoding='utf-8')

# The target line
target = 'print(f"[harness] DONE — val_f1_macro = {{val_f1:.4f}}")\\n'

if target in text:
    new_text = text.replace(target, target + viz_code)
    path.write_text(new_text, encoding='utf-8')
    print('Successfully injected properly escaped viz code!')
else:
    print('Target string not found!')
