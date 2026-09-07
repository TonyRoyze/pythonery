"""Helper module: copy ML code snippets to the clipboard.

Usage (in your notebook):
    from codehelper import scaler, split, mlp, metrics, metric_defs

    scaler('std')
    split(test_size=0.30, random_state=20)
    mlp(['opt', 'momentum_nest'])   # extra MLP param lines by code
    metrics(['acc', 'pres', 'rec', 'f1'])
    metric_defs('pres')      # markdown meaning + formula

Each function copies a snippet/string to the clipboard (pbcopy on macOS)
and returns it as a string.
"""

import shutil
import subprocess
import sys


def _copy(text: str) -> None:
    """Copy a string to the clipboard via the first available mechanism."""
    if _copy_colab(text):
        return
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        return
    if sys.platform.startswith("win"):
        subprocess.run(["clip"], input=text, text=True, check=True)
        return

    # Linux: try Wayland, then X11, then pyperclip
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-bi"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, input=text, text=True, check=True)
            return

    try:
        import pyperclip
        pyperclip.copy(text)
    except ImportError:
        raise RuntimeError(
            "No clipboard tool found. Install one via:\n"
            "  sudo apt-get install wl-clipboard   # Wayland\n"
            "  sudo apt-get install xclip xsel     # X11\n"
            "  pip install pyperclip"
        )


def _is_colab() -> bool:
    """Return True when running inside a Google Colab notebook."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        shell = get_ipython()  # type: ignore[name-defined]
        return shell.config.get("IPKernelApp", {}).get("parent_appname") == "colab"
    except Exception:
        return False


def _copy_colab(text: str) -> bool:
    """Copy via browser JavaScript on Google Colab. Returns False if not Colab."""
    if not _is_colab():
        return False

    import json
    try:
        from google.colab import output  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Colab detected but google.colab unavailable: {exc}")

    safe = json.dumps(text)  # proper JS string escaping
    js = (
        "(() => {\n"
        "  const el = document.createElement('textarea');\n"
        f"  el.value = {safe};\n"
        "  el.style.position = 'fixed';\n"
        "  el.style.opacity = '0';\n"
        "  document.body.appendChild(el);\n"
        "  el.focus();\n"
        "  el.select();\n"
        "  let ok = false;\n"
        "  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }\n"
        "  document.body.removeChild(el);\n"
        "  if (!ok) { try { navigator.clipboard.writeText(" + safe + "); } catch (e) {} }\n"
        "  return ok;\n"
        "})()"
    )
    try:
        output.eval_js(js)
    except Exception as exc:
        raise RuntimeError(
            "Could not run clipboard JS in this Colab runtime. "
            f"Underlying error: {exc}"
        )
    return True


def _join_options(values: list) -> str:
    """Turn a list into 'val1' | 'val2' ..."""
    return " | ".join(f"'{v}'" for v in values)


def _option_list(value, valid):
    """Accept a string or list, validate against valid options."""
    if isinstance(value, str):
        value = [value]
    for v in value:
        if v not in valid:
            raise ValueError(f"Invalid option {v!r}. Choose from {valid}")
    return value


def scaler(kind: str = "std") -> None:
    """Copy a scaler snippet. kind in {'std', 'minmax', 'robust'}."""
    scalers = {
        "std": "StandardScaler",
        "minmax": "MinMaxScaler",
        "robust": "RobustScaler",
    }
    if kind.lower() not in scalers:
        raise ValueError(f"Unknown scaler {kind!r}. Choose from {list(scalers)}")
    name = scalers[kind.lower()]
    snippet = (
        f"from sklearn.preprocessing import {name}\n\n"
        f"scaler = {name}()\n"
        "# The scaler should be fitted only using training data\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_test_scaled = scaler.transform(X_test)\n"
    )
    _copy(snippet)


def split(test_size: float = 0.25, random_state: int = 10,
          stratify: bool = False) -> None:
    """Copy a train_test_split snippet.
    test_size is the fraction for TESTING (e.g. 0.30 => 70% train)."""
    strat = "\n    stratify=y" if stratify else ""
    snippet = (
        "from sklearn.model_selection import train_test_split\n\n"
        f"X_train, X_test, y_train, y_test = train_test_split(\n"
        f"    X,\n"
        f"    y,\n"
        f"    test_size={test_size},\n"
        f"    random_state={random_state},{strat}\n"
        ")\n"
    )
    _copy(snippet)


# Union of values across the three model papers (Wine / Digits / Iris)
# Values are pre-formatted strings/expressions as they should appear in code.
COMMON_MLP_PARAMS = {
    "hidden_layer_sizes": ["(8, 4)", "(16, 8, 4)", "(10, 5)"],
    "activation": ["'relu'", "'tanh'"],
    "solver": ["'adam'"],
    "learning_rate_init": ["0.005", "0.001"],
    "max_iter": ["600", "700", "800"],
    "random_state": ["5", "10", "20"],
}

# Additional params inserted via the extras list, keyed by code.
EXTRA_MLP_PARAMS = {
    "opt": "    solver='adam' | 'sgd' | 'lbfgs',\n",
    "batch": "    batch_size='auto',  # 'auto' = min(200, n_samples)\n",
    "alpha": "    alpha=0.0001,  # L2 regularization strength\n",
    "shuffle": "    shuffle=True,\n",
    "tol": "    tol=0.0001,\n",
    "lr": "    learning_rate='constant',  # constant | invscaling | adaptive\n",
    "momentum": "    momentum=0.9,\n",
    "nesterov": "    nesterovs_momentum=True,\n",
    "early": "    early_stopping=True,\n    n_iter_no_change=10,\n",
    "momentum_nest": "    momentum=0.9,\n    nesterovs_momentum=True,\n",
}


def _alt(values: list) -> str:
    """Render a union of values as 'v1' | 'v2' ... (strings/quoted as given)."""
    return " | ".join(values)


def mlp(extras: list | None = None) -> None:
    """Copy one MLP snippet showing the union of common params across the
    three model papers, plus any extra params you choose.

    extras: list of codes that insert additional param lines, e.g.
        mlp(['opt'])        -> also show optimizer alternatives (adam | sgd | lbfgs)
        mlp(['alpha', 'early'])
        Available codes:
            'opt'          -> solver alternatives (adam | sgd | lbfgs)
            'batch'        -> batch_size
            'alpha'        -> alpha (L2 strength)
            'shuffle'      -> shuffle
            'tol'          -> tolerance
            'lr'           -> learning_rate schedule
            'momentum'     -> momentum (SGD)
            'nesterov'     -> nesterovs_momentum
            'momentum_nest'-> momentum + nesterovs
            'early'        -> early_stopping + n_iter_no_change

    Resulting snippet:
        model = MLPClassifier(
            hidden_layer_sizes=(8, 4) | (16, 8, 4) | (10, 5),
            activation='relu' | 'tanh',
            solver='adam',
            learning_rate_init=0.005 | 0.001,
            max_iter=600 | 700 | 800,
            random_state=5 | 10 | 20,
        )
    """
    if extras is None:
        extras = []
    if isinstance(extras, str):
        extras = [extras]
    for e in extras:
        if e not in EXTRA_MLP_PARAMS:
            raise ValueError(
                f"Unknown extra {e!r}. Choose from {sorted(EXTRA_MLP_PARAMS)}"
            )

    order = [
        "hidden_layer_sizes",
        "activation",
        "solver",
        "learning_rate_init",
        "max_iter",
        "random_state",
    ]
    lines = []
    for key in order:
        lines.append(f"    {key}={_alt(COMMON_MLP_PARAMS[key])},\n")
    for e in extras:
        lines.append(EXTRA_MLP_PARAMS[e])

    snippet = (
        "from sklearn.neural_network import MLPClassifier\n\n"
        "model = MLPClassifier(\n"
        + "".join(lines)
        + ")\n\n"
        "model.fit(X_train_scaled, y_train)\n"
    )
    _copy(snippet)


VALID_METRICS = {
    "acc": "accuracy_score",
    "pres": "precision_score",
    "rec": "recall_score",
    "f1": "f1_score",
}


def metrics(metric_list: list | None = None) -> None:
    """Copy a metrics snippet for the requested metrics.
    metric_list: list of codes, e.g. ['acc', 'pres', 'rec', 'f1'].
    Also always includes the confusion matrix plot + print.
    """
    if metric_list is None:
        metric_list = ["acc", "pres"]
    codes = _option_list(metric_list, set(VALID_METRICS))

    imports = (
        "from sklearn.metrics import accuracy_score, precision_score, "
        "recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay\n"
    )

    lines = [
        imports,
        "\ny_pred = model.predict(X_test_scaled)\n",
    ]
    for code in codes:
        func = VALID_METRICS[code]
        avg = "" if code == "acc" else ", average='weighted'"
        lines.append(f"{code} = {func}(y_test, y_pred{avg})\n")
    lines.append("\nprint('Metrics:')\n")
    for code in codes:
        lines.append(f"print('{code}:', {code})\n")
    lines.append(
        "\ncm = confusion_matrix(y_test, y_pred)\n"
        "ConfusionMatrixDisplay(cm).plot()\n"
        "print('Confusion matrix:')\nprint(cm)\n"
    )

    snippet = "".join(lines)
    _copy(snippet)


METRIC_DEFS = {
    "acc": (
        "### Accuracy\n"
        "Proportion of correct predictions out of all predictions.\n"
        "Useful when classes are balanced.\n\n"
        "**Formula:**\n"
        "$$\n"
        "Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}\n"
        "$$\n"
    ),
    "pres": (
        "### Precision\n"
        "Of the samples predicted as positive, how many are truly "
        "positive. Measures exactness; important when false positives are costly.\n\n"
        "**Formula:**\n"
        "$$\n"
        "Precision = \\frac{TP}{TP + FP}\n"
        "$$\n"
    ),
    "rec": (
        "### Recall (Sensitivity)\n"
        "Of the actually positive samples, how many were correctly "
        "found. Measures completeness; important when false negatives are costly.\n\n"
        "**Formula:**\n"
        "$$\n"
        "Recall = \\frac{TP}{TP + FN}\n"
        "$$\n"
    ),
    "f1": (
        "### F1-Score\n"
        "Harmonic mean of precision and recall. Balances both, "
        "useful with imbalanced classes.\n\n"
        "**Formula:**\n"
        "$$\n"
        "F1 = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}\n"
        "$$\n"
    ),
}


def metric_defs(which: str | list | None = None) -> None:
    """Copy markdown meaning + formula of each requested metric.
    which: single code, list of codes, or None for all."""
    if which is None:
        codes = list(METRIC_DEFS)
    else:
        codes = _option_list(which, set(METRIC_DEFS))
    md = "# Metrics: meaning and formula\n\n" + "\n".join(
        METRIC_DEFS[c] for c in codes
    )
    _copy(md)
