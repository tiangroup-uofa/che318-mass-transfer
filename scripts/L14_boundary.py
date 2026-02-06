# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    # UI elements for interactive controls
    K_slider = mo.ui.slider(
        0.1,
        10.0,
        value=1.0,
        step=0.1,
        label="Equilibrium constant $K=c_2/c_1$",
        show_value=True,
    )
    k1k2_slider = mo.ui.slider(
        0.1, 10.0, value=2.0, step=0.1, label="$k_2 / k_1$ (abs.)", show_value=True
    )
    # k2_slider = mo.ui.slider(0.1, 10.0, value=1.0, step=0.1, label="k2 (phase 2 mass transfer coefficient)", show_value=True)
    c1b_slider = mo.ui.slider(
        0.0,
        1.0,
        value=1.0,
        step=0.05,
        label="$c_{1,b}$ (abs.)",
        show_value=True,
    )
    c2b_slider = mo.ui.slider(
        0.0,
        1.0,
        value=0.0,
        step=0.05,
        label="$c_{2,b}$ (abs.)",
        show_value=True,
    )
    return K_slider, c1b_slider, c2b_slider, k1k2_slider


@app.cell
def _(K_slider, c1b_slider, c2b_slider, k1k2_slider, mo, np, plt):
    # Extract slider values
    K = K_slider.value
    k1 = 1
    k2 = k1k2_slider.value * k1
    c1b = c1b_slider.value
    c2b = c2b_slider.value


    c1i = (k1 * c1b + k2 * c2b) / (k1 + K * k2 + 1e-8)
    c2i = K * c1i

    # For visualization, create exponential profiles from bulk to interface
    x1 = np.linspace(0, 1, 100)  # phase 1: 0 (bulk) to 1 (interface)
    x2 = np.linspace(1, 2, 100)  # phase 2: 1 (interface) to 2 (bulk)

    # Exponential approach to interface (arbitrary decay rate for illustration)
    lambda1 = 6.0
    lambda2 = 6.0
    # c1_profile = c1b - (-c1i + c1b) * (np.exp(-lambda1 * (1 - x1))) / (
    #     1 - np.exp(-lambda1)
    # )
    # c2_profile = c2i + (c2b - c2i) * (1 - np.exp(-lambda2 * (x2 - 1))) / (
    #     1 - np.exp(-lambda2)
    # )

    c1_profile = c1b + (c1i - c1b) * (np.exp((1 - x1) * -lambda1))
    c2_profile = c2b + (c2i - c2b) * (np.exp((x2 - 1) * -lambda1))

    plt.figure(figsize=(6, 4))
    plt.plot(x1, c1_profile, label="Phase 1 (left)", color="tab:blue")
    plt.plot(x2, c2_profile, label="Phase 2 (right)", color="tab:orange")
    plt.scatter([1], [c1i], color="tab:blue", marker="o", s=80)
    plt.scatter([1], [c2i], color="tab:orange", marker="o", s=80)
    plt.scatter([0], [c1b], color="tab:blue", marker="s", s=80)
    plt.scatter([2], [c2b], color="tab:orange", marker="s", s=80)
    plt.axvline(1, color="gray", linestyle="--", alpha=0.5)
    plt.axvspan(-0.05, 1.0, color="tab:blue", alpha=0.1)
    plt.axvspan(1.0, 2.05, color="tab:orange", alpha=0.1)
    plt.xlim(-0.05, 2.05)
    plt.ylim(0, 2)
    plt.xlabel("Position (abs. unit)")
    plt.ylabel("Concentration")
    plt.title(f"2-Phase Mass Transfer\nK={K:.2f}, k1={k1:.2f}, k2={k2:.2f}")
    plt.legend(loc="best")
    plt.tight_layout()
    _ax = plt.gca()

    _md = mo.md(r"""The interface concentrations $c_{1,i}$ and $c_{2,i}$ are solved from:

    $$
    K = \frac{c_{2,i}}{c_{1,i}}\\
    k_1 (c_{1,b} - c_{1,1}) = k_2 (c_{2,i} - c_{2,b})
    $$""")

    mo.vstack(
        [
            _md,
            mo.hstack([K_slider, k1k2_slider], wrap=True, justify="start"),
            mo.hstack([c1b_slider, c2b_slider], wrap=True, justify="start"),
            _ax,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
