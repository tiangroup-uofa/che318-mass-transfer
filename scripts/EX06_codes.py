# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    md1 = mo.md(
    r"""# Minimal Operating Flow Rate In Absorption Tower

    In this demo you will see how to get flow rate of pure liquid, $L'$ affects the operating line and outlet concentrations. The governing equation is 

    $$
    L' \dfrac{x_2}{1 - x_2} + V' \dfrac{y_1}{1 - y_1}
    =
    L' \dfrac{x_1}{1 - x_1} + V' \dfrac{y_2}{1 - y_2}
    $$
    """
    )


    setup = mo.md("""
    ## Setup the system
    1. Choose the parameters from below
    - Gas inlet flow rate $V_1$: {v1} kg mol/s
    - Gas inlet $y_1$: {y1} 
    - Gas outlet $y_2$: {y2} 
    - Liq. inlet $x_2$: {x2}

    2. Set the $L'_{{\\text{{min}}}}$: {L_prime} (kg mol/h). 

    When the value is correct, you should see that the intersect 
    between operating line and equilibrium line is at $y=y_1$.
    """
    ).batch(
        v1=mo.ui.number(value=181.4, step=0.1, ),
        y1=mo.ui.number(value=0.55, step=0.01),
        y2=mo.ui.number(value=0.02, step=0.001),
        x2=mo.ui.number(value=0.0001, step=1e-4, ),
        L_prime=mo.ui.number(value=200, step=0.01)
    )


    return md1, setup


@app.cell
def _(np, root, setup):
    def get_operating_line_y(x, L_prime=None, setups=setup.value):
        """The operating line in this case has outlet (x2, y2) fixed
        """
        if L_prime is None:
            L_prime = setups["L_prime"]
        y1 = setups["y1"]
        V_prime = setups["v1"] * (1 - y1)
        x2 = setups["x2"]
        y2 = setups["y2"]
        factor = (L_prime * x / (1 - x) + V_prime * y2 / (1 - y2) - L_prime * x2 / (1 - x2)) / V_prime
        y = factor / (1 + factor)
        return y
    
    def operating_line(L_prime=None, setups=setup.value):
        """Return the operating line x, y
        """
        x2, y2 = setups["x2"], setups["y2"]
        y1 = setups["y1"]
        x1 = root(lambda x: get_operating_line_y(x, L_prime=L_prime, setups=setups) - y1, 
                  x0=x2).x[0]
        xx = np.linspace(x2, x1, 100)
        yy = get_operating_line_y(xx, L_prime, setups)
        return xx, yy

    return (operating_line,)


@app.cell
def _(np, operating_line, plt, setup, y_eq_303K):
    def plot_system():
        # test l prime

        setups = setup.value
        x2, y2 = setups["x2"], setups["y2"]
        y1 = setups["y1"]
        x_eq = np.linspace(0, 0.30, 100)
        #y_eq = setups["m"] * x_eq
        y_eq = y_eq_303K(x_eq)
    
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.plot(x_eq, y_eq, color="grey", label="Equlibrium line", linewidth=1.5)

        L_prime = setups["L_prime"]
        x_op, y_op = operating_line(L_prime, setups)
        x_op_1_5, y_op_1_5 = operating_line(1.5 * L_prime, setups)

        y_eq_at_x_op = y_eq_303K(x_op[-1])

        # The operating line intersects with the eq line!
        if y_eq_at_x_op > y_op[-1] + 1e-4:
            ax.text(x=0.5, y=0.5, 
                    s="Error! Op. line crossing eq. line!\nToo low $L'$", 
                    ha="center",
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.85),
                    transform=ax.transAxes,
                    color="red")
        if y_eq_at_x_op < y_op[-1] - 0.01:
            ax.text(x=0.5, y=0.5, 
                    s="Warning!\n$L'_{{min}}$ too high!", 
                    ha="center",
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.85),
                    transform=ax.transAxes,
                    color="red")
        
        ax.text(x=0.01, y=y1 + 0.01, s="$y=y_1$ line", color="tab:green")
        ax.axhline(y=y1, ls="--", color="tab:green")

        ax.set_ylim(0.0, 0.6)
        ax.plot(x_op, y_op, color="tab:blue", label=r"Operating line $L'_{\text{min}}$")
        ax.plot(x_op_1_5, y_op_1_5, color="tab:orange", label=r"Operating line $1.5L'_{\text{min}}$")
        ax.plot(x2, y2, "o", color="grey")
    
    
        # plt.plot(x_vals, y_min_line, label="Operating line (L'min)", color='red', linestyle='--')
        # plt.plot(x_vals, y_actual_line, label="Operating line (L' = 1.5 L'min)", color='blue')
        # plt.plot(x_vals, y_eq, label="Equilibrium (y = m x)", color='green')
        # plt.scatter([x2, x1min, x1], [y1, y2, m*x1], color=['black', 'purple', 'orange'],
        #             label=["Inlet liquid (x₂, y₁)", "x₁,min, y₂", "x₁ (1.5 L'min), y₁"])
        ax.set_xlabel("x (liquid phase")
        ax.set_ylabel("y (gas phase)")
        ax.set_title("Absorption Tower Operating and Equilibrium Lines")
        ax.legend()
        ax.grid(True)
        return ax

    ax = plot_system()
    return (ax,)


@app.cell
def _(ax, md1, mo, setup):
    mo.vstack([md1, mo.hstack([setup, ax], widths=[1, 1])])
    return


@app.cell
def _(np):
    # 303 K equilibrium data for NH3 in water
    # From Geankoplis book Appendix A3-22
    _x_data = np.array([
        0.0000,
        0.0126,
        0.0167,
        0.0208,
        0.0258,
        0.0309,
        0.0405,
        0.0503,
        0.0737,
        0.0960,
        0.1370,
        0.1750,
        0.2100,
        0.2410,
        0.2970
    ])

    _y_data = np.array([
        0.0000,
        0.0151,
        0.0201,
        0.0254,
        0.0321,
        0.0390,
        0.0527,
        0.0671,
        0.1050,
        0.1450,
        0.2350,
        0.3420,
        0.4630,
        0.5970,
        0.9450
    ])

    def y_eq_303K(x):
        """
        Return equilibrium vapor mole fraction y for NH3
        at 303 K from liquid mole fraction x, using linear interpolation.
        """
        x = np.asarray(x)
        return np.interp(x, _x_data, _y_data)

    def x_eq_303K(y):
        """Get inverse relation 
        """
        y = np.asarray(y)
        return np.interp(y, _y_data, _x_data)


    return (y_eq_303K,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import root

    return mo, np, plt, root


if __name__ == "__main__":
    app.run()
