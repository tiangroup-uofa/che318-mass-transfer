import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    setup = mo.md("""
    ## Assignment 7 Q1

    In this task, you will see the difference between solving the interfacial composition $(x_i, y_i)$ using the exact formula and the EMCD-like slope $-k_x'/k_y'$. When changing the point of $(x_{{AL}}, y_{{AG}})$,
    when does the interfacial composition solution becomes almost the same?

    **Please change the conditions to the assignment task!**

    - $x_{{AL}}$: {x1}, $y_{{AG}}$: {y1}
    - $k_{{x}}'* 10^3$ kg mol/m$^2$/s: {kx_prime}
    - $k_{{y}}' * 10^3$ kg mol/m$^2$/s: {ky_prime}
    - Display the EMCD-like slope? {show_emcd}
    """).batch(
        x1=mo.ui.slider(
            start=0.0, stop=0.30, step=0.025, show_value=True, value=0.05
        ),
        y1=mo.ui.slider(
            start=0.0, stop=0.50, step=0.025, show_value=True, value=0.25
        ),
        kx_prime=mo.ui.number(step=0.001, value=2.000),
        ky_prime=mo.ui.number(step=0.001, value=2.000),
        show_emcd=mo.ui.switch(value=False),
    )
    setup
    return (setup,)


@app.cell(hide_code=True)
def _(default_curve_func, mo, np, plt, setup, solve_slope):
    def plot_main():
        xx = np.linspace(0.0, 0.35)
        yy = default_curve_func(xx)
        plt.figure(figsize=(6, 5))
        plt.plot(xx, yy, color="grey", linewidth=1.2)
        x1, y1 = setup.value["x1"], setup.value["y1"]

        kx_prime = setup.value["kx_prime"] * 1e-3
        ky_prime = setup.value["ky_prime"] * 1e-3

        # Plot the exact line
        x2, y2, slope = solve_slope(
            x1, y1, kx_prime=kx_prime, ky_prime=ky_prime, use_stagnant=True
        )
        plt.plot((x1, x2), (y1, y2), label=f"Exact Slope={slope:.4f}")
        output = f"- Exact solution: $x_i$: {x2:.4f}, $y_i$: {y2:.4f}, slope: {slope:.4f}"

        if setup.value["show_emcd"]:
            x2_emcd, y2_emcd, slope_emcd = solve_slope(
                x1, y1, kx_prime=kx_prime, ky_prime=ky_prime, use_stagnant=False
            )

            plt.plot(
                (x1, x2_emcd), (y1, y2_emcd), label=f"EMCD Slope={slope_emcd:.4f}"
            )
            output += f"\n- EMCD approx.: $x_i$: {x2_emcd:.4f}, $y_i$: {y2_emcd:.4f}, slope: {slope_emcd:.4f}"

        plt.plot(x1, y1, "o", markersize=6, color="tab:blue")
        plt.text(
            x1, y1 + 0.02, s="$(x_{{AL}}, y_{{AG}})$", va="bottom", ha="center"
        )

        plt.plot(x2, y2, "o", markersize=6, color="tab:blue")
        plt.text(x2, y2 - 0.02, s="$(x_{{Ai}}, y_{{Ai}})$", va="top", ha="center")
        plt.xlabel("Liquid composition $x$")
        plt.ylabel("Liquid composition $y$")
        plt.legend()
        return output, plt.gca()


    output, ax = plot_main()
    mo.vstack([mo.md(output), ax])
    return


@app.cell(hide_code=True)
def _(default_curve_func, fsolve, logmean):
    def solve_slope(
        x,
        y,
        curve_func=default_curve_func,
        kx_prime=1.0,
        ky_prime=1.0,
        use_stagnant=True,
    ):
        """Start from the x, y point find the intercept to the curve that slope follows
        -(k_x' / (1 - x)_im) / (k_y' / (1 - y)_im)
        """

        def slope_loss(x2):
            """Compute the loss
            (y1 - y2) * (ky_prime / (1-y)_im) + (x1 - x2) * (kx_prime / (1-x)_im) = 0
            """
            y2 = curve_func(x2)
            x1 = x
            y1 = y
            if use_stagnant:
                lhs = (y1 - y2) * (ky_prime / logmean((1 - y1), (1 - y2)))
                rhs = (x1 - x2) * (kx_prime / logmean((1 - x1), (1 - x2)))
            else:
                lhs = (y1 - y2) * (ky_prime)
                rhs = (x1 - x2) * (kx_prime)
            return lhs + rhs

        # Add a small displacement to x to prevent overflow
        x2 = fsolve(slope_loss, x0=x + 0.05)[0]
        y2 = curve_func(x2)
        slope = (y - y2) / (x - x2)
        return x2, y2, float(slope)

    return (solve_slope,)


@app.cell(hide_code=True)
def _(UnivariateSpline, np):
    _raw_data = """
    0,0
    0.05,0.022
    0.10,0.052
    0.15,0.087
    0.20,0.131
    0.25,0.187
    0.30,0.265
    0.35,0.385
    """


    def default_curve_func(x):
        """Default equilibrium curve function"""
        # Split the raw data into x and y components
        data_points = _raw_data.strip().split("\n")
        x_data = np.array([float(point.split(",")[0]) for point in data_points])
        y_data = np.array([float(point.split(",")[1]) for point in data_points])

        # Fit a spline to the data
        spline = UnivariateSpline(x_data, y_data, s=0)

        # Evaluate the spline at the given x value
        return spline(x)


    def logmean(a, b):
        """calculate the log-mean of a,b"""
        assert abs(b) != 0
        return (a - b) / np.log(a / b)

    return default_curve_func, logmean


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import fsolve

    return UnivariateSpline, fsolve, np, plt


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
