import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    setup = mo.md("""
    ## Assignment 7 Q2

    In this task, you will see how to compute the tower height $Z$ using the numerical methods. The following parameters are taken from the textbook example 10.7-1. **Please change the conditions to the assignment task!**

    - $x_{{2}}$: {x2};   $y_{{2}}$: {y2}
    - $x_{{1}}$: To be solved;   $y_{{1}}$: {y1}
    - $M_A$ kg / kg mol: {M_A}
    - $S$ m$^2$: {S}
    - $V'$ kg mol/s: {V_prime}
    - $L'$ kg mol/s: {L_prime}
    - Number of points on Op. Line: {npts}
    """).batch(
        x2=mo.ui.number(
            start=0.0, stop=0.01, step=0.001, value=0.00
        ),
        y2=mo.ui.number(
            start=0.0, stop=0.2, step=0.01, value=0.02
        ),
        y1=mo.ui.number(
            start=0.0, stop=1.0, step=0.01, value=0.20
        ),
        V_prime=mo.ui.number(
            start=0.0, stop=1.0, step=1e-6, value=6.53e-4
        ),
        L_prime=mo.ui.number(
            start=0.0, stop=1.0, step=1e-6, value=4.20e-2
        ),
        M_A=mo.ui.number(start=0.0, step=0.01, value=64.07),
        S=mo.ui.number(start=0.0, step=1e-4, value=0.0929),
        npts=mo.ui.slider(start=5, stop=50, value=5, step=1, show_value=True)
    )
    setup
    return (setup,)


@app.cell(hide_code=True)
def _(ax):
    ax
    return


@app.cell(hide_code=True)
def _(columns, mo, pd, summary):
    df = pd.DataFrame(data=columns)


    csv_download = mo.download(
        data=df.to_csv().encode("utf-8"),
        filename="data.csv",
        mimetype="text/csv",
        label="Download CSV",
    )

    mo.vstack([
        summary,
        mo.ui.table(df),
        csv_download,
    ])
    return


@app.cell(hide_code=True)
def _(
    calculate_Gx_Gy,
    calculate_kxa_kya,
    default_curve_func,
    get_x_from_curve,
    logmean,
    mo,
    np,
    operating_line,
    plt,
    setup,
    solve_slope,
):
    def plot_main():
        xx = np.linspace(0.0, 0.027)
        yy = default_curve_func(xx)
        npts = setup.value["npts"]
        middle = npts // 2
        plt.figure(figsize=(6, 5))
        plt.plot(xx, yy, color="grey")
        plt.text(xx[10], yy[10] - 0.005, s="Eq. Line", ha="left")

        # Tower top
        x2, y2 = setup.value["x2"], setup.value["y2"]
        plt.text(x2, y2, s="$(x_{{2}}, y_{{2}})$ \n Tower top", va="top", ha="center")
        plt.plot(x2, y2, "o", markersize=6, color="tab:blue")

        y1 = setup.value["y1"]
        x1_eq = get_x_from_curve(y1)
        # Tower bottom y1 fixed
        plt.axhline(y=y1, ls="--", color="tab:green")
        plt.text(x=1e-4, y=y1 + 0.01, s="$y=y_1$ line", color="tab:green")

        plt.xlim(-1e-3, x1_eq + 0.001)
        plt.ylim(-1e-2, y1 + 0.1)

        ## Finish setting up

        # Plot the current operating line and get current x1
        V_prime, L_prime = setup.value["V_prime"], setup.value["L_prime"]
        op_xx, op_yy = operating_line(x2, y2, y1, npts=npts, V_prime=V_prime, L_prime=L_prime)
        plt.plot(op_xx, op_yy)
        plt.plot(op_xx[-1], op_yy[-1], "o", markersize=6, color="tab:blue")
        x1 = op_xx[-1]
        print(x1)


        if default_curve_func(x1) > op_yy[-1] + 1e-4:
            plt.text(x=0.5, y=0.5, 
                    s="Error! Op. line crossing eq. line!\nToo low $L'$", 
                    ha="center",
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.85),
                    transform=plt.gca().transAxes,
                    color="red")
        else:
            # Only when in the allowed region will we plot the labels
            plt.text(op_xx[-1], op_yy[-1] + 0.01, s="$(x_{{1}}, y_{{1}})$ \n Tower bottom", va="bottom", ha="center")
            plt.text(op_xx[middle] - 0.0002, op_yy[middle] + 0.005, s="Op. Line", ha="right", color="tab:blue")

        ## Once the student knows how to get the L' and V', then can output the arrays

        Gx, Gy = calculate_Gx_Gy(L_prime, V_prime, op_xx, op_yy, S=setup.value["S"], M_A=setup.value["M_A"])
        # print(Gx, Gy)
        kxa, kya = calculate_kxa_kya(Gx, Gy)
        #print(kxa, kya)

        xi, yi, slope = [], [], []
        for x_, y_, kxa_, kya_ in zip(op_xx, op_yy, kxa, kya):
            xi_, yi_, slope_ = solve_slope(x_, y_, curve_func=default_curve_func, kx_prime=kxa_, ky_prime=kya_, use_stagnant=True)
            plt.plot((x_, xi_), (y_, yi_), ls="--", color="grey", alpha=0.5)
            xi.append(xi_)
            yi.append(yi_)
            slope.append(slope_)
        xi = np.array(xi)
        yi = np.array(yi)
        slope = np.array(slope)
        one_minus_y = 1 - op_yy
        one_minus_y_im = logmean((1 - op_yy), (1 - yi))
        y_minus_yi = op_yy - yi

        f = V_prime / kya / setup.value["S"] * one_minus_y_im / one_minus_y ** 2 / y_minus_yi
        Z = np.trapezoid(f, op_yy)

        columns = {
            "i": list(range(1, len(op_xx) + 1)),
            "x": op_xx.tolist(),
            "y": op_yy.tolist(),
            "Gx": Gx.tolist(),
            "Gy": Gy.tolist(),
            "kxa": kxa.tolist(),
            "kya": kya.tolist(),
            "x_i": xi.tolist(),
            "y_i": yi.tolist(),
            "slope": slope.tolist(),
            "1_minus_y": one_minus_y.tolist(),
            "1_minus_y_im": one_minus_y_im.tolist(),
            "y_minus_yi": y_minus_yi.tolist(),
            # "integrand": integrand.tolist(),
        }

        summary = mo.md(f"""
    ## Results for tower height calculation

    - Number of points: {npts}
    - Tower height: $Z = {Z:.4f}$ m

    Detailed data see below
    """)
    
        plt.xlabel("Liquid composition $x$")
        plt.ylabel("Gas composition $y$")
        plt.legend()
        ax = plt.gca()
        return summary, columns, ax


    summary, columns, ax = plot_main()
    return ax, columns, summary


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
        # For this system we need to be really small
        x2 = fsolve(slope_loss, x0=x + 0.0005)[0]
        y2 = curve_func(x2)
        slope = (y - y2) / (x - x2)
        return x2, y2, float(slope)

    return (solve_slope,)


@app.cell(hide_code=True)
def _(UnivariateSpline, fsolve, np):
    # SO2-H2O data from Appendix A3-19 293K
    _raw_data = """
    0,0
    0.0000562,0.000658
    0.0001403,0.00158
    0.0002800,0.00421
    0.0004220,0.00763
    0.0005640,0.01120
    0.0008420,0.01855
    0.0014030,0.0342
    0.0019650,0.0513
    0.0027900,0.0775
    0.0042000,0.121
    0.0069800,0.212
    0.0138500,0.443
    0.0206000,0.682
    0.0273000,0.917
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

    def get_x_from_curve(y, func=default_curve_func):
        return fsolve(lambda x: default_curve_func(x) - y, x0=0.00)[0]

    def logmean(a, b):
        """calculate the log-mean of a,b"""
        assert np.all(np.abs(b) != 0)
        return (a - b) / np.log(a / b)

    M_air = 29
    M_water = 18
    _default_M_A = 64.07
    _default_S = 0.10
    _default_V_prime = 8.0e-4

    # Solving the weight rate Gx and Gy
    # Gy = (Mair V’ + MA Vy) / S            Gx = (Mwater L’ + MA Lx) / S
    # ky' a = 0.0594 Gy0.7Gx0.25            kx' a = 0.152 Gx0.82


    def calculate_Gx_Gy(L_prime, V_prime, x, y, S=_default_S, M_A=_default_M_A):
        """Calculate the weight rates Gx and Gy in kg/m^2/s
        output is a tupel (Gx, Gy)
        """
        Vy = V_prime / (1 - y) * y
        Lx = L_prime / (1 -x) * x
        Gy = (M_air * V_prime + M_A * Vy) / S
        Gx = (M_water * L_prime + M_A * Lx) / S
        return Gx, Gy

    def calculate_kxa_kya(Gx, Gy):
        kya = 0.0594 * Gy ** 0.7 * Gx ** 0.25
        kxa = 0.152 * Gx ** 0.82
        return kxa, kya

    def get_operating_line_y(x, x2, y2, L_prime, V_prime):
        """The operating line in this case has outlet (x2, y2) fixed

        give an input x in ndarray return y in ndarray
        """
        factor = (L_prime * x / (1 - x) + V_prime * y2 / (1 - y2) - L_prime * x2 / (1 - x2)) / V_prime
        y = factor / (1 + factor)
        return y

    def operating_line(x2, y2, y1, npts=10, V_prime=_default_V_prime, L_prime=None):
        """Return the operating line xx, yy
        """
        x1 = fsolve(lambda x: get_operating_line_y(x, x2=x2, y2=y2, L_prime=L_prime, V_prime=V_prime) - y1, x0=x2 * 1.20)[0]
        xx = np.linspace(x2, x1, npts)
        yy = get_operating_line_y(xx, x2, y2, L_prime=L_prime, V_prime=V_prime)
        return xx, yy

    return (
        calculate_Gx_Gy,
        calculate_kxa_kya,
        default_curve_func,
        get_x_from_curve,
        logmean,
        operating_line,
    )


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import fsolve

    return UnivariateSpline, fsolve, np, pd, plt


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
