import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    setup = mo.md("""
    - Temp range in chart (℃) {T_range}
    - $T_{{L1}}$ (℃) {T1}
    - $H_{{y1}}$ (kJ/kg): {Hy1}
    - $T_{{L2}}$ (℃): {T2}
    - $h_L \\cdot a$ (kJ/m$^3$ s K): {hLa}
    - $k_G \\cdot a \\cdot P$ (kg mol/s m$^3$): {kGaP}
    - Liquid rate $L$ (kg / s): {L}
    - Gas rate $G$ (kg / s): {G}
    - Display diagonal lines? {switch}
    - Points for integral {npts}
    """).batch(
        T_range=mo.ui.range_slider(
            start=10, stop=85, step=5, value=(25, 55), show_value=True
        ),
        T1=mo.ui.number(step=0.1, value=30),
        Hy1=mo.ui.number(step=0.01, value=45.0),
        T2=mo.ui.number(step=0.1, value=50),
        hLa=mo.ui.number(step=0.01, value=14.5),
        kGaP=mo.ui.number(step=1e-4, value=0.012),
        L=mo.ui.number(step=0.005, value=2.0),
        G=mo.ui.number(step=0.005, value=2.0),
        switch=mo.ui.switch(value=False),
        npts=mo.ui.slider(value=5, start=3, stop=25, step=1, show_value=True),
    )
    # setup
    return (setup,)


@app.cell
def _(
    calculate_integral,
    display_enthalpy_plot,
    display_integration,
    display_operating_line,
    mo,
    setup,
):
    setup_value = setup.value
    ax, _eq_line = display_enthalpy_plot(
        T_min=setup_value["T_range"][0], T_max=setup_value["T_range"][1]
    )
    ax, _op_line = display_operating_line(
        ax,
        T_L1=setup_value["T1"],
        H_y1=setup_value["Hy1"],
        T_L2=setup_value["T2"],
        L=setup_value["L"],
        G=setup_value["G"],
        eq_line=_eq_line,
    )
    if setup_value["switch"]:
        ax, integrant = display_integration(
            ax,
            hLa=setup_value["hLa"],
            kGaP=setup_value["kGaP"],
            op_line=_op_line,
            eq_line=_eq_line,
            npts=setup_value["npts"],
        )
        summary = calculate_integral(
            G=setup_value["G"], kGaP=setup_value["kGaP"], integrant=integrant
        )
    else:
        integrant, summary = None, None


    mo.hstack([setup, ax], widths=[1, 2])
    return integrant, summary


@app.cell
def _(integrant, mo, summary):
    if summary:
        disp = mo.vstack([summary, mo.ui.table(integrant)])
    else:
        disp = None
    disp
    return


@app.cell
def _(H_sat, enthalpy, mo, np, operating_line, pd, plt, slope_to_interface):
    def display_enthalpy_plot(T_min=10, T_max=60):
        fig, ax = plt.subplots(figsize=(8, 5))

        T = np.linspace(T_min, T_max, 100)
        Hs = H_sat(T)
        Hy_sat = enthalpy(Hs, T)

        ax.xaxis.set_major_locator(plt.MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(20.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(10.0))
        ax.grid(True, alpha=0.85, which="major", lw=1.0)
        ax.grid(True, alpha=0.5, which="minor", lw=0.75)
        # ax.set_yticks(np.arange(0.01, 0.14001, 0.01))
        ax.plot(T, Hy_sat, lw=1.5, color="0.10")
        ax.text(T[-1] * 1.01, Hy_sat[-1], s="Eq.\nCurve", ha="left", va="center")

        ax.set_xlabel("Liquid temp. $T_L$ (℃)")
        ax.set_ylabel("Gas enthalpy $H_y$ (kJ / kg dry air)")
        ax.set_xlim(T_min, T_max)
        ax.set_ylim(0, Hy_sat[-1] * 1.05)

        fig.tight_layout()

        eq_line = {"T_L": T, "H_y": Hy_sat}
        return ax, eq_line


    def error_box(ax, msg="Error!"):
        ax.text(
            0.25,
            0.75,
            s=msg,
            color="tab:red",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.5),
        )
        return


    def display_operating_line(ax, T_L1, H_y1, T_L2, L, G, eq_line):
        """Display operating line"""

        if not (T_L2 > T_L1):
            error_box(
                ax, msg="Error!\n $T_{{L2}}$ must be greater than $T_{{L1}}$"
            )
            return ax

        T_range = np.linspace(T_L1, T_L2)
        Hy_op = operating_line(L, G, T_range, T_L1, H_y1)
        ax.axvline(x=T_L2, ls="--", color="tab:red", alpha=0.5)
        ax.plot(
            [T_range[0], T_range[-1]],
            [Hy_op[0], Hy_op[-1]],
            "o",
            markersize=8,
            color="tab:red",
        )
        ax.plot(T_range, Hy_op, color="tab:red")

        # Check if operating line is higher than equilibrium line at any point
        Hy_eq_at_T_range = np.interp(T_range, eq_line["T_L"], eq_line["H_y"])
        if np.any(Hy_op > Hy_eq_at_T_range):
            error_box(ax, msg="Error!\n$G$ is too low")

        # Calculate slope
        op_line = {"T_L": T_range, "H_y": Hy_op}

        return ax, op_line


    def display_integration(ax, hLa, kGaP, op_line, eq_line, npts=5):
        T_range = op_line["T_L"]
        Hy_op = op_line["H_y"]

        # Select npts from indices
        indices = np.linspace(0, len(T_range) - 1, npts, dtype=int)
        T_range_pts = np.linspace(T_range[0], T_range[-1], npts)
        Hy_op_pts = np.interp(T_range_pts, T_range, Hy_op)
        T_Li, H_yi = slope_to_interface(
            hLa, kGaP, T_L=T_range_pts, H_y=Hy_op_pts, eq_line=eq_line
        )

        for T_Li_, H_yi_, T_L_, H_y_ in zip(T_Li, H_yi, T_range_pts, Hy_op_pts):
            ax.plot(
                [T_Li_, T_L_], [H_yi_, H_y_], ls="--", color="grey", alpha=0.75
            )

        # Finally calculate the integrant
        integrant_data = {
            "T_L": T_range_pts,
            "H_y": Hy_op_pts,
            "H_yi": H_yi,
            "1/(H_yi - H_y)": 1 / (H_yi - Hy_op_pts),
        }
        integrant_df = pd.DataFrame(integrant_data)

        return ax, integrant_df


    def calculate_integral(G, kGaP, integrant):
        MB = 28.97
        factor = G / kGaP / MB
        integral_value = np.trapezoid(
            integrant["1/(H_yi - H_y)"], integrant["H_y"]
        )
        print(integral_value)
        Z = factor * integral_value
        summary = mo.md(f"""
        Tower height $Z={Z:.2f}$ m, Transfer unit $H_G={factor:.2f}$ m
        """)
        return summary

    return (
        calculate_integral,
        display_enthalpy_plot,
        display_integration,
        display_operating_line,
    )


@app.cell
def _(UnivariateSpline, fsolve, np):
    ## Duplicated code from L29 psychrometric chart, probably can merge into one

    _T_C = np.array([0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    _P_kPa = np.array(
        [
            0.611,
            1.228,
            2.338,
            3.168,
            4.242,
            7.375,
            12.333,
            19.92,
            31.16,
            47.34,
            70.10,
            101.325,
        ],
        dtype=float,
    )


    def pvap_water(T):
        """Get the vapour pressure of water by T
        T is celsius
        """
        spline = UnivariateSpline(_T_C, _P_kPa, s=0)
        return spline(T)


    def H_abs(p, pT=101.325):
        """Give absolute humidity H as p in kPa"""
        return 0.622 * p / (pT - p)


    def H_sat(T, pT=101.325):
        return H_abs(p=pvap_water(T), pT=pT)


    def cs(H):
        """Return humid heat in kJ / kg / K"""
        return 1.005 + 1.88 * H


    def enthalpy(H, T):
        """Return enthalpy of gas in kJ / kg"""
        return cs(H) * T + 2501.4 * H


    def operating_line(L, G, T_L, T_L1, H_y1):
        """Return points of Hy on an operating line
        unit kJ/kg

        L and G are in kg / s
        """
        cL = 4.187
        Hy = L * cL / G * (T_L - T_L1) + H_y1
        return Hy


    def slope_to_interface(hLa, kGaP, T_L, H_y, eq_line):
        """Get the interfacial enthalpy"""
        M_B = 28.97
        slope = -hLa / kGaP / M_B

        def loss_(T):
            H_yeq = np.interp(T, eq_line["T_L"], eq_line["H_y"])
            return (H_yeq - H_y) - (T - T_L) * slope

        T_Li = fsolve(loss_, x0=T_L)
        H_yi = np.interp(T_Li, eq_line["T_L"], eq_line["H_y"])
        return T_Li, H_yi

    return H_sat, enthalpy, operating_line, slope_to_interface


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    import pandas as pd
    from scipy.optimize import fsolve

    return UnivariateSpline, fsolve, np, pd, plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
