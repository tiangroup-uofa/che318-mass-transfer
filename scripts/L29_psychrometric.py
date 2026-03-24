import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    temp_range = mo.ui.range_slider(start=10, stop=100, step=5, value=(10, 60), show_value=True, label="Temperature range in chart (℃)")
    #temp_range
    return (temp_range,)


@app.cell
def _(mo, temp_range):
    T_current = mo.ui.slider(
        start=temp_range.value[0],
        stop=temp_range.value[-1],
        step=0.1,
        show_value=True,
        label="Current temperature (℃)",
    )
    #T_current
    return (T_current,)


@app.cell
def _(H_sat, T_current, mo):
    H_current = mo.ui.slider(
        start=1e-6,
        stop=H_sat(T_current.value),
        step=5e-5,
        show_value=True,
        label="Current humidity $H$",
    )
    #H_current

    show_switch = mo.ui.switch(value=False, label="Display solution?")
    return H_current, show_switch


@app.cell
def _(H_current, T_current, mo, show_switch, temp_range):
    mo.hstack([temp_range, T_current, H_current, show_switch], wrap=True, align="start")
    return


@app.cell
def _(
    H_current,
    T_current,
    display_plot,
    mo,
    show_point,
    show_switch,
    temp_range,
):
    ax = display_plot(T_min=temp_range.value[0], T_max=temp_range.value[1])
    if show_switch.value:
        ax, summary = show_point(ax, T=T_current.value, H=H_current.value)
        display = mo.hstack([summary, ax], widths=[1, 3])
    else:
        display = ax

    display
    return


@app.cell
def _(
    H_sat,
    adiabatic_from_Tw,
    adiabatic_from_enthalpy,
    enthalpy,
    fsolve,
    mo,
    np,
    plt,
    pvap_water,
):
    def display_plot(T_min=10, T_max=60):
        fig, ax = plt.subplots(figsize=(8, 5))

        T = np.linspace(T_min, T_max, 100)
        Hs = H_sat(T)

        ax.xaxis.set_major_locator(plt.MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.005))
        ax.grid(True, alpha=0.85, which="major", lw=1.0)
        ax.grid(True, alpha=0.5, which="minor", lw=0.75)
        # ax.set_yticks(np.arange(0.01, 0.14001, 0.01))
        ax.plot(T, Hs, lw=1.5, color="0.10")
        ax.text(T[-1] * 1.01, Hs[-1], s=r"$H_p = 100$%", ha="left", va="center")

        hp = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        for i_, hp_ in enumerate(hp):
            H_ = Hs * hp_
            ax.plot(
                T,
                H_,
                lw=0.75,
                color="tab:orange",
                alpha=0.65,
                label="Percentage Humidity" if i_ == 0 else None,
            )
            ax.text(
                T_max * 1.01,
                H_[-1],
                s=f"{int(hp_ * 100):0d}%",
                ha="left",
                va="center",
            )

        for i_, T_ in enumerate(np.arange(T_min, T_max + 1, 5)):
            tt = np.linspace(T_, T_max + 10, 100)
            hh = adiabatic_from_Tw(T_, tt)
            ax.plot(
                tt,
                hh,
                color="tab:blue",
                alpha=0.65,
                lw=0.75,
                label="Adiabatic sat. curve" if i_ == 0 else None,
            )
        # ax.set_box_aspect(1)
        ax.set_xlabel("$T$ (℃)")
        ax.set_ylabel("$H$ (absolute humidity)")
        ax.set_xlim(T_min, T_max)
        ax.set_ylim(0, Hs[-1] * 1.05)
        ax.legend(loc=0)
        # ax.minorticks_on()
        fig.tight_layout()
        # fig.savefig("./psychrometric.png", dpi=250)

        return ax


    def show_point(ax, T, H, T_min=10, T_max=60):
        """Display the current point, and adiabatic line, const concentration line
        etc
        """
        Hs = H_sat(T)
        Hy = enthalpy(H, T)
        assert H < Hs
        Hp = H / Hs
        pvap = pvap_water(T)
        pa = 101.325 / (0.622 / H + 1)
        HR = pa / pvap
        ax.plot(T, H, "o", color="black")
        # The constant T curve
        ax.plot([T, T], [0, Hs], color="tab:brown", alpha=0.8)

        # The constant Hs curve
        (T_dew,) = fsolve(lambda T: H - H_sat(T), T)
        ax.plot([T_dew, 100], [H, H], color="tab:brown", alpha=0.8)

        # The adiabatic curve
        T_test = np.linspace(T_min, T_max, 1000)
        H_test = adiabatic_from_enthalpy(Hy, T_test)
        Hs_test = H_sat(T_test)
        cond = np.where(H_test <= Hs_test)

        T_w = T_test[cond][0]
        ax.plot(T_test[cond], H_test[cond], color="tab:brown", alpha=0.8)

        summary_md = mo.md(f"""
    ### Current state summary

    - $T_d = {T:.2f}~^\\circ\\mathrm{{C}}$
    - $H = {H:.6f}~\\mathrm{{kg\\ water/kg\\ dry\\ air}}$
    - $T_w = {float(T_w):.2f}~^\\circ\\mathrm{{C}}$
    - $T_{{dew}} = {float(T_dew):.2f}~^\\circ\\mathrm{{C}}$
    - $H_p = {Hp * 100:.1f}\\%$
    - $H_R = {100 * HR:.1f}\\%$
    - $Hy = {Hy:.2f}~\\mathrm{{kJ/kg\\ dry\\ air}}$
    """)

        return ax, summary_md

    return display_plot, show_point


@app.cell
def _(UnivariateSpline, np):
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


    def adiabatic_from_enthalpy(Hy, T):
        """return the adiabatic line for give Ts

        Use the form
        H(T) = \frac{Hy - 1.005T}{2501.4 + 1.88T}
        and H is from entlapy of Hs at Tw
        """
        H = (Hy - 1.005 * T) / (2501.4 + 1.88 * T)

        return H


    def adiabatic_from_Tw(Tw, T):
        """return the adiabatic line for give Ts

        Use the form
        H(T) = \frac{Hy - 1.005T}{2501.4 + 1.88T}
        and H is from entlapy of Hs at Tw
        """
        assert np.all(T >= Tw)
        Hs = H_sat(Tw)
        Hy = enthalpy(Hs, Tw)
        H = (Hy - 1.005 * T) / (2501.4 + 1.88 * T)

        return H

    return (
        H_sat,
        adiabatic_from_Tw,
        adiabatic_from_enthalpy,
        enthalpy,
        pvap_water,
    )


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import fsolve

    return UnivariateSpline, fsolve, np, plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
