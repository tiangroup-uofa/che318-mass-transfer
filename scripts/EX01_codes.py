import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full", app_title="CHE 318 Exercise 1 Playground (Q3)")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A Playground for Q3

    This is an interactive playground for Q3 of Exercise 1 of CHE 318.

    We have compiled some python codes to calculate the $D_{AB}$  and flow rate
    in different assumptions. You can make use of the tool to help check your answers
    and make the plots.

    Please follow the instructions below. Have fun 😃!

    _Privacy Disclosure: Everything will run inside your browser and **no data** is being shared with external server._
    """)
    return


@app.cell
def _(D0_slider, T0_slider, T1_slider, fuller_method_relative, mo):
    _T1 = T1_slider.value + 273.15
    _T0 = 10 + 273.15
    _D0 = D0_slider.value * 1e-5
    _dab = fuller_method_relative(_D0, _T0, _T1)
    mo.md(f"""
    ## 1) Test your calculated value of $D_{{AB}}$ using Fuller method (Q 3.b)



    Acetone: {D0_slider} {T0_slider}

    New temperature {T1_slider}

    The corrected $D_{{AB}}$ at {T1_slider.value} ℃ is {_dab / 1e-5:.2f}⨉10$^{{-5}}$ m$^2$/s 
    """)
    return


@app.cell
def _(LJ_chapman_array, T_chap_slider, chapman_enskog_method, mo, np):
    _T_chap = T_chap_slider.value + 273.15
    _LJ_params = LJ_chapman_array.value
    _warning = (
        "**Missing some input values for LJ parameters!!**"
        if any([_i is None for _i in _LJ_params])
        else ""
    )
    _dab = (
        chapman_enskog_method(*_LJ_params, T=_T_chap, P=1)
        if not _warning
        else np.nan
    )

    mo.md(f"""
    ## 2) Test your calculated value of $D_{{AB}}$ using Chapman-Enskog method (Q 3.e)

    Please enter the Lennard-Jone parameters for acetone (A) and air (B)

    {LJ_chapman_array}

    Please select the target temperature

    {T_chap_slider} {_warning}


    The calculated $D_{{AB}}$ at {T_chap_slider.value} ℃ is {_dab / 1e-5:.2f}⨉10$^{{-5}}$ m$^2$/s.

    You should get $D_{{AB}} = 0.63\\times 10^{{-5}}$ m$^2$/s at $T=-40$ ℃.
    """)
    return


@app.cell
def _(LJ_chapman_array, T_range, geometry_array, mo, plot_choice, plot_func):
    _md1 = mo.md("""
    ## 3) Plot loss rate with different assumptions of $D_{{AB}}$ (Q 3.c, 3.d, 3.f)

    If your calculations for 1) and 2) are correct, please proceed to the plot tasks.
    """)

    _md2 = mo.md(f"""

    1. Define your system setup

    {T_range}

    {geometry_array}

    2. For Chapman-Enskog theory, provide the parameters

    {LJ_chapman_array}

    3. Choose which to plot

    {plot_choice}

    """)

    _plot = plot_func()


    mo.vstack(
        [_md1, mo.hstack([_md2, _plot], widths=[1, 2])],
        align="start",
        heights=[1, 10],
    )
    return


@app.cell
def _(
    LJ_chapman_array,
    T_range,
    chapman_enskog_method,
    flow_rate,
    fuller_method_relative,
    geometry_array,
    mo,
    np,
    plot_choice,
    plt,
):
    _inputs = {
        "T": T_range.value,
        "LJ": LJ_chapman_array.value,
        "geom": geometry_array.value,
        "plot": plot_choice.value,
    }
    _methods = plot_choice.value

    _T_range = np.linspace(_inputs["T"][0], _inputs["T"][1], 128) + 273.15


    def _plot_flow_rate_const(ax):
        L, D = _inputs["geom"]
        M_A = _inputs["LJ"][4]
        P_T = 101325
        tot_NA = flow_rate(L, D, _T_range, M_A)
        ax.plot(_T_range, tot_NA, color="#3f77d1", label="Constant $D_{AB}$")
        return


    def _plot_flow_rate_fuller(ax):
        L, D = _inputs["geom"]
        M_A = _inputs["LJ"][4]
        P_T = 101325
        tot_NA = flow_rate(
            L,
            D,
            _T_range,
            M_A,
            D_AB_func=lambda T: fuller_method_relative(
                D0=1e-5, T0=273.15 + 10, T1=T
            ),
        )
        ax.plot(_T_range, tot_NA, color="#d1a03f", label="Fuller method $D_{AB}$")
        return


    def _plot_flow_rate_chapman(ax):
        L, D = _inputs["geom"]
        sigma_A, sigma_B, eps_A, eps_B, M_A, M_B = _inputs["LJ"]
        P_T = 101325
        tot_NA = flow_rate(
            L,
            D,
            _T_range,
            M_A,
            D_AB_func=lambda T: chapman_enskog_method(
                sigma_A=sigma_A,
                sigma_B=sigma_B,
                eps_A=eps_A,
                eps_B=eps_B,
                m_A=M_A,
                m_B=M_B,
                T=T,
            ),
        )
        ax.plot(_T_range, tot_NA, color="#c72c93", label="Chapman-Enskog $D_{AB}$")
        return


    def plot_func():
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_title("Total Acetone Loss Rate vs Temperature")
        ax.set_xlabel("Temperature $T$ (K)")
        ax.set_ylabel("$M_A \\times N_A$ (kg / day)")
        if any([None in _inputs["LJ"]]) or any([None in _inputs["geom"]]):
            # Display warning text on the center
            ax.text(
                x=0.5,
                y=0.5,
                s="Missing Values!!!!",
                transform=ax.transAxes,
                color="red",
            )
            return ax

        # Constant D
        if _inputs["plot"][0]:
            _plot_flow_rate_const(ax)

        # Fuller method
        if _inputs["plot"][1]:
            _plot_flow_rate_fuller(ax)

        # Chapman-Enskog method
        if _inputs["plot"][2]:
            _plot_flow_rate_chapman(ax)

        ax.grid(True)
        ax.legend()
        return mo.mpl.interactive(ax)
    return (plot_func,)


@app.cell
def _(mo):
    # UI components

    # slider components for test in b)
    T1_slider = mo.ui.slider(
        start=-40, stop=40, step=5, value=-40, show_value=True, label="$T$ (℃)"
    )
    T0_slider = mo.ui.slider(
        start=-20,
        stop=30,
        step=5,
        value=10,
        show_value=True,
        label="Experimental $T$ (℃) when $D_{AB}$ measured",
    )
    D0_slider = mo.ui.slider(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=1.0,
        show_value=True,
        label="Experimental $D_{AB}$ (10$^{-5}$ m$^{2}$/s)",
    )

    # slider components for test in e) Chapman-Enskog
    T_chap_slider = mo.ui.slider(
        start=-40, stop=40, step=5, value=-40, show_value=True, label="$T$ (℃)"
    )
    LJ_chapman_array = mo.ui.array(
        label="Lennard-Jone Parameters",
        elements=[
            mo.ui.number(label=r"$\sigma_{A}$ (Å)"),
            mo.ui.number(label=r"$\sigma_{B}$ (Å)"),
            mo.ui.number(label=r"$\epsilon_{A}$ ($k_B\cdot$K)"),
            mo.ui.number(label=r"$\epsilon_{B}$ ($k_B\cdot$K)"),
            mo.ui.number(label=r"$M_{A}$ (kg/kgmol)"),
            mo.ui.number(label=r"$M_{B}$ (kg/kg mol)"),
        ],
    )

    # slider components for test in c), d), and f)
    T_range = mo.ui.range_slider(
        start=-80,
        stop=80,
        step=10,
        show_value=True,
        value=(-40, 40),
        label="$T$ range (℃)",
    )
    geometry_array = mo.ui.array(
        label="Geometric parameters",
        elements=[
            mo.ui.number(label=r"$L$ (m)"),
            mo.ui.number(label=r"Diameter (m)"),
        ],
    )
    plot_choice = mo.ui.array(
        label="Choose which to plot",
        elements=[
            mo.ui.switch(label="Constant $D_{AB}$", value=False),
            mo.ui.switch(label="Fuller", value=False),
            mo.ui.switch(label="Chapman-Enskog", value=False),
        ],
    )
    return (
        D0_slider,
        LJ_chapman_array,
        T0_slider,
        T1_slider,
        T_chap_slider,
        T_range,
        geometry_array,
        plot_choice,
    )


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt


    def fuller_method_relative(
        D0: float,
        T0: float,
        T1: float,
        P0: float = 101325.0,
        P1: float = 101325.0,
    ) -> float:
        """
        Calculates gas diffusivity at new conditions (T1, P1) relative to
        reference conditions (T0, P0) using the Fuller method.

        The Fuller method estimates gas diffusivity (D) to be proportional to
        temperature (T) raised to the power of 1.75 and inversely proportional
        to pressure (P).

        Fuller method: D is proportional to T^1.75 / P
        D1 / D0 = (T1/T0)^1.75 * (P0/P1)

        Args:
            D0: Reference diffusivity at T0, P0 in m^2/s.
            T0: Reference absolute temperature in Kelvin.
            T1: New absolute temperature in Kelvin.
            P0: Reference pressure in Pascals (default to 101325 Pa).
            P1: New pressure in Pascals (default to 101325 Pa).

        Returns:
            The diffusivity at T1, P1 in m^2/s.
        """

        D1 = D0 * (T1 / T0) ** 1.75 * (P0 / P1)
        return D1
    return fuller_method_relative, np, plt


@app.cell
def _():
    return


@app.cell
def _(np):
    from __future__ import annotations

    from typing import Callable, Union

    # Table 2: Collision integral values (T* = k_B T / ε, Ω_D)
    _LJ_OMEGA_D_TABLE = {
        "T_star": np.array(
            [
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
                1.05,
                1.10,
                1.15,
                1.20,
                1.25,
                1.30,
                1.35,
                1.40,
                1.45,
                1.50,
                1.55,
                1.60,
                1.65,
                1.70,
                1.75,
                1.80,
                1.85,
                1.90,
                1.95,
                2.00,
                2.10,
                2.20,
                2.30,
                2.40,
                2.50,
                2.60,
                2.70,
                2.80,
                2.90,
                3.00,
                3.10,
                3.20,
                3.30,
                3.40,
                3.50,
                3.60,
                3.70,
                3.80,
                3.90,
                4.00,
                4.10,
                4.20,
                4.30,
                4.40,
                4.50,
                4.60,
                4.70,
                4.80,
                4.90,
                5.00,
                6.00,
                7.00,
                8.00,
                9.00,
                10.00,
                20.00,
                30.00,
                40.00,
                50.00,
                60.00,
                70.00,
                80.00,
                90.00,
                100.00,
                200.00,
                300.00,
                400.00,
            ],
            dtype=float,
        ),
        "Omega_D": np.array(
            [
                2.662,
                2.476,
                2.318,
                2.184,
                2.066,
                1.966,
                1.877,
                1.798,
                1.729,
                1.667,
                1.612,
                1.562,
                1.517,
                1.476,
                1.439,
                1.406,
                1.375,
                1.346,
                1.320,
                1.296,
                1.273,
                1.253,
                1.233,
                1.215,
                1.198,
                1.182,
                1.167,
                1.153,
                1.140,
                1.128,
                1.116,
                1.105,
                1.094,
                1.084,
                1.075,
                1.057,
                1.041,
                1.026,
                1.012,
                0.9996,
                0.9878,
                0.9770,
                0.9672,
                0.9576,
                0.9490,
                0.9406,
                0.9328,
                0.9256,
                0.9186,
                0.9120,
                0.9058,
                0.8998,
                0.8942,
                0.8888,
                0.8836,
                0.8788,
                0.8740,
                0.8694,
                0.8652,
                0.8610,
                0.8568,
                0.8530,
                0.8492,
                0.8456,
                0.8422,
                0.8124,
                0.7896,
                0.7712,
                0.7556,
                0.7424,
                0.6640,
                0.6232,
                0.5960,
                0.5756,
                0.5596,
                0.5464,
                0.5352,
                0.5256,
                0.5170,
                0.4644,
                0.4360,
                0.4172,
            ],
            dtype=float,
        ),
    }


    def lj_omega_d(T_star: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Returns the Lennard–Jones collision integral for diffusion, Ω_D(T*),
        by interpolating the tabulated values versus reduced temperature T*.

        Args:
            T_star: Reduced temperature T* = k_B T / ε_AB (dimensionless).

        Returns:
            Collision integral Ω_D (dimensionless).

        Notes:
            Log-log interpolation is used for smooth behavior over wide T* range:
            ln(Ω_D) is linearly interpolated in ln(T*).

            ```{=tex}
            \\begin{align}
            \\ln\\Omega_D(T^*) \\approx \\ln\\Omega_{D,1}
            + \\frac{\\ln T^* - \\ln T_1^*}{\\ln T_2^* - \\ln T_1^*}
            \\left(\\ln\\Omega_{D,2} - \\ln\\Omega_{D,1}\\right)
            \\end{align}
            ```
        """
        _T_is_scalar = np.isscalar(T_star)
        T_star = np.atleast_1d(T_star)
        if np.any(T_star <= 0):
            raise ValueError("T_star must be > 0.")

        Ttab = _LJ_OMEGA_D_TABLE["T_star"]
        Otab = _LJ_OMEGA_D_TABLE["Omega_D"]

        # log-log interpolation with end extrapolation using nearest segment
        x = np.log(T_star)
        x_tab = np.log(Ttab)
        y_tab = np.log(Otab)

        y = np.interp(x, x_tab, y_tab, left=y_tab[0], right=y_tab[-1])
        recov_y = np.exp(y)
        if _T_is_scalar:
            recov_y = float(recov_y[0])
        return recov_y
    return Callable, Union, lj_omega_d


@app.cell
def _(Callable, lj_omega_d, np):
    def chapman_enskog_method(
        sigma_A: float,
        sigma_B: float,
        eps_A: float,
        eps_B: float,
        m_A: float,
        m_B: float,
        T: np.ndarray=298.15,
        P: float=1.0,
        omega_D: Callable[[float], float]=lj_omega_d,
    ) -> float:
        """
        Estimates binary gas diffusivity using the Chapman–Enskog method for
        Lennard–Jones (12-6) molecules.

        This implementation uses the common Chapman–Enskog form with an LJ collision
        integral Ω_D,AB(T*) and produces D_AB in molar flux units.

        Args:
            m_A: Molecular weight of A kg / kg mol
            m_B: Molecular weight of B kg / kg mol
            eps_A: Lennard–Jones energy parameter ε_A/k_B in Kelvin.
            eps_B: Lennard–Jones energy parameter ε_B/k_B in Kelvin.
            sigma_A: Lennard–Jones size parameter σ_A in Å.
            sigma_B: Lennard–Jones size parameter σ_B in Å.
            T: Absolute temperature in Kelvin.
            P: Pressure in atm.
            omega_D: Function that returns Ω_D,AB given reduced temperature T*.

        Returns:
            Binary diffusivity D_AB from Chapman–Enskog
            D_AB = 1.8583e-7 * T^(3/2) / (P * σ_AB^2 * Ω_D,AB) * (1/ma + 1/mb)^2
        """
        # if T <= 0:
        #     raise ValueError("T must be > 0 K.")
        # if P <= 0:
        #     raise ValueError("P must be > 0 atm.")
        # if sigma_A <= 0 or sigma_B <= 0:
        #     raise ValueError("sigma_A and sigma_B must be > 0 Å.")
        # if eps_A <= 0 or eps_B <= 0:
        #     raise ValueError("eps_A and eps_B must be > 0 K.")

        # Mixing rules
        eps_AB = np.sqrt(eps_A * eps_B)          # [K]
        sigma_AB = 0.5 * (sigma_A + sigma_B)       # [Å]

        # Reduced temperature
        T_star = T / eps_AB
        Omega_DAB = omega_D(T_star)

        # if Omega_DAB <= 0:
        #     raise ValueError("Collision integral omega_D(T*) must be > 0.")


        m_ab_sq = (1 / m_A + 1 / m_B) ** 0.5
        D_m2_s = 1.8583e-7 * (T ** 1.5) / (P * (sigma_AB ** 2) * Omega_DAB) * m_ab_sq

        return D_m2_s
    return (chapman_enskog_method,)


@app.cell
def _(Union, np):
    def vapor_pressure(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculates the vapor pressure in Pascals using the Antoine-like equation.

        Args:
            T: Temperature in Kelvin. Can be a float or a numpy array.

        Returns:
            The vapor pressure in Pascals. Returns a float if T is a float,
            or a numpy array if T is a numpy array.
        """
        _B = 7.904
        _A = 7641.5
        _TORR_TO_PA = 133.322

        # Calculate log_10(p) in torr
        _log_p_torr = _B - 0.2185 * (_A / T)

        # Convert log_10(p) to p in torr
        _p_torr = 10**_log_p_torr

        # Convert p from torr to Pascals
        _p_pa = _p_torr * _TORR_TO_PA
        return _p_pa
    return (vapor_pressure,)


@app.cell
def _(np, vapor_pressure):
    def _const_D(T, D0=1.0e-5):
        d = np.ones_like(T) * D0
        return d


    def flow_rate(
        L, D, T, M_A, P_T=101325, P_A_func=vapor_pressure, D_AB_func=_const_D
    ):
        """return flow rate from stagnant B case in kg/day"""
        area = (D / 2) ** 2 * np.pi
        P_A1 = P_A_func(T)
        P_A2 = 0
        P_B1 = P_T - P_A1
        P_B2 = P_T - P_A2
        P_BM = (P_B1 - P_B2) / np.log(P_B1 / P_B2)
        D_AB = D_AB_func(T)
        R = 8314
        N_A = D_AB / L / R / T * P_T / P_BM * (P_A1 - P_A2)
        seconds_in_day = 3600 * 24
        TOT_N_A = area * N_A * M_A * seconds_in_day
        return TOT_N_A
    return (flow_rate,)


if __name__ == "__main__":
    app.run()
