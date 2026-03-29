import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.integrate import cumulative_trapezoid, solve_ivp
    from scipy.interpolate import UnivariateSpline, interp1d
    from scipy.optimize import fsolve

    return (
        UnivariateSpline,
        cumulative_trapezoid,
        fsolve,
        interp1d,
        mo,
        np,
        pd,
        plt,
        solve_ivp,
    )


@app.cell(hide_code=True)
def _(
    UnivariateSpline,
    cumulative_trapezoid,
    fsolve,
    interp1d,
    np,
    pd,
    solve_ivp,
):
    MB = 28.97

    _T_C = np.array([0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    _P_kPa = np.array(
        [0.611, 1.228, 2.338, 3.168, 4.242, 7.375, 12.333, 19.92, 31.16, 47.34, 70.10, 101.325],
        dtype=float,
    )

    def pvap_water(T):
        spline = UnivariateSpline(_T_C, _P_kPa, s=0)
        return spline(T)

    def H_abs(p, pT=101.325):
        return 0.622 * p / (pT - p)

    def H_sat(T, pT=101.325):
        return H_abs(p=pvap_water(T), pT=pT)

    def cs(H):
        return 1.005 + 1.88 * H

    def enthalpy(H, T):
        return cs(H) * T + 2501.4 * H

    def cL(T=None):
        return 4.187

    def operating_line(L, G, T_L, T_L1, H_y1):
        return L * cL() / G * (T_L - T_L1) + H_y1

    def slope_to_interface(hLa, kGaP, T_L, H_y, eq_line):
        slope = -hLa / kGaP / MB

        def solve_one(TL_, Hy_):
            def loss(T):
                H_eq = np.interp(T, eq_line["T_L"], eq_line["H_y"])
                return (H_eq - Hy_) - (T - TL_) * slope

            Ti = float(fsolve(loss, x0=TL_)[0])
            Hyi = float(np.interp(Ti, eq_line["T_L"], eq_line["H_y"]))
            return Ti, Hyi

        Ti_list, Hyi_list = zip(
            *(solve_one(TL_, Hy_) for TL_, Hy_ in zip(np.atleast_1d(T_L), np.atleast_1d(H_y)))
        )
        return np.asarray(Ti_list, dtype=float), np.asarray(Hyi_list, dtype=float)

    def integrate_everything(hLa, kGaP, T_G1, op_line, eq_line, L, G):
        T_L = np.asarray(op_line["T_L"], dtype=float)
        H_y = np.asarray(op_line["H_y"], dtype=float)
        T_i, H_yi = slope_to_interface(hLa=hLa, kGaP=kGaP, T_L=T_L, H_y=H_y, eq_line=eq_line)

        Hy_fun = interp1d(T_L, H_y, kind="linear", bounds_error=False, fill_value="extrapolate")
        Ti_fun = interp1d(T_L, T_i, kind="linear", bounds_error=False, fill_value="extrapolate")
        Hyi_fun = interp1d(T_L, H_yi, kind="linear", bounds_error=False, fill_value="extrapolate")

        K = L * cL() / G

        def rhs(TL, TG):
            denom = float(Hyi_fun(TL) - Hy_fun(TL))
            if abs(denom) < 1e-10:
                denom = np.sign(denom) * 1e-10 if denom != 0 else 1e-10
            return [K * (float(Ti_fun(TL)) - TG[0]) / denom]

        sol = solve_ivp(
            rhs,
            t_span=(float(T_L[0]), float(T_L[-1])),
            y0=[float(T_G1)],
            t_eval=T_L,
            method="RK45",
            max_step=max((T_L[-1] - T_L[0]) / 50.0, 1e-6),
        )
        if not sol.success:
            raise RuntimeError(f"T_G integration failed: {sol.message}")
        T_G = sol.y[0]

        denom_z = H_yi - H_y
        denom_z = np.where(np.abs(denom_z) < 1e-10, np.sign(denom_z) * 1e-10, denom_z)
        dz_dHy = G / (kGaP * MB) / denom_z
        z = cumulative_trapezoid(dz_dHy, H_y, initial=0.0)
        z_total = float(z[-1]) if len(z) else 0.0
        z_frac = z / z_total if z_total > 0 else np.zeros_like(z)

        results = pd.DataFrame({
                "idx": np.arange(len(T_L), dtype=int),
                "z": z,
                "z_frac": z_frac,
                "T_L": T_L,
                "T_i": T_i,
                "T_G": T_G,
                "H_y": H_y,
                "H_yi": H_yi,
                "1/(H_yi-H_y)": 1.0 / denom_z,
            })
        return results

    return (
        H_sat,
        enthalpy,
        integrate_everything,
        operating_line,
        slope_to_interface,
    )


@app.cell(hide_code=True)
def _(mo):
    base_setup = mo.md(
        r"""
    ## System setup

    - $T_{{L1}}$ (℃) {T1}
    - $T_{{G1}}$ (℃) {TG1}
    - $H_{{y1}}$ (kJ/kg dry air) {Hy1}
    - $T_{{L2}}$ (℃) {T2}
    - $h_L a$ (kJ/m$^3$ s K) {hLa}
    - $k_G a P$ (kg mol/s m$^3$) {kGaP}
    - Liquid rate $L$ (kg / s m$^2$) {L}
    - Gas rate $G$ (kg / s m$^2$) {G}
    """
    ).batch(
        T1=mo.ui.number(step=0.1, value=30.0),
        TG1=mo.ui.number(step=0.1, value=30.0),
        Hy1=mo.ui.number(step=0.1, value=45.0),
        T2=mo.ui.number(step=0.1, value=50.0),
        hLa=mo.ui.number(step=0.1, value=14.5),
        kGaP=mo.ui.number(step=1e-4, value=0.012),
        L=mo.ui.number(step=0.01, value=2.0),
        G=mo.ui.number(step=0.01, value=2.0),
    )

    display_controls = mo.md(
        r"""
    ## Display and illustration controls

    - Temp range in chart (℃) {T_range}
    - Show diagonal interface lines {show_diagonals}
    - Show bulk-gas $T_G$-$H_y$ path {show_tg}
    - Operating-line points for integration {npts}
    - Current location in tower (% of total height) {z_pct}
    """
    ).batch(
        T_range=mo.ui.range_slider(start=10, stop=85, step=5, value=(25, 55), show_value=True),
        show_diagonals=mo.ui.switch(value=True),
        show_tg=mo.ui.switch(value=True),
        npts=mo.ui.slider(value=7, start=3, stop=25, step=1, show_value=True),
        z_pct=mo.ui.slider(value=50, start=0, stop=100, step=1, show_value=True),
        # gas_shape=mo.ui.slider(value=3.0, start=0.5, stop=8.0, step=0.1, show_value=True),
        # liq_shape=mo.ui.slider(value=3.0, start=0.5, stop=8.0, step=0.1, show_value=True),
    )
    mo.hstack([base_setup, display_controls], widths=[1, 1])
    return base_setup, display_controls


@app.cell(hide_code=True)
def _(
    H_sat,
    base_setup,
    display_controls,
    enthalpy,
    integrate_everything,
    mo,
    np,
    operating_line,
    pd,
    plt,
    slope_to_interface,
):
    def build_eq_line(T_min=10, T_max=60):
        T = np.linspace(T_min, T_max, 250)
        Hs = H_sat(T)
        Hy_sat = enthalpy(Hs, T)
        return {"T_L": T, "H_y": Hy_sat}


    def build_op_line(T_L1, H_y1, T_L2, L, G):
        T_range = np.linspace(T_L1, T_L2, 240)
        Hy_op = operating_line(L, G, T_range, T_L1, H_y1)
        return {"T_L": T_range, "H_y": Hy_op}


    def make_integrant_preview(op_line, eq_line, hLa, kGaP, npts):
        T_range = op_line["T_L"]
        H_y = op_line["H_y"]
        T_pts = np.linspace(T_range[0], T_range[-1], npts)
        Hy_pts = np.interp(T_pts, T_range, H_y)
        T_i_pts, H_yi_pts = slope_to_interface(
            hLa=hLa, kGaP=kGaP, T_L=T_pts, H_y=Hy_pts, eq_line=eq_line
        )
        return pd.DataFrame(
            {
                "T_L": T_pts,
                "H_y": Hy_pts,
                "T_i": T_i_pts,
                "H_yi": H_yi_pts,
                "1/(H_yi-H_y)": 1.0 / (H_yi_pts - Hy_pts),
            }
        )


    def draw_column(ax, z_frac):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        wall_x1, wall_x2 = 0.20, 0.80
        y1, y2 = 0.08, 0.92
        theta = np.linspace(0, np.pi, 200)
        ax.plot([wall_x1, wall_x1], [y1, y2], color="0.2", lw=2)
        ax.plot([wall_x2, wall_x2], [y1, y2], color="0.2", lw=2)
        ax.plot(
            (wall_x1 + wall_x2) / 2 + (wall_x2 - wall_x1) / 2 * np.cos(theta),
            y2 + 0.035 * np.sin(theta),
            color="0.2",
            lw=2,
        )
        ax.plot(
            (wall_x1 + wall_x2) / 2 + (wall_x2 - wall_x1) / 2 * np.cos(theta),
            y2 - 0.035 * np.sin(theta),
            color="0.2",
            lw=2,
        )
        ax.plot(
            (wall_x1 + wall_x2) / 2 + (wall_x2 - wall_x1) / 2 * np.cos(theta),
            y1 - 0.035 * np.sin(theta),
            color="0.2",
            lw=2,
        )
        y_line = y1 + z_frac * (y2 - y1)
        ax.hlines(y_line, wall_x1 - 0.06, wall_x2 + 0.06, colors="tab:red", lw=2)
        ax.text(0.5, 1.00, "Column", ha="center", va="bottom")
        ax.text(
            wall_x2 + 0.05,
            y_line,
            f"z/Z = {z_frac:.2f}",
            ha="left",
            va="bottom",
            color="tab:red",
        )


    def draw_pseudo_temperature_profile(
        ax, row, gas_shape=5.5, liq_shape=5.5, ylim=None
    ):
        xi_liq = np.linspace(-1.0, 0.0, 150)
        xi_gas = np.linspace(0.0, 1.0, 150)
        # s_liq = xi_liq + 1.0
        s_liq = -xi_liq
        s_gas = xi_gas
        Tliq = row["T_i"] + (row["T_L"] - row["T_i"]) * (
            1.0 - np.exp(-liq_shape * s_liq)
        ) / (1.0 - np.exp(-liq_shape))
        Tgas = row["T_i"] + (row["T_G"] - row["T_i"]) * (
            1.0 - np.exp(-gas_shape * s_gas)
        ) / (1.0 - np.exp(-gas_shape))

        ax.plot(xi_liq, Tliq, lw=2, label="liquid film")
        ax.plot(xi_gas, Tgas, lw=2, label="gas film")
        ax.axvline(0.0, ls="--", color="0.5", lw=1)
        ax.axhline(row["T_L"], ls=":", color="0.5", lw=1)
        ax.axhline(row["T_i"], ls=":", color="0.5", lw=1)
        ax.axhline(row["T_G"], ls=":", color="0.5", lw=1)
        ax.scatter(
            [-1, 0, 1], [row["T_L"], row["T_i"], row["T_G"]], s=40, color="grey", zorder=3
        )
        ax.text(-1.0, row["T_L"], "  $T_L$", va="bottom")
        ax.text(0.0, row["T_i"], "  $T_i=T_{Gi}=T_{Li}$", va="bottom")
        ax.text(1.0, row["T_G"], "  $T_G$", va="bottom", ha="right")
        ax.set_xlabel("horizontal coordinate (arb. unit)")
        ax.set_xticks([])
        ax.set_ylabel("temperature (℃)")
        ax.set_title("Temperature profile at selected height")
        # ax.set_ylim(30, 60)
        if ylim is not None:
            ax.set_ylim(*display_controls.value["T_range"])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.05, 1.05)


    def make_dashboard(eq_line, op_line, all_df, integrant_df, controls):
        z_pct = controls["z_pct"]
        idx = int(np.argmin(np.abs(all_df["z_frac"] - z_pct / 100.0)))
        row = all_df.iloc[idx]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 5),
            gridspec_kw={"width_ratios": [1.8, 1.3, 0.5]},
        )
        ax0, ax1, ax2 = axes

        ax0.plot(
            eq_line["T_L"],
            eq_line["H_y"],
            lw=1.8,
            color="0.10",
            label="Equilibrium curve ($H_{{yi}} - T_{{L}}$)",
        )
        ax0.plot(
            op_line["T_L"],
            op_line["H_y"],
            color="tab:red",
            lw=2.0,
            label="Operating line ($H_{{y}} - T_{{L}}$)",
        )
        ax0.plot(
            [op_line["T_L"][0], op_line["T_L"][-1]],
            [op_line["H_y"][0], op_line["H_y"][-1]],
            "o",
            color="tab:red",
        )
        if controls["show_tg"]:
            ax0.plot(
                all_df["T_G"],
                all_df["H_y"],
                color="tab:blue",
                lw=2.0,
                label="Gas operating line ($H_{{y}} - T_{{G}}$)",
            )
            ax0.plot(
                [all_df["T_G"].iloc[0], all_df["T_G"].iloc[-1]],
                [all_df["H_y"].iloc[0], all_df["H_y"].iloc[-1]],
                "o",
                color="tab:blue",
            )
        if controls["show_diagonals"]:
            for _, r in integrant_df.iterrows():
                ax0.plot(
                    [r["T_L"], r["T_i"]],
                    [r["H_y"], r["H_yi"]],
                    ls="--",
                    color="0.55",
                    alpha=0.8,
                )
        ax0.scatter([row["T_L"]], [row["H_y"]], s=64, color="grey", zorder=4)
        ax0.text(row["T_L"], row["H_y"]* 0.98, s="Liq.", ha="left", va="top")
        ax0.scatter([row["T_i"]], [row["H_yi"]], s=64, color="grey", zorder=4)
        ax0.text(row["T_i"], row["H_yi"] * 1.02, s="Interface", ha="right", va="bottom")
        if controls["show_tg"]:
            ax0.scatter(
                [row["T_G"]], [row["H_y"]], s=64, color="grey", zorder=4
            )
            ax0.text(row["T_G"] * 0.98, row["H_y"], s="Gas.", ha="right", va="bottom")
        ax0.set_title("Enthalpy-temperature chart")
        ax0.set_xlabel("temperature (℃)")
        ax0.set_ylabel("gas enthalpy $H_y$ (kJ / kg dry air)")
        ax0.grid(True, alpha=0.35)
        ax0.legend(loc="upper left")

        draw_pseudo_temperature_profile(
            ax1, row, ylim=(op_line["T_L"][0], op_line["T_L"][-1])
        )
        # controls["gas_shape"], controls["liq_shape"])
        draw_column(ax2, float(row["z_frac"]))

        fig.tight_layout()
        return fig, idx


    setup_value = base_setup.value
    controls = display_controls.value

    eq_line = build_eq_line(*controls["T_range"])
    op_line = build_op_line(
        T_L1=setup_value["T1"],
        H_y1=setup_value["Hy1"],
        T_L2=setup_value["T2"],
        L=setup_value["L"],
        G=setup_value["G"],
    )
    all_df = integrate_everything(
        hLa=setup_value["hLa"],
        kGaP=setup_value["kGaP"],
        T_G1=setup_value["TG1"],
        op_line=op_line,
        eq_line=eq_line,
        L=setup_value["L"],
        G=setup_value["G"],
    )
    integrant_df = make_integrant_preview(
        op_line, eq_line, setup_value["hLa"], setup_value["kGaP"], controls["npts"]
    )
    fig, current_idx = make_dashboard(
        eq_line, op_line, all_df, integrant_df, controls
    )

    z_total = float(all_df["z"].iloc[-1])
    current_row = all_df.iloc[current_idx]

    status = mo.md(
        f"""
    ### Current integrated solution

    - Total tower height $Z = {z_total:.3f}$ m
    - Current operating-line index = {current_idx}
    - Current location $z = {current_row["z"]:.3f}$ m ({100 * current_row["z_frac"]:.1f}% of $Z$)
    - Selected states: $T_L = {current_row["T_L"]:.2f}$ ℃, $T_i = {current_row["T_i"]:.2f}$ ℃, $T_G = {current_row["T_G"]:.2f}$ ℃
    - Selected enthalpies: $H_y = {current_row["H_y"]:.2f}$, $H_{{yi}} = {current_row["H_yi"]:.2f}$ kJ/kg dry air
    """
    )
    return current_row, fig, z_total


@app.cell(hide_code=True)
def _(current_row, display_controls, fig, mo, z_total):
    # current_view = all_df.loc[[current_idx], ["idx", "z", "z_frac", "T_L", "T_i", "T_G", "H_y", "H_yi"]]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(f"Total tower height $Z={z_total:.2f}$ m."),
                    mo.md("Move the slider to control current height percentage %"),
                    display_controls.elements["z_pct"],
                    mo.md(f"Current $z={current_row['z']:.2f}$ m"),
                ],
                justify="start",
            ),
            fig,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
