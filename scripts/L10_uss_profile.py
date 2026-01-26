# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    # Define solvent properties: name, vapor pressure (Pa), D_AB (m^2/s)
    # UI definitions
    solvents = [
        {"name": "Water", "p_vap": 3167, "D_AB": 2.60e-5},
        {"name": "Ethanol", "p_vap": 7890, "D_AB": 1.35e-5},
        {"name": "Acetone", "p_vap": 30700, "D_AB": 1.6e-5},
        {"name": "Benzene", "p_vap": 13200, "D_AB": 0.962e-5},
    ]

    solvent_names = [s["name"] for s in solvents]

    # UI elements
    solvent_dropdown = mo.ui.dropdown(solvent_names, label="Solvent", value="Water", allow_select_none=False)
    L_slider = mo.ui.slider(0.1, 1.0, value=0.50, step=0.1, show_value=True, label="Tube Length L (m)")
    time_slider = mo.ui.slider(0, 3600, value=60, step=60, show_value=True, label="Max Time (s)",)

    return L_slider, solvent_dropdown, solvents, time_slider


@app.cell
def _(
    L_slider,
    mo,
    plot_time_evolution,
    solvent_dropdown,
    system_information,
    time_slider,
):
    mo.vstack(
        [
            mo.hstack(
                [
                    [
                        solvent_dropdown,
                        L_slider,
                        time_slider,
                    ],
                    mo.md(f"""
                        Current Parameters (1 atm, 298.15 K):

                        - $D_{{AB}}$: {system_information()["D_AB"]:.2e} m²/s
                        - $p_{{vap}}$: {system_information()["p_vap"]:.1f} Pa
                        - $x_{{A0}}$: {system_information()["x_a0"]:.4f}
                        """),
                ],
                widths=[1, 1.5],
            ),
            plot_time_evolution(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(np):
    # Solve steady state solution
    def _solve_K1_K2(z1, z2, xA1, xA2, s=1):
        """
        Solve for K1 and K2 in x_A = s - K1 exp(K2 z)
        given x_A(z1)=xA1 and x_A(z2)=xA2
        """
        K2 = np.log((s - xA2) / (s - xA1)) / (z2 - z1)
        K1 = (s - xA1) / np.exp(K2 * z1)
        return K1, K2


    def _xa_profile(z1, z2, xA1, xA2, s=1):
        K1, K2 = _solve_K1_K2(z1, z2, xA1, xA2, s)
        z_arr = np.linspace(z1, z2, 100)
        x_arr = s - K1 * np.exp(K2 * z_arr)
        return z_arr, x_arr


    def stead_state_solution(x_a0, L, D_AB, c=101325 / 8314 / 298.15):
        """Return z, x_a, N_A and N_B
        By default use 1 atm
        """
        z, x_a = _xa_profile(0, L, x_a0, 0, s=1)
        N_A = c * D_AB / L * np.log((1 - 0) / (1 - x_a0))
        N_B = 0
        return z, x_a, N_A, N_B
    return (stead_state_solution,)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.special import erf
    from scipy.optimize import fmin
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    return erf, interp1d, mo, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(evaporate_stagnantB_mol, plt, stead_state_solution, system_information):
    def plot_time_evolution() -> plt.Figure:
        """
        Plot the time evolution of concentration across the grid using system parameters.

        Returns:
            Matplotlib figure object
        """
        # Get system parameters
        info = system_information()

        # Get transient solution
        z, t, x, NA, NB = evaporate_stagnantB_mol(
            t_span=[0, info["time"]],
            L=info["L"],
            x_a0=info["x_a0"],
            D_AB=info["D_AB"],
            return_history=True,
        )

        # Get steady state solution
        # c = info["P_total"] / (info["R"] * info["T"])
        z_ss, x_ss, NA_ss, NB_ss = stead_state_solution(
            L=info["L"],
            x_a0=info["x_a0"],
            D_AB=info["D_AB"],
            # c=c
        )

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

        # Select time points to plot
        t_indices = range(len(t))

        # Plot concentration profiles
        for idx in t_indices:
            time_point = t[idx]
            axes[0].plot(z, x[idx, :], label=f"t = {time_point:.1f} s")

        # Add steady state concentration profile
        axes[0].plot(z_ss, x_ss, "k--", linewidth=2, label="Steady State")

        # Plot flux profiles
        for idx in t_indices:
            time_point = t[idx]
            axes[1].plot(z, NA[idx, :], label=f"t = {time_point:.1f} s")

        # Add steady state flux
        axes[1].plot(
            [0, info["L"]],
            [NA_ss, NA_ss],
            "k--",
            linewidth=2,
            label="Steady State",
        )

        # Format concentration plot
        axes[0].set_xlabel("Position (m)")
        axes[0].set_ylabel("Fraction $x_{A}$")
        axes[0].set_title(f"Concentration Profile - {info['solvent']}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Format flux plot
        axes[1].set_xlabel("Position (m)")
        axes[1].set_ylabel("$N_A$ (kg mol/m²/s)")
        axes[1].set_title(f"Flux $N_A$ Profile - {info['solvent']}")
        axes[1].set_ylim(1e-8, NA_ss * 1.618)
        # axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        return fig


    # Create and display the plot
    # plot_time_evolution()
    return (plot_time_evolution,)


@app.cell(hide_code=True)
def _(erf, interp1d, np):
    # Analytical solution according to Bird ch 20.1. Only works for 1D problem when L --> infty
    def _phi_to_xa0(phi: np.ndarray) -> np.ndarray:
        xa0 = 1.0 / (
            1.0 + 1.0 / (np.sqrt(np.pi) * (1.0 + erf(phi)) * phi * np.exp(phi**2))
        )
        return xa0


    _phi_array = np.linspace(0, 5, 1024)
    _x_a0_array = _phi_to_xa0(_phi_array)


    def xa0_to_phi(xa0_value: np.ndarray) -> np.ndarray:
        """
        Interpolates phi value from x_a0 value using pre-calculated arrays.

        Args:
            xa0_value: Value of x_a0, should be between 0 and 0.99

        Returns:
            Interpolated phi value
        """
        # Create interpolation function
        interp_func = interp1d(
            _x_a0_array,
            _phi_array,
            bounds_error=False,
            fill_value=(_phi_array[0], _phi_array[-1]),
        )

        return interp_func(xa0_value)


    def solve_evap_analytical(
        z: np.ndarray, x_a0: float, D_AB: float, t: float
    ) -> np.ndarray:
        """
        Solve the analytical solution for 1D evaporation/diffusion problem.

        This function calculates the concentration profile x_A(z,t) for a binary
        mixture undergoing evaporation, based on an analytical solution to Fick's
        second law with appropriate boundary conditions.

        Parameters
        ----------
        z : np.ndarray
            Spatial positions [m]
        x_a0 : float
            Initial mole fraction of component A
        D_AB : float
            Binary diffusion coefficient [m²/s]
        t : float
            Time [s]

        Returns
        -------
        np.ndarray
            Mole fraction of component A as a function of position z

        Notes
        -----
        The solution has the form:
        x(z, t) = x_a0 * (1 - erf(Z-phi))/(1 + erf(phi))
        where Z = z / sqrt(4*D_AB*t) and phi is a parameter solved implicitly.
        """
        # Define helper function to solve for phi

        # Calculate Z parameter
        Z = z / np.sqrt(4.0 * D_AB * t)
        phi = xa0_to_phi(x_a0)
        # Calculate the concentration profile
        numerator = 1.0 - erf(Z - phi)
        denominator = 1.0 + erf(phi)

        return x_a0 * numerator / denominator
    return


@app.cell(hide_code=True)
def _(np, solve_ivp):
    def evaporate_stagnantB_mol(
        t_span: tuple[float, float],
        L: float,
        x_a0: float = 0.9,
        x_aL: float = 0.0,
        x_init: np.ndarray = None,
        D_AB: float = 2e-5,
        p: float = 1.0,  # pressure in atm
        T: float = 298.15,  # temperature in K
        method: str = "BDF",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        z: np.ndarray = None,
        N: int = 200,
        return_history: bool = True,
    ):
        """
        Method-of-lines (finite differences in z, ODE in t) for

            ∂x/∂t = [ (D_AB/(1-x0)) (∂x/∂z|_{z=0}) ] ∂x/∂z + D_AB ∂²x/∂z²

        on a finite domain z ∈ [0, L] with Dirichlet BCs:
            x(0,t) = x0,   x(L,t) = xL

        Parameters
        ----------
        t_span : (t0, tf)
            Time interval to integrate over.
        z : ndarray, shape (N+1,), optional
            Spatial grid including endpoints (must be uniform).
            If not provided, will be constructed using L and N.
        x0, xL : float
            Boundary mole fractions at z=0 and z=L.
        x_init : ndarray, shape (N+1,), optional
            Initial condition x(z,0) on the full grid (endpoints will be overridden by BCs).
            If not provided, will be initialized to xL with x[0]=x0.
        D_AB : float
            Binary diffusivity (constant).
        p : float
            Total pressure in atm.
        T : float
            Temperature in K.
        method : str
            solve_ivp method, e.g. "RK45" (nonstiff) or "BDF" (stiff).
        rtol, atol : float
            solve_ivp tolerances.
        L : float
            Domain length if z is not provided.
        N : int
            Number of intervals if z is not provided (N+1 grid points).
        return_history : bool
            If True, return the full time history of x at all grid points.

        Returns
        -------
        If return_history is False:
            tuple (sol, NA, NB)
                sol : OdeResult
                    solve_ivp solution object. sol.y contains interior states (nodes 1..N-1).
                NA : ndarray
                    Molar flux of component A on the z grid (kgmol/m²/s)
                NB : ndarray
                    Molar flux of component B on the z grid (kgmol/m²/s)
        If return_history is True:
            tuple (z, t, x, NA, NB)
                z : spatial grid
                t : time points
                x : concentration array with shape (len(t), len(z))
                NA : molar flux of component A with shape (len(t), len(z)) (kgmol/m²/s)
                NB : molar flux of component B with shape (len(t), len(z)) (kgmol/m²/s)
        """
        # Create grid if not provided
        if z is None:
            z = np.linspace(0, L, N + 1)
        else:
            z = np.asarray(z, dtype=float)

        # Create initial condition if not provided
        if x_init is None:
            x_init = np.full_like(z, x_aL)
            x_init[0] = x_a0
        else:
            x_init = np.asarray(x_init, dtype=float)

        if z.ndim != 1 or x_init.ndim != 1:
            raise ValueError("z and x_init must be 1D arrays.")
        if z.size != x_init.size:
            raise ValueError("z and x_init must have the same length.")
        if z.size < 3:
            raise ValueError("Need at least 3 grid points (N>=2).")

        dz = z[1] - z[0]
        if not np.allclose(np.diff(z), dz, rtol=0, atol=1e-12):
            raise ValueError("z grid must be uniform.")

        N = z.size - 1  # nodes 0..N
        # Unknowns are interior nodes 1..N-1
        y0 = x_init[1:N].copy()

        # Gas constant R in J/(kmol·K)
        R = 8314.0

        # Total concentration c = P/(RT) in kmol/m³
        c = p * 101325 / (R * T)  # Convert p from atm to Pa (1 atm = 101325 Pa)

        def rhs(t, y):
            # Reconstruct full x including Dirichlet boundaries
            # RHS for the x(z) profile
            x = np.empty(N + 1, dtype=float)
            x[0] = x_a0
            x[N] = x_aL
            x[1:N] = y

            # Surface gradient (one-sided)
            dx_dz_0 = (x[1] - x[0]) / dz

            # Drift coefficient u(t)
            u = (D_AB / (1.0 - x_a0)) * dx_dz_0

            dydt = np.empty_like(y)

            # Interior updates, central differences
            # i runs 1..N-1 in full grid -> j runs 0..N-2 in y
            for j, i in enumerate(range(1, N)):
                # second derivative
                d2 = (x[i + 1] - 2.0 * x[i] + x[i - 1]) / dz**2
                # first derivative
                d1 = (x[i + 1] - x[i - 1]) / (2.0 * dz)
                dydt[j] = D_AB * d2 + u * d1

            return dydt

        sol = solve_ivp(
            rhs,
            t_span=t_span,
            y0=y0,
            method=method,
            rtol=rtol,
            atol=atol,
            vectorized=False,
            t_eval=np.linspace(t_span[0] + 1, t_span[-1], 5),
        )

        # Get x values from the solution
        if not return_history:
            x_values = reconstruct_x_profile(sol.y[:, -1], x_a0, x_aL, N)
            # Calculate fluxes at final time
            NA_final, NB_final = calculate_fluxes(x_values, dz, D_AB, x_a0, c)
            return z, t_span[-1], x_values, NA_final, NB_final
        else:
            # Get full time history
            z, t, x_history = reconstruct_x_history(z, sol.t, sol.y, x_a0, x_aL)
            # Calculate fluxes for all time points
            NA_history = np.zeros_like(x_history)
            NB_history = np.zeros_like(x_history)

            for i in range(len(t)):
                NA_history[i], NB_history[i] = calculate_fluxes(
                    x_history[i], dz, D_AB, x_a0, c
                )

            return z, t, x_history, NA_history, NB_history


    def calculate_fluxes(
        x_values: np.ndarray, dz: float, D_AB: float, x_a0: float, c: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate molar fluxes for component A and B based on concentration profile.

        Parameters
        ----------
        x_values : ndarray
            Concentration profile array
        dz : float
            Grid spacing
        D_AB : float
            Binary diffusivity
        x_a0 : float
            Boundary mole fraction at z=0
        c : float
            Total concentration

        Returns
        -------
        tuple (NA, NB)
            NA : ndarray
                Molar flux of component A
            NB : ndarray
                Molar flux of component B
        """
        # Calculate the gradients at each point
        dx_dz = np.zeros_like(x_values)
        # 1st derivative
        dx_dz[1:-1] = (x_values[2:] - x_values[:-2]) / (2.0 * dz)
        # boundary dxdz
        dx_dz[0] = (x_values[1] - x_values[0]) / dz
        dx_dz[-1] = (x_values[-1] - x_values[-2]) / dz

        # Surface gradient
        dx_dz_0 = dx_dz[0]

        # Calculate NA flux
        NA = -c * D_AB * dx_dz + x_values * ((-c * D_AB) / (1.0 - x_a0) * dx_dz_0)

        # NB = -NA (for equimolar counter-diffusion)
        NT = (-c * D_AB) / (1.0 - x_a0) * dx_dz_0

        NB = NT - NA

        return NA, NB


    def reconstruct_x_profile(
        y: np.ndarray, x_a0: float, x_aL: float, N: int
    ) -> np.ndarray:
        """
        Reconstruct the full concentration profile including boundary conditions.

        Parameters
        ----------
        y : ndarray
            Interior concentration values
        x_a0 : float
            Boundary value at z=0
        x_aL : float
            Boundary value at z=L
        N : int
            Total number of intervals

        Returns
        -------
        ndarray
            Full concentration profile including boundaries
        """
        x_values = np.empty(N + 1, dtype=float)
        x_values[0] = x_a0
        x_values[N] = x_aL
        x_values[1:N] = y
        return x_values


    def reconstruct_x_history(
        z: np.ndarray, t: np.ndarray, y: np.ndarray, x_a0: float, x_aL: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct the full concentration history from solution.

        Parameters
        ----------
        z : ndarray
            Spatial grid
        t : ndarray
            Time points
        y : ndarray
            Solution array from solve_ivp (interior points only)
        x_a0 : float
            Boundary value at z=0
        x_aL : float
            Boundary value at z=L

        Returns
        -------
        tuple (z, t, x_history)
            z : spatial grid
            t : time points
            x_history : concentration array with shape (len(t), len(z))
        """
        x_history = np.empty((len(t), len(z)))

        # Set boundary conditions for all time points
        x_history[:, 0] = x_a0
        x_history[:, -1] = x_aL

        # Fill in interior points from the solution
        x_history[:, 1:-1] = y.T

        return z, t, x_history
    return (evaporate_stagnantB_mol,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(L_slider, solvent_dropdown, solvents, time_slider):
    def system_information() -> dict:
        # Get selected solvent properties
        try:
            selected_solvent = next(
                (s for s in solvents if s["name"] == solvent_dropdown.value),
                solvents[0],
            )  # Default to first solvent if not found
        except (StopIteration, IndexError):
            # Fallback to first solvent if there's an issue
            selected_solvent = (
                solvents[0]
                if solvents
                else {"name": "Unknown", "p_vap": 0, "D_AB": 0}
            )

        p_vap = selected_solvent["p_vap"]  # Pa
        D_AB = selected_solvent["D_AB"]  # m^2/s

        L = L_slider.value  # m
        time = time_slider.value  # s

        # Constants
        R = 8.314  # J/(mol K)
        T = 298  # K (assume room temp)
        P_total = 101325  # Pa (1 atm)

        # Initial and boundary conditions
        # c_A at z=0 (liquid surface): c_As = p_vap/(R*T) in kg mol/m^3
        # c_A at z=L (open end): c_A = 0
        c_a0 = p_vap / (R * T)
        x_a0 = p_vap / P_total

        return {
            "solvent": selected_solvent["name"],
            "p_vap": p_vap,
            "D_AB": D_AB,
            "L": L,
            "time": time,
            "R": R,
            "T": T,
            "P_total": P_total,
            "c_A0": c_a0,
            "x_a0": x_a0,
        }
    return (system_information,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
