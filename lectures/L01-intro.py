import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="full",
    layout_file="layouts/L01-intro.slides.json",
    css_file="custom_presentation.css",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # **CHE 318 Mass Transfer**
    ## Lecture 01: Introduction and Diffusion Laws

    Dr. Tian Tian<br>Jan 05, 2026<br>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Land Acknowledgement

    <h5> The University of Alberta acknowledges that we are located on Treaty 6 territory,
    and respects the histories, languages, and cultures of the First Nations, Métis, Inuit, and all First Peoples of Canada, whose presence continues t enrich our vibrant community. </h5>
    """)
    return


@app.cell
def _(mo):
    _learning_outcomes = mo.md(
        r"""
        ## Learning Outcomes

        After today's lecture, you will be able to:

        - **Identify** the key components of the course syllabus, content and grading schemes.
        - **Recall** common interaction methods and resources available in the course.
        - **Define** basic concepts in mass transfer, including concentration, mass fraction, and molar flux.
        - **State** Fick's First Law of Diffusion and **identify** its components and units
        - **Describe** the general shape of solutions for stationary Fick's Law.
        """
    )
    _learning_outcomes
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Course Information

    - **Course:** CH E 318 – Mass Transfer
    - **Term:** Winter 2026
    - **Lectures:** Mon, Wed, Fri
    - **Time:** 10:00 – 10:50
    - **Location:** MEC 3-1
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Meet the Instructor

    **Office**: DICE 12-245

    **Email**: tian.tian@ualberta.ca

    **Office hour**: by appointment (will send survey in Canvas)


    - I joined CME in 2025. CHE 318 is my first course at UofA
    - Research fields: machine learning, multiscale materials simulations, computational tools
    - Let’s enjoy learning together!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## TAs & Seminar Sessions

    - **Teaching Assistants**
      - Ethan Lockwood — elockwoo@ualberta.ca
      - Prince (Nkenna) Ezeano — ezeano@ualberta.ca

    - **Seminar (Lab) Session**
      - **Time:** Tuesday 15:30 – 17:20
      - **Location:** MEC 4-3

    - **What seminars are for**
      - Problem-solving practice
      - Worked examples and discussion
      - Concept clarification and Q&A
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Course Grading

    - **Assignments:** 25%
      - 8 total (best 7 counted)
      - Deadlines to be posted on Canvas.

    - **Midterm Exam:** 30%
      - In class (**50 min**), closed book
      - Time to be determined by end of Janurary 2026

    - **Final Exam:** 45%
      - In person
      - Scheduled **Apr 15, 2026 · 8:30 a.m.**

    *Details please see the [course syllabus](https://canvas.ualberta.ca/courses/31008/files/6950659?module_item_id=3529042)*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Textbook

    Our primary textbook for this course is:

    **Transport Processes and Separation Process Principles (Includes Unit Operations), 4th Edition** by Christie J. Geankoplis.

    <div style="width: 50%; text-align: center;">
      <img
        src="public/L01/300w.jpeg"
        alt="Geankoplis 4th Edition Textbook Cover"
        style="width: 300px; height: auto; display: block; margin: 0 auto;"
      />
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What do Mass Transfer Study (1)?
    Scent diffuser:
    \(
    J = -D \nabla C
    \)
    <div style="width: 100%;">
      <img
        src="public/L01/scent-diffuser.jpg"
        style="
          width: 100%;
          max-height: 80vh;
          object-fit: cover;
          display: block;
        "
      />
    </div>
    <!-- ![scent-diffuser](public/L01/scent-diffuser.jpg) -->
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What do Mass Transfer Study (2)?

    Thin-membrane gas exchange:
    \(
    N_{A} = D_{A} \dfrac{C_{A,1} - C_{A,2}}{\delta}
    \)

    <div style="width: 100%;">
      <img
        src="public/L01/gas-exchange.png"
        style="
          width: 100%;
          max-height: 80vh;
          object-fit: cover;
          display: block;
        "
      />
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What do Mass Transfer Study (3)?

    Packed tower: \(
    N_{A} = K_y (y_{AG} - y_{A}^*)
    \)

    <div style="width: 100%;">
      <img
        src="public/L01/packed-tower.png"
        style="
          width: 100%;
          max-height: 80vh;
          object-fit: cover;
          display: block;
        "
      />
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What do Mass Transfer Study (4)?

    Manhattan project ☢️: \(
    D \propto 1/\sqrt{M}
    \)

    <div style="width: 100%;">
      <img
        src="https://www.osti.gov/opennet/manhattan-project-history/Processes/UraniumSeparation/images/k25-diffusion.jpg"
        style="
          width: 100%;
          max-height: 80vh;
          object-fit: cover;
          display: block;
        "
      />
    </div>
    """)
    return


@app.cell
def _(mo):
    _pdf = mo.pdf(src="public/L01/l01-wooclap-results.pdf",width="80%")

    _md = mo.md(f"""
    ## Let’s Get to Know You!

    We will use Wooclap in this course!

    Participation link: https://app.wooclap.com/318L01?from=instruction-slide

    *Results to be published after the class*
    """)

    mo.vstack([_md, _pdf])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Basic Quantities and Their Units in Mass Transfer

    We will only deal with **SI units** in this course!

    - (Molar) Concentraion of A: $c_{A}$
      <span style="margin-left: 1em;"></span>
      Unit: $\text{mol}\cdot\text{m}^{-3}$ or $\text{kg mol}\cdot\text{m}^{-3}$
        - Total concentration: $c_{T} = \sum_i c_{i} = c_A + c_B + \cdots$
    - Molar fraction of A: $y_{A} = \dfrac{c_A}{c_T}$
    - Molar ratio (or binary mixture): $Y_A = \dfrac{y_A}{y_B} = \dfrac{y_A}{1 - y_A}$
        - For ideal gas: $y_A = \dfrac{p_A}{p_T}$ (partial pressure of A and total pressure, respectively)
    - Mass concentraion of A: $\rho_A$<span style="margin-left: 1em;"></span>Unit: $\text{kg}\cdot\text{m}^{-3}$
        - Total mass concentration $\rho_{T} = \sum_i \rho_{i} = \rho_A + \rho_B + \cdots$
        - $c_i = \dfrac{\rho_i}{m_i}$ ($m_i$: molecular weight<span style="margin-left: 1em;"></span>unit $\text{kg}\cdot\text{kg mol}^{-1}$ **Non-SI!**)
    - Mass fraction of A: $w_A = \dfrac{\rho_A}{\rho_T}$, $\sum_i w_i = 1$
    """)
    return


@app.cell
def _(mo):
    _m1 = mo.md(r"""
    ## Introduction to Molecular Transport Equations

    <div style="width: 80%;">
      <img
        src="public/L01/geankoplis-2_3_1-fig.png"
        style="
          width: 100%;
          max-height: 80vh;
          object-fit: cover;
          display: block;
        "
      />
    </div>

    Generatl molecular transport equation (*Geankoplis 4ed, eq 2.3-1*)

    \(
    \text{[rate of transfer process]} = 
    \dfrac{\text{[driving force]}}{\text{[resistance]}}
    \)
    or
    \(
    \psi_z = 
    -\delta \dfrac{d \Gamma}{dz}
    \)

    Properties to be transferred:
    - Momentum (fluid mechanics)
    - Heat (thermodynamics)
    - **Species** (**mass transfer**)

    Units:
    - Flux / rate ($\psi_z$): $\text{[property]}\cdot{\text{m}^{-2}}\cdot{\text{s}^{-1}}$
    - Concentration of property $\Gamma$: $\text{[property]}\cdot{\text{m}^{-3}}$
    """)
    _acc = mo.accordion({r"What is the unit of $\delta$?":
                       r"$\text{m}^{2}\cdot{s^{-1}}$",
                       r"Why is there a negative sign?": "To ensure flux is positive."})
    mo.vstack([_m1, _acc])
    return


@app.cell
def _(mo):
    _md1 = mo.md(r"""
    ## Fick's 1st Law of Diffusion

    <h5>Diffusion flux of A in B, z-direction</h5>
    \(
    \Huge J^*_{Az} = - D_{AB} \dfrac{d c_A}{d z}
    \) 
    """)

    _md2 = mo.image(src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Adolf_Fick_8bit_korr_klein1.jpg/500px-Adolf_Fick_8bit_korr_klein1.jpg",
                  caption="Adolf Fick, German physiologist")
    _md3 = mo.md(r"""
    - Fick was the first to propose the relation between diffusion and concentration gradient driving force
    - Analogous to heat (Fourier's law) and momentum (Newton equation)
    - The coefficient, diffusivity $D_{AB}$, was later linked to molecular Brownian motion by Albert Einstein
    - $D_{AB}$ has unit of $\text{m}^2\cdot\text{s}^{-1}$
    """)

    mo.vstack([
        _md1,
        mo.hstack([_md2, _md3], widths=[1, 3])
    ])
    return


@app.cell
def _(mo):
    _md1 = mo.md("## Behaviour of Molecular Diffusion")
    _img = mo.image(src="https://upload.wikimedia.org/wikipedia/commons/4/4d/DiffusionMicroMacro.gif",
            caption="Diffusion and Brownian motion of molecules, adapted from wikipedia")
    _md2 = mo.md(r"""
    - Brownian motion (Bm) is the inherent motion of molecules
    - Bm leads to redistribution of molecules until $d c_{A}/dz = 0$
    - Non-zero flux only when $d c_{A}/dz \neq 0$
    - Is diffusivity temperature dependent? pressure dependent?
    - How fast is molecular diffusion?
    """)
    mo.vstack([
        _md1,
        mo.hstack([_img, _md2], widths=[2, 3])
    ])
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import plotly.graph_objects as go

    # Constants
    _R_gas_constant: float = 8.314  # J/(mol·K)
    _ref_temp: float = 298.15  # K (25 °C)

    # Reference diffusivities at _ref_temp (m^2/s)
    _D_gas_ref: float = 1e-5
    _D_liquid_ref: float = 5e-10
    _D_solid_ref: float = 1e-14

    # Temperature dependency parameters
    # For gases: D ~ T^exponent
    _gas_exponent: float = 1.75
    # For liquids/solids: Arrhenius-like D ~ exp(-Ea/RT)
    _Ea_liquid: float = 15_000  # J/mol (typical for diffusion in water)
    _Ea_solid: float = 50_000  # J/mol (typical for diffusion in solids)

    # Temperature slider


    def calculate_diffusivity_gas(temp: float) -> float:
        """Calculate gas diffusivity based on temperature."""
        return _D_gas_ref * (temp / _ref_temp)**_gas_exponent

    def calculate_diffusivity_liquid(temp: float) -> float:
        """Calculate liquid diffusivity based on temperature."""
        return _D_liquid_ref * np.exp(
            (_Ea_liquid / _R_gas_constant) * (1 / _ref_temp - 1 / temp)
        )

    def calculate_diffusivity_solid(temp: float) -> float:
        """Calculate solid diffusivity based on temperature."""
        return _D_solid_ref * np.exp(
            (_Ea_solid / _R_gas_constant) * (1 / _ref_temp - 1 / temp)
        )

    # Calculate diffusivities at the current slider temperature
    # _D_gas_current: float = calculate_diffusivity_gas(current_temp.value)
    # _D_liquid_current: float = calculate_diffusivity_liquid(current_temp.value)
    # _D_solid_current: float = calculate_diffusivity_solid(current_temp.value)

    # Display current values
    # _display_values = mo.md(
    #     f"""
    #     ### Diffusivity at {current_temp.value:.2f} K
    #     ({current_temp.value - 273.15:.2f} °C)
    #     - **Gas:** `{_D_gas_current:.2e}` m²/s
    #     - **Liquid:** `{_D_liquid_current:.2e}` m²/s
    #     - **Solid:** `{_D_solid_current:.2e}` m²/s
    #     """
    # )

    # Generate data for plotting over the temperature range
    _temps_for_plot: np.ndarray = np.linspace(273.15, 373.15, 100)
    _D_gas_plot: np.ndarray = np.array(
        [calculate_diffusivity_gas(t) for t in _temps_for_plot]
    )
    _D_liquid_plot: np.ndarray = np.array(
        [calculate_diffusivity_liquid(t) for t in _temps_for_plot]
    )
    _D_solid_plot: np.ndarray = np.array(
        [calculate_diffusivity_solid(t) for t in _temps_for_plot]
    )

    # Create Plotly figure
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=_temps_for_plot,
        y=_D_gas_plot,
        mode='lines',
        name='Gas',
        line=dict(color='blue'),
        hovertemplate=(
            'Temperature: %{x:.2f} K<br>'
            'Diffusivity: %{y:.2e} m²/s<extra>Gas</extra>'
        )
    ))
    _fig.add_trace(go.Scatter(
        x=_temps_for_plot,
        y=_D_liquid_plot,
        mode='lines',
        name='Liquid',
        line=dict(color='red'),
        hovertemplate=(
            'Temperature: %{x:.2f} K<br>'
            'Diffusivity: %{y:.2e} m²/s<extra>Liquid</extra>'
        )
    ))
    _fig.add_trace(go.Scatter(
        x=_temps_for_plot,
        y=_D_solid_plot,
        mode='lines',
        name='Solid',
        line=dict(color='green'),
        hovertemplate=(
            'Temperature: %{x:.2f} K<br>'
            'Diffusivity: %{y:.2e} m²/s<extra>Solid</extra>'
        )
    ))

    _fig.update_layout(
        title="Diffusivity vs. Temperature for Different Phases",
        xaxis_title="Temperature (K)",
        yaxis_title="Diffusivity (m²/s)",
        yaxis_type="log",  # Use log scale for y-axis due to large range
        hovermode="x unified",
        legend_title="Phase",
        height=500,
        template="plotly_white"
    )
    D_AB_plot = _fig
    return (D_AB_plot,)


@app.cell
def _(D_AB_plot, mo):
    _md1 = mo.md(r"""
    ## Typical $D_{AB}$ Range
    """)
    mo.vstack([_md1, D_AB_plot])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Different forms of Fick's first law

    - Original form:
      \(
      J^*_{Az} = - D_{AB} \dfrac{d c_A}{d z}
      \)
    - In molar fraction:
      \(
      J^*_{Az} = - c_{T}D_{AB} \dfrac{d x_A}{d z}
      \)
    - Diffusive velocity:
      \(
      J^*_{Az} = v_{Ad}\overline{c}_{A}
      \)
      - $v_{Ad}$: diffusive velocity of A ($\text{m}\cdot\text{s}^{-1}$)
      - $\overline{c}_{A}$: average concentration of A ($\text{mol}\cdot\text{m}^{-3}\cdot\text{s}^{-1}$)
      - Which reference frame is used?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Combining Diffusion and Convection

    - Real-world mass transfer scenarios have diffusion + convection
    - Using a lab frame, the total mass transfer flux can be written in diffusive + fluid velocities


    The **total molar flux** of species A, \(N_A\), can be expressed as:

    \(
    \Large N_A = J_{Az}^{*} + c_A v_m
    \)

    where:
    - \(N_A\) is the total molar flux of A (\(\text{mol}\cdot\text{m}^{-2}\cdot\text{s}^{-1}\))
    - \(J_{Az}^{*}\) is the diffusive molar flux of A relative to the molar average velocity (\(\text{mol}\cdot\text{m}^{-2}\cdot\text{s}^{-1}\))
    - \(c_A\) is the molar concentration of A (\(\text{mol}\cdot\text{m}^{-3}\))
    - \(v_m\) is the molar average velocity of the mixture (\(\text{m}\cdot\text{s}^{-1}\))

    For **binary mixture** of A and B, we have total flux for 2 components $N$:

    \(
    \Large N = N_A + N_B = v_m c = v_m (c_A + c_B)
    \)

    Substituting $v_m$ we get

    \(
    \Large N_A = J_{Az}^* + \dfrac{c_A}{c}(N_A + N_B)
    \)

    which further rewrites to

    \(
    \Large N_A = -c D_{AB}\dfrac{d x_A}{dz} + \dfrac{c_A}{c}(N_A + N_B)
    \)

    This is the **governing equation** for all steady-state mass transfer in the following lectures!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Brief Introduction to Course AI Helper

    - A Socratic Gemini chatbot aiming to help course learning and key concepts
    - **Access the AI helper here:**
    👉 https://gemini.google.com/gem/f2d47200f0bf
    <div style="width: 100%; text-align: center;">
      <img
        src="https://quickchart.io/qr?text=https://gemini.google.com/gem/f2d47200f0bf&size=150"
        alt="QR Code for AI Helper"
        style="width: 150px; height: 150px; display: block; margin: 0 auto;"
      />
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    What we learned today:
    - Syllabus / course contents of CHE 318
    - Basic concepts in mass transfer
    - Fick's 1st law of diffusion
    - Governing equation of mass transfer

    **See you next time!**
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
