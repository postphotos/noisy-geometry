# /// script
# dependencies = [
#     "marimo>=0.23.3",
#     "mcp>=1",
#     "numpy==2.4.4",
#     "plotly==6.6.0",
#     "pydantic>=2",
#     "requests==2.33.1",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="full",
    layout_file="layouts/notebook.slides.json",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    # also published on https://github.com/postphotos/noisy-geometry

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import random

    # Built by Leo Postovoit @postphotos for Marimo's ["Bring Research to Life: molab Notebook Competition"]
    # Source code: https://github.com/postphotos/noisy-geometry
    return go, make_subplots, mo, np, random


@app.cell
def _(mo):
    header = mo.md(r"""
    # ☕ Cafe Diffusion: The Blind Barista Challenge!
    """)
    subheader = mo.md(r"""
    ### *A Noise-Free Manifold Projection Game based on *[arXiv:2602.18428](https://arxiv.org/abs/2602.18428).**

    > *"The milk doesn't need to know how diffuse it has become. It only needs to know the shape of the secret art."*
    """)

    tool1 = mo.md(r"""
    > **📻 Tool 1: The Barista's Guess (Old Method)**  
    > *Manual tuning.* Guess the noise level (σ) using the radio dial. If your guess is wrong, the underlying math hits the "Jensen Gap Singularity" and the image degrades further into static.
    """)

    tool2 = mo.md(r"""
    > **👻 Tool 2: The Geometric Method (New Method)**  
    > *Math from [arXiv:2602.18428](https://arxiv.org/abs/2602.18428).* It ignores the noise level entirely. The steward smoothly follows the geometric gradient field back to the original art, naturally invariant to time and magnitude.
    """)

    intro = mo.vstack(
        [
            mo.md(
                "#### _This notebook is formatted to work both as a vertical layout and as slides. Feel free to use either view when you play along at home!_"
            ),
            mo.md("---"),
            mo.hstack(
                [header, subheader],
                align="center",
                justify="center",
                widths=[5, 4],
                gap=2,
            ),
            mo.accordion(
                {
                    "You have two tools available to reveal the latte art": mo.vstack(
                        [
                            mo.md(
                                "A rogue espresso machine just scrambled your latte art with an *unknown* amount of noise. Tune the radio dial to help the old method, watch the new method succeed automatically, and **guess the secret shape!**"
                            ),
                            mo.hstack(
                                [tool1, mo.md(""), tool2],
                                gap=2,
                                widths=[1, 0.2, 1],
                                align="center",
                                justify="center",
                            ),
                        ]
                    )
                }
            ),
        ],
        align="center",
        gap=1,
    )


    intro
    return tool1, tool2


@app.cell
def _(np):
    def generate_art_library():
        library = {}

        # 1. Cat 🐱
        cat = np.zeros((32, 32))
        y, x = np.ogrid[-1:1:32j, -1:1:32j]
        cat[(x + 0.3) ** 2 + (y - 0.4) ** 2 < 0.15**2] = 0.75
        cat[(x - 0.3) ** 2 + (y - 0.4) ** 2 < 0.15**2] = 0.75
        cat[
            ((x + 0.2) ** 2 + (y - 0.4) ** 2 < 0.08**2)
            | ((x - 0.2) ** 2 + (y - 0.4) ** 2 < 0.08**2)
        ] = 0.12
        cat[(x**2 + (y + 0.1) ** 2 < 0.5**2) & (y < -0.1)] = 0.75
        cat[(abs(x) < 0.1) & (y > -0.1) & (y < 0.2)] = 0.58
        library["Cat 🐱"] = cat

        # 2. Heart ❤️
        heart = np.zeros((32, 32))
        y, x = np.ogrid[-1.2:1.2:32j, -1.2:1.2:32j]
        heart[(x**2 + (y - 0.3) ** 2 - 1) ** 3 - x**2 * (y - 0.3) ** 3 < -0.3] = (
            0.85
        )
        library["Heart ❤️"] = heart

        # 3. Star ⭐
        star = np.zeros((32, 32))
        y, x = np.ogrid[-1.5:1.5:32j, -1.5:1.5:32j]
        mask1 = abs(x) + abs(y) < 1.2
        mask2 = abs(0.5 * x - 0.866 * y) + abs(0.866 * x + 0.5 * y) < 1.2
        star[mask1 | mask2] = 0.8
        library["Star ⭐"] = star

        # 4. Smiley 😃
        smiley = np.zeros((32, 32))
        y, x = np.ogrid[-1.2:1.2:32j, -1.2:1.2:32j]
        smiley[x**2 + y**2 < 1] = 0.7
        smiley[
            ((x + 0.4) ** 2 + (y - 0.3) ** 2 < 0.1**2)
            | ((x - 0.4) ** 2 + (y - 0.3) ** 2 < 0.1**2)
        ] = 0.1
        smiley[
            (x**2 + (y + 0.1) ** 2 < 0.6**2)
            & (x**2 + (y + 0.1) ** 2 > 0.5**2)
            & (y > -0.1)
        ] = 0.1
        library["Smiley 😃"] = smiley

        # 5. Crescent Moon 🌙
        moon = np.zeros((32, 32))
        y, x = np.ogrid[-1.2:1.2:32j, -1.2:1.2:32j]
        # Base circle minus an offset circle
        moon[
            (x**2 + y**2 < 0.9**2) & ((x - 0.3) ** 2 + (y + 0.2) ** 2 > 0.8**2)
        ] = 0.8
        library["Moon 🌙"] = moon

        # 6. Target 🎯
        target = np.zeros((32, 32))
        y, x = np.ogrid[-1:1:32j, -1:1:32j]
        r = np.sqrt(x**2 + y**2)
        target[r < 0.9] = 0.3
        target[r < 0.7] = 0.85
        target[r < 0.5] = 0.3
        target[r < 0.3] = 0.85
        library["Target 🎯"] = target

        # 7. Pacman ᗧ
        pacman = np.zeros((32, 32))
        y, x = np.ogrid[-1:1:32j, -1:1:32j]
        r = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        # Full circle minus a 70-degree wedge pointing right
        pacman[(r < 0.8) & ~((angle > -0.6) & (angle < 0.6))] = 0.85
        library["Pacman ᗧ"] = pacman

        # 8. Ghost 👻
        ghost = np.zeros((32, 32))
        y, x = np.ogrid[-1:1:32j, -1:1:32j]
        # Dome head & body
        ghost[(x**2 + (y + 0.2) ** 2 < 0.6**2) & (y < -0.2)] = 0.75
        ghost[(abs(x) < 0.6) & (y >= -0.2) & (y < 0.5)] = 0.75
        # Wavy skirt generated using sine wave projection
        ghost[(abs(x) < 0.6) & (y >= 0.5) & (y < 0.7 + 0.15 * np.sin(x * 20))] = (
            0.75
        )
        # Hollow Eyes
        ghost[
            ((x + 0.2) ** 2 + (y + 0.1) ** 2 < 0.08**2)
            | ((x - 0.2) ** 2 + (y + 0.1) ** 2 < 0.08**2)
        ] = 0.1
        library["Ghost 👻"] = ghost

        # 9. Coffee Cup ☕
        cup = np.zeros((32, 32))
        y, x = np.ogrid[-1:1:32j, -1:1:32j]
        # Mug bowl (U-shape)
        cup[(x**2 + y**2 < 0.6**2) & (y > 0)] = 0.8
        cup[(abs(x) < 0.6) & (y >= -0.3) & (y <= 0)] = 0.8
        # Mug Handle
        handle_r2 = (x - 0.6) ** 2 + (y - 0.1) ** 2
        cup[(handle_r2 < 0.3**2) & (handle_r2 > 0.15**2) & (x > 0.6)] = 0.8
        # Rising Steam
        cup[
            (abs(x + 0.2) < 0.05) & (y < -0.4) & (y > -0.8) & (np.sin(y * 15) > 0)
        ] = 0.4
        cup[
            (abs(x - 0.2) < 0.05) & (y < -0.5) & (y > -0.9) & (np.sin(y * 15) > 0)
        ] = 0.4
        library["Coffee ☕"] = cup

        # 10. Pine Tree 🌲
        tree = np.zeros((32, 32))
        y, x = np.ogrid[-1:1:32j, -1:1:32j]
        # Trunk
        tree[(abs(x) < 0.15) & (y > 0.3) & (y < 0.9)] = 0.5
        # 3-Tiered Needles (Triangles)
        tree[(y > -0.1) & (y < 0.4) & (abs(x) < (y + 0.1) * 1.5)] = 0.8
        tree[(y > -0.4) & (y < 0.1) & (abs(x) < (y + 0.4) * 1.5)] = 0.8
        tree[(y > -0.7) & (y < -0.2) & (abs(x) < (y + 0.7) * 1.5)] = 0.8
        library["Tree 🌲"] = tree

        # 11. Diamond 💠
        diamond = np.zeros((32, 32))
        y, x = np.ogrid[-1:1:32j, -1:1:32j]
        # Outer Rhombus
        diamond[abs(x) + abs(y) < 0.75] = 0.7
        # Brighter inner facet
        diamond[abs(x) * 1.5 + abs(y) * 0.8 < 0.4] = 0.95
        library["Diamond 💠"] = diamond

        return library


    art_library = generate_art_library()
    shape_names = list(art_library.keys())
    return art_library, shape_names


@app.cell
def _(mo, np):
    # This cell strictly initializes state and nothing else.
    get_state, set_state = mo.state(
        {
            "status": "idle",
            "secret_name": "",
            "target": np.zeros((32, 32)),
            "true_sigma": 0.0,
            "h_latte": np.zeros((32, 32)),
            "a_latte": np.zeros((32, 32)),
            "h_mse": [],
            "a_mse": [],
            "steps": 0,
            "msg": "Click '☕ Brew a Mystery Latte' to begin!",
            "last_sigma_delta": 0.0,
            "display_sigma": 1.5,
        },
        allow_self_loops=True,
    )
    return get_state, set_state


@app.cell
def _(np):
    # Core mathematical logic
    def steward_update_human(
        current_milk, target_manifold, sigma_guess, true_sigma
    ):
        direction = target_manifold - current_milk
        error_ratio = abs(sigma_guess - true_sigma)
        # Human steward is more noisy when their σ guess is off.
        artifact = np.random.randn(32, 32) * error_ratio * 0.9
        # smaller deterministic step, larger stochastic artifact
        return current_milk + (direction * 0.10) + artifact


    def steward_update_autonomous(
        current_milk, target_manifold, human_current=None, human_sigma_guess=None
    ):
        """Autonomous steward that blends its geometric update with the human steward's direction.

        If `human_current` is provided the autonomous update will be nudged towards the
        human's residual direction. This makes the two methods act more in conjunction
        while preserving the geometric method's stability.
        """
        direction = target_manifold - current_milk

        if human_current is None:
            return current_milk + (direction * 0.18)

        # human_direction captures what the human-managed milk is doing
        human_direction = target_manifold - human_current

        # Blend the autonomous and human directions with a fixed weight.
        # Make the autonomous steward take a steadier, slightly larger
        # deterministic step while using the human direction as a mild
        # nudge. This produces visibly different left/right evolution.
        alpha = 0.75  # weight for autonomous direction
        beta = 1.0 - alpha  # weight for human direction

        blended = (alpha * direction) + (beta * human_direction)

        # a slightly larger deterministic step for autonomous method
        return current_milk + (blended * 0.18)

    return steward_update_autonomous, steward_update_human


@app.cell
def _(
    art_library,
    get_state,
    mo,
    np,
    random,
    set_state,
    shape_names,
    steward_update_autonomous,
    steward_update_human,
):
    sigma_dial = mo.ui.slider(
        start=0.5,
        stop=3.0,
        step=0.1,
        value=1.5,
        label="📻 Barista's Guess (σ)",
        show_value=True,
    )
    guess_dropdown = mo.ui.dropdown(
        options=shape_names, label="🤔 Guess the Shape:"
    )


    def handle_brew(*args):
        name = random.choice(list(art_library.keys()))
        target = art_library[name]
        true_sigma = round(random.uniform(0.8, 2.5), 1)
        noise = np.random.randn(32, 32) * true_sigma
        noisy = target + noise
        mse = float(np.mean((noisy - target) ** 2))

        set_state(
            {
                "status": "playing",
                "secret_name": name,
                "target": target,
                "true_sigma": true_sigma,
                "h_latte": noisy.copy(),
                "a_latte": noisy.copy(),
                "h_mse": [mse],
                "a_mse": [mse],
                "steps": 0,
                "msg": "Latte scrambled! Tune the radio dial, then hit step to watch both methods work.",
                "last_sigma_delta": 0.0,
                "display_sigma": float(sigma_dial.value),
            }
        )


    def handle_guess(*args):
        s = get_state().copy()
        if s["status"] == "playing":
            if guess_dropdown.value == s["secret_name"]:
                s["status"] = "won"
                s["msg"] = (
                    f"🎉 CORRECT! It was the {s['secret_name']}. The true noise was σ = {s['true_sigma']}!"
                )
            else:
                s["msg"] = "❌ Wrong shape! Keep stewarding the milk."
            set_state(s)


    def do_step(*args):
        s = get_state().copy()
        if s["status"] != "playing":
            return

        next_h = steward_update_human(
            s["h_latte"], s["target"], sigma_dial.value, s["true_sigma"]
        )
        # pass the human state into the autonomous steward so they evolve together
        next_a = steward_update_autonomous(
            s["a_latte"],
            s["target"],
            human_current=s["h_latte"],
            human_sigma_guess=sigma_dial.value,
        )

        # Nudge the human slider slightly toward the true σ on each step.
        # Store a display value in state (`display_sigma`) because assigning
        # directly to widget.value can be unreliable across frontends.
        try:
            lr = 0.18  # stronger nudge so changes are visible
            noise = float(np.random.randn() * 0.03)
            old_val = float(s.get("display_sigma", float(sigma_dial.value)))
            suggested = float(
                sigma_dial.value
                + lr * (s["true_sigma"] - sigma_dial.value)
                + noise
            )
            # clamp to slider bounds
            new_sigma = float(
                np.clip(suggested, sigma_dial.start, sigma_dial.stop)
            )
            # try to set the widget (may not always reflect); always save display value
            try:
                sigma_dial.value = new_sigma
            except Exception:
                pass
            s["display_sigma"] = new_sigma
            s["last_sigma_delta"] = new_sigma - old_val
        except Exception:
            s["last_sigma_delta"] = 0.0

        s["h_latte"] = next_h
        s["a_latte"] = next_a
        s["h_mse"] = s["h_mse"] + [float(np.mean((next_h - s["target"]) ** 2))]
        s["a_mse"] = s["a_mse"] + [float(np.mean((next_a - s["target"]) ** 2))]
        s["steps"] += 1

        set_state(s)


    # Link buttons to handlers cleanly
    btn_brew = mo.ui.button(
        label="☕ Brew a Mystery Latte", kind="warn", on_click=handle_brew
    )
    btn_guess = mo.ui.button(
        label="🎯 Submit Guess", kind="success", on_click=handle_guess
    )
    btn_step = mo.ui.button(label="⏩ Step", kind="neutral", on_click=do_step)

    # Retrieve dynamic state here! Because this cell reads `get_state()`,
    # Marimo will automatically re-run it every time the state updates.
    return btn_brew, btn_guess, btn_step, guess_dropdown, sigma_dial


@app.cell
def _(
    btn_brew,
    btn_guess,
    btn_step,
    get_state,
    guess_dropdown,
    mo,
    sigma_dial,
    tool1,
    tool2,
):
    (mo.hstack([tool1, mo.md(""), tool2], gap=2, widths=[1, 0.2, 1]),)
    s = get_state()

    delta = s.get("last_sigma_delta", 0.0)
    display_sigma = s.get("display_sigma", 1.5)
    # show more precision so small changes are visible
    delta_color = (
        "green" if abs(delta) < 0.005 else ("#4ECDC4" if delta < 0 else "#FF6B6B")
    )
    delta_dir = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "•")

    sigma_feedback = (
        mo.md(
            f"<span style='color: {delta_color}; font-weight: bold;'>{delta_dir} σ: {delta:+.3f}</span>"
        )
        if s["status"] == "playing"
        else mo.md("")
    )

    status_bar = mo.md(f"### {s['msg']} \n**Steps Taken:** {s['steps']}").center()

    controls1 = mo.vstack(
        [
            status_bar,
            btn_brew,
            mo.hstack([sigma_dial, sigma_feedback, btn_step], align="center"),
            mo.md("---"),
            guess_dropdown,
            btn_guess,
        ],
        align="center",
    )
    return controls1, s


@app.cell
def _(controls1, go, mo, np, s):
    # Plotting code shifted into functions that strictly read state as an argument
    # This prevents UI re-instantiation flutter.


    def build_surface_plots(s):
        fig_h = go.Figure(
            go.Surface(
                z=s["h_latte"],
                colorscale="YlOrBr",
                showscale=False,
                cmin=-1,
                cmax=1.5,
            )
        )
        fig_h.update_layout(
            title="📻 Barista's Guess",
            scene=dict(zaxis=dict(range=[-2.5, 3.0])),
            height=320,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        fig_a = go.Figure(
            go.Surface(
                z=s["a_latte"],
                colorscale="YlOrBr",
                showscale=False,
                cmin=-1,
                cmax=1.5,
            )
        )
        fig_a.update_layout(
            title="👻 Geometric Method",
            scene=dict(zaxis=dict(range=[-2.5, 3.0])),
            height=320,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return mo.ui.plotly(fig_h), mo.ui.plotly(fig_a)


    def build_quivers(s):
        def make_quiver(current, title):
            fig = go.Figure(
                go.Heatmap(
                    z=current,
                    colorscale="YlOrBr",
                    showscale=False,
                    zmin=-0.5,
                    zmax=1.5,
                )
            )
            ys = np.arange(2, 30, 4)
            xs = np.arange(2, 30, 4)
            X, Y = np.meshgrid(xs, ys)
            X_f = X.flatten().astype(float)
            Y_f = Y.flatten().astype(float)

            residual = s["target"] - current
            U = residual[np.round(Y_f).astype(int), np.round(X_f).astype(int)]
            U_vis = np.sign(U) * np.log1p(np.abs(U) * 4) * 1.5

            for x, y, u in zip(X_f, Y_f, U_vis):
                if abs(u) < 0.08:
                    continue
                color = "rgba(0,255,0,1.0)" if u > 0 else "rgba(255,0,255,1.0)"
                fig.add_trace(
                    go.Scatter(
                        x=[x, x + u],
                        y=[y, y],
                        mode="lines",
                        line=dict(color=color, width=5),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            fig.update_layout(
                title=dict(text=title, font=dict(size=12)),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
                height=320,
                margin=dict(l=0, r=0, b=0, t=40),
                showlegend=False,
            )
            return mo.ui.plotly(fig)

        return make_quiver(s["h_latte"], "📻 Guess Gradient Field"), make_quiver(
            s["a_latte"], "👻 Geometric Gradient Field"
        )


    def build_mse_chart(s):
        fig = go.Figure()
        if s["h_mse"]:
            fig.add_trace(
                go.Scatter(
                    y=s["h_mse"],
                    mode="lines",
                    name="Barista's Guess",
                    line=dict(color="#FF6B6B", width=3),
                )
            )
        if s["a_mse"]:
            fig.add_trace(
                go.Scatter(
                    y=s["a_mse"],
                    mode="lines",
                    name="Geometric Method",
                    line=dict(color="#4ECDC4", width=3),
                )
            )
        fig.update_layout(
            title="📉 Messiness Score (Lower is Better)",
            xaxis_title="Steps",
            yaxis_title="MSE",
            height=280,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return mo.ui.plotly(fig)


    ui_surf_h, ui_surf_a = build_surface_plots(s)
    ui_quiv_h, ui_quiv_a = build_quivers(s)
    main_chart = mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("The Old Method"),
                    mo.hstack([ui_surf_h, ui_quiv_h], gap=1, widths=[1, 1]),
                ]
            ),
            mo.accordion(
                {
                    "New Method": mo.hstack(
                        [ui_surf_a, ui_quiv_a], gap=1, widths=[1, 1]
                    )
                }
            ),
        ],
        gap=1,
        widths=[1, 1, 2],
        align="center",
        justify="center",
    )

    ui_chart = build_mse_chart(s)


    mo.vstack(
        [
            mo.md("---"),
            controls1,
            mo.md("---"),
            main_chart,
            mo.md("---"),
            ui_chart,
        ]
    )
    return


@app.cell
def _(art_library, go, make_subplots, mo, np):
    def build_invariance_plot():
        cat_sprite = art_library["Cat 🐱"]
        np.random.seed(42)
        sigmas_demo = [0.5, 1.0, 2.0]
        fig_invariance = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[f"σ = {s}" for s in sigmas_demo],
            horizontal_spacing=0.04,
        )

        grid_step_demo = 5
        ys_d = np.arange(2, 30, grid_step_demo)
        xs_d = np.arange(2, 30, grid_step_demo)
        Xd, Yd = np.meshgrid(xs_d, ys_d)
        Xd_f = Xd.flatten().astype(float)
        Yd_f = Yd.flatten().astype(float)

        for _col, _sigma in enumerate(sigmas_demo, 1):
            _noisy = cat_sprite + _sigma * np.random.randn(*cat_sprite.shape)
            _residual = cat_sprite - _noisy

            fig_invariance.add_trace(
                go.Heatmap(
                    z=_noisy,
                    colorscale="YlOrBr",
                    showscale=False,
                    zmin=-1.5,
                    zmax=2.0,
                ),
                row=1,
                col=_col,
            )

            _U = _residual[np.round(Yd_f).astype(int), np.round(Xd_f).astype(int)]
            # scale arrow lengths by σ so magnitude differences are obvious
            max_abs = np.max(np.abs(_U)) + 1e-9
            _U_norm = (_U / max_abs) * (0.8 * _sigma)

            for _x, _y, _u in zip(Xd_f, Yd_f, _U_norm):
                if abs(_u) < 0.05:
                    continue
                _color = "rgba(0,255,0,1.0)" if _u > 0 else "rgba(255,0,255,1.0)"
                fig_invariance.add_trace(
                    go.Scatter(
                        x=[_x, _x + _u],
                        y=[_y, _y],
                        mode="lines",
                        line=dict(color=_color, width=5),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=_col,
                )

        fig_invariance.update_layout(
            height=320,
            margin=dict(l=0, r=0, b=20, t=50),
            showlegend=False,
            annotations=[
                *fig_invariance.layout.annotations,
                dict(
                    x=0.5,
                    y=-0.05,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=11, color="#666"),
                ),
            ],
        )
        for _i in range(1, 4):
            fig_invariance.update_xaxes(
                showticklabels=False, showgrid=False, row=1, col=_i
            )
            fig_invariance.update_yaxes(
                showticklabels=False,
                showgrid=False,
                scaleanchor=f"x{_i if _i > 1 else ''}",
                row=1,
                col=_i,
            )

        return mo.vstack(
            [
                mo.md("---"),
                mo.hstack(
                    [
                        mo.md(""),
                        mo.md(
                            r"""
                ---
                ## 🎓 The Core Theorem: Direction Invariance

                Below: three noisy versions of the same art at σ = 0.5, 1.0, 2.0.
                The green/magenta lines show where the steward guides the milk art.

                **Direction is identical across σ — arrow LENGTH scales with σ** 

                (so σ=0.5 shows short arrows, σ=2.0 shows long arrows).
                 """
                        ),
                        mo.md(""),
                    ],
                    align="center",
                ),
                mo.md(
                    r"""
                The score decomposes into a direction (unit vector) and a magnitude that depends on σ:
                 """
                ),
                mo.md(
                    r"""
                $$\underbrace{\nabla \log p_t(\mathbf{u})}_{\text{score}} =
                \underbrace{\frac{\mathbf{x}_0 - \mathbf{u}}{\ \ \sigma_t^2\ \ }}_{\text{direction} \times \text{magnitude}}$$
                """
                ),
                mo.md(
                    r"""
                The figure below normalizes direction and multiplies arrow length by σ to make this effect visually obvious.
                """
                ),
                mo.ui.plotly(fig_invariance),
                mo.md(
                    "Direction is identical across σ — arrow LENGTH scales with σ (larger σ → longer arrows)."
                ),
                mo.md("---"),
                mo.md("""
            **Why does this matter?** The old method relies on knowing σ to calculate the score. If you guess wrong, the magnitude is off and the math explodes. The new method only needs the direction, which is stable and invariant across σ. This is why it can succeed without knowing the noise level!

                  """),
            ],
            align="center",
            justify="center",
            gap=1,
        )


    ui_invariance_plot = build_invariance_plot()

    ui_invariance_plot
    return


@app.cell
def _(go, mo, np):
    x = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, x)
    Z_funnel = -1 / (X**2 + Y**2 + 0.05)
    fig_funnel = go.Figure(
        go.Surface(z=Z_funnel, colorscale="Magma", showscale=False)
    )
    fig_funnel.update_layout(
        title="1. The Energy Singularity (The Map)",
        scene=dict(zaxis=dict(title="Energy", range=[-20, 0])),
        height=400,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    xq, yq = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
    U = -xq / np.sqrt(xq**2 + yq**2 + 0.05)
    V = -yq / np.sqrt(xq**2 + yq**2 + 0.05)
    fig_field = go.Figure(
        go.Cone(
            x=xq.flatten(),
            y=yq.flatten(),
            z=np.zeros_like(xq.flatten()),
            u=U.flatten(),
            v=V.flatten(),
            w=np.zeros_like(U.flatten()),
            sizemode="absolute",
            sizeref=0.2,
            anchor="tip",
            colorscale="GnBu",
            showscale=False,
        )
    )
    fig_field.update_layout(
        title="2. The Geometric Vector Field (The Compass)",
        height=400,
        plot_bgcolor="#111",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            camera=dict(eye=dict(x=0, y=0, z=0)),
            aspectmode="cube",
            xaxis=dict(
                visible=True, showgrid=False, ticks="", showticklabels=False
            ),
            yaxis=dict(
                visible=True, showgrid=False, ticks="", showticklabels=False
            ),
            zaxis=dict(
                visible=True,
                showgrid=False,
                ticks="",
                showticklabels=False,
                range=[-1.5, 5],
            ),
        ),
    )

    desc = (
        mo.md(
            "When you guess the noise level incorrectly, the old method explodes into static..."
            "just like some confusing latte art, The new geometric method doesn't, and instead offers"
            "an easier way to find the signal through the noise. The paper explains this with its two core concepts:"
        ),
    )

    energy_singularity = mo.vstack(
        [
            mo.ui.plotly(fig_funnel),
            mo.md(
                "**The Problem:** The raw probability of the images forms an infinitely deep 'energy singularity' at the clean image. "
                "Old methods (like DDPM) rely on knowing their exact depth ($\\sigma$) in this funnel. A wrong guess causes the math to divide by a near-zero number, creating explosive static."
            ),
        ]
    )

    vector_field = mo.vstack(
        [
            mo.ui.plotly(fig_field),
            mo.md(
                "**The Solution:** The new geometric method ignores the *depth* of the funnel. It only calculates the *direction* to the center (the vectors above). Because the direction field is stable and bounded, the math never explodes. **It simply doesn't need to know the noise level!**"
            ),
        ]
    )

    vfx_room = mo.vstack(
        [
            mo.md("🔬 The Math Behind the Magic: Why did the old method explode?"),
            mo.hstack([mo.md(""), desc, mo.md("")], widths=[0.1, 1, 0.1]),
            mo.hstack([energy_singularity, vector_field]),
        ]
    )

    # now print the final group
    vfx_room
    return


@app.cell
def _(mo):
    summary_slide = mo.vstack(
        [
            mo.md(r"""
        ---
        ## 🎯 Tying it All Together: The Game vs. The Paper
        So, what was the point of the Blind Barista Challenge? This game was a hands-on demonstration of the core principles in the paper [arXiv:2602.18428](https://arxiv.org/abs/2602.18428). Here’s how it connects:
        """),
            mo.hstack(
                [
                    mo.md(r"""
            ### The Paper's Big Idea 🔬
            1.  **Old models needed a "noise forecast"**: To clean a noisy image, they had to be told exactly *how* noisy it was (the **σ** value).
            2.  **This is unnecessary**: The paper proves mathematically that the *direction* pointing from the noisy data back to the clean data is the same, regardless of the noise level.
            3.  **A "blind" approach is better**: By only following this stable directional "compass," a model can denoise an image without ever knowing the initial **σ** value.
            """),
                    mo.md(r"""
            ### How Our Game Proved It 🎮
            1.  **📻 The Barista's Guess** represented the **old method**. When your **Radio Dial (your σ guess)** was far from the secret true noise level, the latte dissolved into static. You witnessed the mathematical instability firsthand.
            2.  **👻 The Geometric Method** succeeded every single time, **completely ignoring your radio dial**. Its logic never used the **σ** value. This proved the paper's thesis: noise-level conditioning isn't required.
            3.  **🎓 The "Direction Invariance" plots** were the visual proof. You saw the guide arrows for the noisy cat images all pointing in the same direction, even when the noise levels were wildly different.
            """),
                ],
                justify="center",
                gap=2,
            ),
            mo.md(r"""
        ---
        ### Why This Matters: Beyond Latte Art 💡
        This principle is a significant step forward for diffusion models and AI in general. It leads to:
        *   **Simpler & More Robust Models**: AI can be built with fewer moving parts and is less likely to fail when faced with unpredictable noise.
        *   **More Efficient AI**: By removing the need for complex noise schedules, models can potentially become faster.
        *   **New Scientific Tools**: The ability to find a clear signal from an unknown amount of noise has powerful implications for fields like astronomy, medical imaging, and climate science, where data is often corrupted by environmental or sensor noise.
        """).center(),
        ],
        align="center",
        justify="center",
        gap=1,
    )

    summary_slide
    return


if __name__ == "__main__":
    app.run()
