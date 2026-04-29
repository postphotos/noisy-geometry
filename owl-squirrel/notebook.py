# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "mcp>=1",
#     "numpy==2.4.4",
#     "plotly==6.6.0",
#     "polars==1.39.3",
#     "pydantic>=2",
#     "scipy==1.17.1",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    # also published on https://github.com/postphotos/noisy-geometry
    
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from scipy.stats import gaussian_kde

    return gaussian_kde, go, mo, np, pl


@app.cell
def _(mo):
    intro = mo.md("""# 🐿️ The Geometry of Noise: Habitat Recovery 🦉

    This notebook translates the core thesis of the alphaXiv paper **["The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning"](https://arxiv.org/abs/2602.18428)** into a geographic reality. (_p.s. This notebook is formatted for both vertical and slide mode, so feel free to explore both views!_) 

    **The Thesis:** Diffusion models typically need to know exactly *how much* noise was added to an image to denoise it. The authors mathematically prove this is false. The geometry of the data distribution creates a vector field (or 'Energy Attractor'). The authors demonstrate that by following this field's gradients, the noise resolves itself without the model ever needing to be told how 'lost' it is." And by using a `gaussian_kde` to define the "home" (the manifold) and then using gradient ascent to move noisy points toward high-density areas, I have created a 2D visual proof of the paper's N-dimensional calculus.

    **Experiment 1:** We treat Central Park as our 2D Data Manifold. We use the **[NYC Squirrel Census](https://www.thesquirrelcensus.com/)** (reported by humans, not GPS) as our ground truth. You will inject extreme lat/long noise, scattering the squirrels across Manhattan. Using a purely blind gradient flow, we will watch the geometry of the park pull them home and show the cluster always as Central Park.

    **Experiment 2:** We repeat our 2D Data Manifold work, but with a "realer" dataset more susceptable to external Gaussian noise - a [GPS-tracked cluster of barnyard owl breeding locations](https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study1426277950) after wildfires ravaged Napa, California. Again, you will inject extreme GPS noise, scattering the squirrels across Napa. The geometry of the math pulls them back to their home.

    ## Why? 

    Using tools like this means, with careful application, physical and social science researchers are more able to surface likely geographic center, and surface a better triangulated space to explore when considering GPS data (or any other "noisy" dataset).
 
    """)


    intro
    return


@app.cell
def _(np, pl):
    # 1. Fast Data Ingestion with Polars
    # Using the 2019 TidyTuesday NYC Squirrel dataset
    # url1 = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2019/2019-10-29/nyc_squirrels.csv"
    url1 = "nyc_squirrels.csv"

    # We only need coordinates and fur color. Drop nulls to keep the manifold clean.
    df_squirrels = (
        pl.read_csv(url1).select(["lat", "long", "primary_fur_color"]).drop_nulls()
    )

    # Sample down slightly to keep real-time UI smooth (1000 squirrels)
    df_sample = df_squirrels.sample(n=2000, seed=42)

    # Extract coordinate arrays (latitude = lat, longitude = long)
    true_lat = df_sample["lat"].to_numpy()
    true_lon = df_sample["long"].to_numpy()
    fur_colors = df_sample["primary_fur_color"].to_list()

    # Stack as [lon, lat] for KDE calculations
    true_coords = np.vstack([true_lon, true_lat])

    # true_coords
    return df_sample, fur_colors, true_coords, true_lat, true_lon


@app.cell
def _(gaussian_kde, true_coords):
    # 2. Compute the Energy Manifold (Score Field)
    # This represents the actual "Geometry" of the data. We compute it once.
    # bandwidth controls how 'tight' the energy attractor is around the exact trees.
    kde = gaussian_kde(true_coords, bw_method=0.08)
    return (kde,)


@app.cell
def _(np):
    # 3. Vectorized Math: The Score Function
    # Score = Gradient of the log probability density: ∇_x log p(x)
    def get_score_field(points, _kde, eps=1e-5):
        """Computes the gradient of the log-density using central differences."""
        # Baseline density
        p = _kde(points)

        # Longitude shift
        dx = np.array([[eps], [0]])
        dp_dx = (_kde(points + dx) - _kde(points - dx)) / (2 * eps)

        # Latitude shift
        dy = np.array([[0], [eps]])
        dp_dy = (_kde(points + dy) - _kde(points - dy)) / (2 * eps)

        # Score = ∇p / p. Add small constant to prevent division by zero in empty space
        score_x = dp_dx / (p + 1e-8)
        score_y = dp_dy / (p + 1e-8)

        return np.vstack([score_x, score_y])

    return (get_score_field,)


@app.cell
def _(mo):
    # 4. The Interactive UI (Marimo Espresso Dials
    mo.md("### 🎛️ Inject GPS Noise & Control Denoising Physics")

    noise_slider = mo.ui.slider(
        start=0.001,
        stop=0.030,
        step=0.001,
        value=0.001,
        label="🛰️ GPS Noise Level (Degrees)",
    )

    steps_slider = mo.ui.slider(
        start=10, stop=100, step=10, value=50, label="⏱️ Flow Steps (Iterations)"
    )

    step_size_slider = mo.ui.slider(
        start=0.00001,
        stop=0.0005,
        step=0.00001,
        value=0.0001,
        label="🧲 Attractor Strength (Step Size)",
    )

    ui_panel = mo.hstack(
        [noise_slider, steps_slider, step_size_slider], justify="space-around"
    )

    # load the panel later so it appears on notebook slide for map chart
    # ui_panel
    return noise_slider, step_size_slider, steps_slider, ui_panel


@app.cell
def _(
    get_score_field,
    kde,
    noise_slider,
    np,
    step_size_slider,
    steps_slider,
    true_coords,
):
    # 5. Execute the Blind Gradient Flow
    # Add noise to the true coordinates
    noise = np.random.normal(0, noise_slider.value, true_coords.shape)
    noisy_coords = true_coords + noise

    # Store trajectory for the animation
    trajectories_lon = [noisy_coords[0, :].copy()]
    trajectories_lat = [noisy_coords[1, :].copy()]

    # Iterate: Move points along the score field (Gradient Ascent on Log Density)
    # Notice: 'noise_slider.value' is NOT used in the update step. It is unconditioned.
    current_coords = noisy_coords.copy()

    for _ in range(int(steps_slider.value)):
        score = get_score_field(current_coords, kde)

        # Clip the score to prevent extreme jumps when far from the manifold
        norm = np.linalg.norm(score, axis=0)
        max_norm = 500.0
        mask = norm > max_norm
        if mask.any():
            score[:, mask] = score[:, mask] * (max_norm / norm[mask])

        current_coords += step_size_slider.value * score
        trajectories_lon.append(current_coords[0, :].copy())
        trajectories_lat.append(current_coords[1, :].copy())
    return noisy_coords, trajectories_lat, trajectories_lon


@app.cell
def _(
    df_sample,
    fur_colors,
    go,
    mo,
    noisy_coords,
    trajectories_lat,
    trajectories_lon,
    true_lat,
    true_lon,
    ui_panel,
):
    fig = go.Figure()

    # Layer 1: The True Manifold (Green)
    fig.add_trace(
        go.Scattermapbox(
            lat=true_lat,
            lon=true_lon,
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=6, color="rgb(46, 139, 87)", opacity=0.4
            ),
            name="True Habitat Manifold",
            hovertext=fur_colors,
        )
    )

    # Layer 2: The Noisy 'Lost' Points (Red)
    fig.add_trace(
        go.Scattermapbox(
            lat=noisy_coords[1, :],
            lon=noisy_coords[0, :],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=5, color="rgb(220, 20, 60)", opacity=0.8
            ),
            name="Noisy GPS Data",
        )
    )

    # Layer 3: The Denoising Flow (Lines)
    # To avoid rendering 1000s of traces, we will plot a subset of lines
    n_lines = 100
    for i in range(n_lines):
        fig.add_trace(
            go.Scattermapbox(
                lat=[step[i] for step in trajectories_lat],
                lon=[step[i] for step in trajectories_lon],
                mode="lines",
                line=dict(width=1, color="rgba(100, 149, 237, 0.5)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add a dummy trace for the legend of the lines
    fig.add_trace(
        go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode="lines",
            line=dict(width=2, color="rgba(100, 149, 237, 1.0)"),
            name="Gradient Flow Path",
        )
    )

    # Update Mapbox layout using an open-source style that does not require a token
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=12.5,
        mapbox_center={
            "lat": df_sample["lat"].mean(),
            "lon": df_sample["long"].mean(),
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    desc = mo.md(
        # 6.
        """
        ### Build the Visual Story with Plotly: Show the magic of "de-noised noise"
        We will plot 1) The True Manifold, 2) The Noisy Start, 3) The Trajectories

        Observe how when adding GPS noise the squirrels scatter; with flow steps and attractors the "blue lines" show we can always find the likely home for that dataset.
        """
    )
    mo.vstack([desc, ui_panel, fig])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🧪 Why this validates the paper:
    Notice how the blue lines curve and bend *before* they reach the park. Because the unconditioned score function is evaluated based solely on the current position on the manifold, the denoising path is emergent, not prescribed. When this denoising is used, the "true" squirrels are findable, and any noise can be resolved.

    Even if you set the noise multiplier to maximum (scattering squirrels into the East River), the unconditioned gradient flow still maps them back perfectly to the rigid, rectangular shape of Central Park. **The algorithm does not know it is raining noise; it only knows the shape of home.**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    Second Dataset: Owls
    ---
    ## 🦉 Barn Owl Breeding Napa Valley, California Dataset

    This section loads the **Barn Owl** tracking data CSV, cleans missing coordinates, and prepares it for visualization.

    - Filter by individual owls or trace their paths over specific dates.
    - Add artificial noise, and observe the tracing pattern - it always goes back to "home."
    - Watch the diffusion model re-map corrupted coordinates back to their true hunting grounds.
    """)
    return


@app.cell
def _(pl):
    url2 = "./Barn Owl Breeding Napa Valley, California.csv"

    # Load full dataset with all columns
    df_owl_full = (
        pl.read_csv(url2)
        .select(
            [
                "event-id",
                "visible",
                "timestamp",
                "location-long",
                "location-lat",
                "sensor-type",
                "individual-taxon-canonical-name",
                "tag-local-identifier",
                "individual-local-identifier",
                "study-name",
            ]
        )
        .drop_nulls()
    )
    return (df_owl_full,)


@app.cell
def _(df_owl_full, mo):
    _ac1 = mo.md(
        "### To understand the owl data, optionally review the source data table (open accordion)"
    )
    _ac2 = mo.accordion({f"df_rows: {df_owl_full.shape[0]} total": df_owl_full})

    mo.vstack([_ac1, _ac2])
    return


@app.cell
def _(df_owl_full, mo, pl):
    # 1. Dynamically find the min and max dates in the dataset
    time_col = pl.col("timestamp").str.to_datetime()

    min_dt = df_owl_full.select(time_col.min()).item()
    max_dt = df_owl_full.select(time_col.max()).item()

    # Convert Polars datetime to standard Python date objects
    dataset_start = min_dt.date() if min_dt else None
    dataset_end = max_dt.date() if max_dt else None

    # 2. Wire the dates into the UI, setting bounds and default values
    start_date = mo.ui.date(
        label="Start Date",
        value=dataset_start,
        start=dataset_start,
        stop=dataset_end,
    )

    end_date = mo.ui.date(
        label="End Date", value=dataset_end, start=dataset_start, stop=dataset_end
    )

    # 3. Define remaining sliders
    n_lines2 = mo.ui.slider(
        start=10, stop=1000, step=10, value=150, label="Plot Sample Size"
    )

    noise_slider2 = mo.ui.slider(
        start=0.001,
        stop=0.030,
        step=0.001,
        value=0.015,
        label="🛰️ GPS Noise Level (Degrees)",
    )

    steps_slider2 = mo.ui.slider(
        start=10, stop=100, step=10, value=50, label="⏱️ Flow Steps (Iterations)"
    )

    step_size_slider2 = mo.ui.slider(
        start=0.00001,
        stop=0.0005,
        step=0.00001,
        value=0.0001,
        label="🧲 Attractor Strength (Step Size)",
    )

    # 4. Define Owl Dropdown
    owl_options = ["All"]

    f_owl_names = (
        df_owl_full.get_column("individual-local-identifier")
        .drop_nulls()
        .unique()
        .to_list()
    )

    owl_options.extend(f_owl_names)

    id_dropdown = mo.ui.dropdown(
        options=owl_options, value="All", label="Owl Name"
    )
    return (
        end_date,
        id_dropdown,
        n_lines2,
        noise_slider2,
        start_date,
        step_size_slider2,
        steps_slider2,
    )


@app.cell
def _(
    end_date,
    id_dropdown,
    mo,
    n_lines2,
    noise_slider2,
    start_date,
    step_size_slider2,
    steps_slider2,
):
    ui_panel2 = mo.vstack(
        [
            mo.hstack(
                [noise_slider2, steps_slider2, step_size_slider2],
                justify="space-around",
            ),
            mo.hstack(
                [n_lines2, id_dropdown, start_date, end_date],
                justify="space-between",
                gap=3,
            ),
        ]
    )

    # display the panel later for preso view
    # ui_panel2
    return (ui_panel2,)


@app.cell
def _(df_owl_full, end_date, id_dropdown, mo, n_lines2, np, pl, start_date):
    filtered_df = df_owl_full.clone()

    # Safely convert strings to Datetimes so we can accurately compare against Marimo date pickers
    filtered_df = filtered_df.with_columns(
        pl.col("timestamp").str.to_datetime().alias("parsed_dt")
    )

    # 1. Apply Start Date
    if start_date.value is not None:
        filtered_df = filtered_df.filter(
            pl.col("parsed_dt").dt.date() >= start_date.value
        )

    # 2. Apply End Date
    if end_date.value is not None:
        filtered_df = filtered_df.filter(
            pl.col("parsed_dt").dt.date() <= end_date.value
        )

    # 3. Apply ID Dropdown (Skip if "All" is selected)
    if id_dropdown.value is not None and id_dropdown.value != "All":
        filtered_df = filtered_df.filter(
            pl.col("individual-local-identifier") == id_dropdown.value
        )

    # 4. SAFETY CHECK: Halt gracefully if filters are too strict (prevents KDE crash!)
    mo.stop(
        filtered_df.height < 2,
        mo.md(
            "⚠️ **Not enough data points found for the selected dates/IDs. Please adjust your filters.**"
        ),
    )

    # Select coordinates and ID
    filtered_owls = filtered_df.select(
        ["location-lat", "location-long", "individual-local-identifier"]
    )

    # Sample for UI smoothness. Bound the sample max by the total rows available
    n_samples = min(n_lines2.value, filtered_owls.height)
    df_sampledown = filtered_owls.sample(
        n=n_samples, seed=42, with_replacement=False
    )

    # Extract coordinate arrays (latitude = lat, longitude = lon)
    owl_lat = df_sampledown["location-lat"].to_numpy()
    owl_lon = df_sampledown["location-long"].to_numpy()
    owl_name = df_sampledown["individual-local-identifier"].to_list()

    # Stack as [lon, lat] for KDE calculations
    true_coords2 = np.vstack([owl_lon, owl_lat])
    return df_sampledown, filtered_df, owl_lat, owl_lon, owl_name, true_coords2


@app.cell
def _(filtered_df, mo):
    # Debug: rows after filtering
    filtered = mo.md(f"Rows matching criteria: **{filtered_df.height}**")
    return (filtered,)


@app.cell
def _(gaussian_kde, true_coords2):
    # Compute KDE for Owl data
    owl_kde = gaussian_kde(true_coords2, bw_method=0.08)
    return (owl_kde,)


@app.cell
def _(
    get_score_field,
    noise_slider2,
    np,
    owl_kde,
    step_size_slider2,
    steps_slider2,
    true_coords2,
):
    # 5. Execute the Blind Gradient Flow for Owl data
    # Add noise to the true coordinates
    noise2 = np.random.normal(0, noise_slider2.value, true_coords2.shape)
    noisy_coords2 = true_coords2 + noise2

    # Store trajectory for the animation
    trajectories_lon2 = [noisy_coords2[0, :].copy()]
    trajectories_lat2 = [noisy_coords2[1, :].copy()]

    # Initialize current coordinates
    current_coords2 = noisy_coords2.copy()

    # Iterate: Move points along the score field using the owl KDE
    for _ in range(int(steps_slider2.value)):
        _score = get_score_field(current_coords2, owl_kde)

        # Clip the score to prevent extreme jumps when far from the manifold
        _norm = np.linalg.norm(_score, axis=0)
        _max_norm = 500.0
        _mask = _norm > _max_norm
        if _mask.any():
            _score[:, _mask] = _score[:, _mask] * (_max_norm / _norm[_mask])

        current_coords2 += step_size_slider2.value * _score
        trajectories_lon2.append(current_coords2[0, :].copy())
        trajectories_lat2.append(current_coords2[1, :].copy())
    return noisy_coords2, trajectories_lat2, trajectories_lon2


@app.cell
def _(
    df_sampledown,
    filtered,
    go,
    mo,
    noisy_coords2,
    owl_lat,
    owl_lon,
    owl_name,
    trajectories_lat2,
    trajectories_lon2,
    ui_panel2,
):
    # 6. Build the Visual Story with Plotly
    fig2 = go.Figure()

    # Layer 1: The True Manifold (Green)
    fig2.add_trace(
        go.Scattermapbox(
            lat=owl_lat,
            lon=owl_lon,
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=6, color="rgb(46, 139, 87)", opacity=0.4
            ),
            name="True Owl Trajectory",
            hovertext=owl_name,
        )
    )

    # Layer 2: The Noisy 'Lost' Points (Red)
    fig2.add_trace(
        go.Scattermapbox(
            lat=noisy_coords2[1, :],
            lon=noisy_coords2[0, :],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=5, color="rgb(220, 20, 60)", opacity=0.8
            ),
            name="Noisy GPS Data",
        )
    )

    # Layer 3: The Denoising Flow (Lines)
    # Safely loop only over the number of points we *actually* sampled
    actual_lines = noisy_coords2.shape[1]

    for _i in range(actual_lines):
        fig2.add_trace(
            go.Scattermapbox(
                lat=[step[_i] for step in trajectories_lat2],
                lon=[step[_i] for step in trajectories_lon2],
                mode="lines",
                line=dict(width=1, color="rgba(100, 149, 237, 0.5)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add a dummy trace for the legend of the lines
    fig2.add_trace(
        go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode="lines",
            line=dict(width=2, color="rgba(100, 149, 237, 1.0)"),
            name="Gradient Flow Path",
        )
    )

    # Update Mapbox layout using an open-source style
    fig2.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=12,
        mapbox_center={
            "lat": df_sampledown["location-lat"].mean(),
            "lon": df_sampledown["location-long"].mean(),
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )


    desc2 = mo.md(
        # 6.
        """
        ### Plotting real (e.g. "probably noisy") data: Owls 
        Instructions: 1) Select an owl name
        2) adjust GPS Noise/Flow/Attract
        3) See how blue line trajectories always cluster at "home"

    Observe how when adding GPS noise the owl locations are broad; with flow steps and attractors the "blue lines" show we can always find the likely home for that dataset.
        """
    )
    mo.vstack([desc2, ui_panel2, fig2, filtered])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🧪 Why this validates the paper/Applications of the paper:

    Even more than the squirrel data (which is mostly human reported, not noise-polluted GPS data)  the blue lines curve and bend more significantly, especially when comparing individual owls. Because the score function is evaluated continuously based on the topology of the density estimate (the true squirrels), the noise naturally resolves itself. The unconditioned gradient flow still maps them back perfectly to the rigid, rectangular shape of Central Park.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
