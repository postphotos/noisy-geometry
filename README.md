# 🦉 The Geometry of Noise: Translating Theory into Geography

**An interactive exploration of diffusion models, noise, and ecology.**  
Submitted to the [alphaXiv × marimo Notebook Competition](https://marimo.io/pages/events/notebook-competition).

**Author:** Leo Postovoit | [@postphotos](https://github.com/postphotos) | [leopostovoit.com](https://leopostovoit.com/)  
I'm a strategic product leader and technologist specializing in developing intuitive tools that bridge the gap between human needs, historical data, and ecological systems. My foundation in journalism and anthropology deeply informs my product philosophy. I've led strategic initiatives for enterprise clients including Google and Automattic, and built specialized solutions alongside legendary publications like the Wall Street Journal, San Francisco Chronicle, and Lowrider Magazine.

---

## 🧭 Project Process
I built this project to explore how theoretical AI research can be translated into practical, real-world tools. After reading through the paper *The Geometry of Noise*, I took its core mathematical concept—that diffusion models don't actually need to know the magnitude of noise to reverse it—and built interactive examples to test it. 

### About my motivation to apply the paper to real data

Historically, if a dataset was heavily corrupted by a storm or a bad sensor, ecologists had to throw it out or spend months manually correcting it. This new mathematical approach proves that "junky" or noisy data is completely fine to use. The algorithm simply looks at the overall "shape" (geometry) of all the scattered data points and naturally triangulates the true center. This suggests a future where real-time tracking during environmental disasters, guided by strong data practices, can reduce the reliance on manual data cleaning to determine the central tendency and true spatial distribution of the dataset.

This means we can spend less time agonizing over:
- a scrambled lat-long ping from a damaged GPS collar, or:
- a typo in a crowdsourced field form, 

and more time focusing on what actually matters: 
- applying these insights to direct emergency interventions, 
- protecting vulnerable habitats, 
- and translating raw information into real-world impact. 

That said, this isn't an excuse to intentionally collect bad data. It's simply an acknowledgment that imperfect data still has a story to tell, and with the right mathematical lens, that story will reveal itself. Even somewhat junky data—when collected at scale—holds vital insights. You just have to find the right way to squeeze the juice (and make sense of it all!)

### Here is how I broke it down:
1. Playful exploration: I first applied the math to a theoretical concept (a barista "latte art" game) to visually explain the paper's thesis and show why unknown noise isn't a big deal.
2. Grounded data: I then applied the concept to a small, controlled dataset (NYC squirrels) to demonstrate exactly how the gradient math naturally clusters scattered points back together.
3. Real application: Finally, I applied it to a larger, inherently noisy ecological dataset (GPS tracking of Napa Valley owls) to show the actual relevance of this math for fields like wildlife conservation.

---

## 🌍 The "Why": Ecology, Megafires, and the GPS Noise Problem

This repository uses real-world ecological data to prove a complex mathematical theorem from the AI diffusion space. To understand why this matters, look at the intersection of wildlife conservation and GPS technology during a crisis.

### 1. The Ecological Stakes
The 2021 KNP Complex Fire in the Sequoia & Kings Canyon National Parks was catastrophic. Reporting from the [*Los Angeles Times*](https://www.latimes.com/california/story/2021-11-26/how-the-knp-complex-fire-devastated-one-giant-sequoia-grove) and [*San Francisco Chronicle*](https://www.sfchronicle.com/california-wildfires/article/Southern-Sierra-wildfires-wiping-out-giant-16496614.php) detailed how the blaze, alongside the Windy Fire, decimated an estimated 3% to 5% of the world's remaining mature Giant Sequoias. Compounded with previous megafires, nearly 20% of the entire global population was lost in a matter of months.

For the state-endangered Great Gray Owl, which requires these exact mature forests to survive, the fire was an existential threat. Most ecological fire studies are *post-hoc* (evaluating the damage months after the smoke clears). However, the [Movebank Dataset: Great Gray Owl Responses to the KNP Complex Fire](https://datarepository.movebank.org/items/d38655cd-dab5-4b4d-842c-5c0f8c41fb73) is a rare "eye of the storm" dataset. It captures the active survival decisions of the owls mid-blaze.

When a habitat literally goes up in smoke overnight, conservationists need to know *immediately* where surviving populations are fleeing so they can protect remaining unburned patches. If wildlife GPS data is delayed because it has to be manually scrubbed for errors, it becomes useless for real-time triage. Reliable, fast data processing is a matter of life and death for these species.

### 2. The Dirty Data (Why GPS Noise is an Issue)

Wildlife GPS tracking is notoriously noisy. During a megafire, two factors make it incredibly challenging:
*   **Canopy Bias:** Heavy redwood forest canopies can bias GPS track lengths by up to 27.5%, scrambling the signal as it bounces off massive branches.
*   **Atmospheric Interference:** Heat-distorted air, massive smoke plumes, and rapidly changing canopy structures (as the trees burn) suddenly change the "noise profile" mid-tracking. 

### 3. The Mathematical Solution: Why Noise Isn't a Problem
This is where the core paper of this project comes in: [**alphaXiv:2602.18428 — The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning**](https://www.alphaxiv.org/abs/2602.18428). 

Traditionally, algorithms needed to know *exactly how much noise* was added to a dataset to clean it (noise conditioning). This paper mathematically proves that modern autonomous models can learn the **intrinsic geometry** of the data instead. By treating noise as a topological surface to be navigated rather than a barrier, the geometric gradient naturally pulls scattered, noisy tracking points back to their true geographical centers.

### 4. Expanding the Horizon (Why "Junky" Data is Now OK!)
**TL;DR: You don't need pristine, heavily scrubbed data to find the truth.** 

Historically, if a dataset was heavily corrupted by a storm or a bad sensor, ecologists had to throw it out or spend months manually correcting it. This new mathematical approach proves that "junky" or noisy data is completely fine to use. The algorithm simply looks at the overall "shape" (geometry) of all the scattered data points and naturally triangulates the true center. This suggests a future where real-time tracking during environmental disasters doesn't require human-in-the-loop cleaning to be accurate; LLM.

---

## 📚 The Notebooks

### 1. ☕ Cafe Diffusion: The Blind Barista Challenge  
📁 `barista/notebook.py`  
A playful, UI-driven explanation of the core theorem. A rogue espresso machine has scrambled your latte art... How close can you get to the self-healing denoising?!
*   **The Old Method:** Guess the noise level (σ) using a radio dial. Guess wrong, and the math hits an energy singularity, exploding into static.
*   **The New Method:** Watch the geometric gradient field automatically recover the art without ever knowing the noise level.

### 2. 🐿️ The Geometry of Noise: Habitat Recovery  
📁 `owl-squirrel/notebook.py`  
A geographic proof-of-concept translating N-dimensional calculus into 2D map recovery.
*   **Experiment 1 (NYC Squirrels):** We inject extreme GPS noise into the NYC Squirrel Census. Using purely blind gradient flow, watch the geometry pull the center of the scattered squirrels back to the rectangular boundary of Central Park.
*   **Experiment 2 (Napa Valley Owls):** We repeat the process using GPS-tracked barn owl breeding locations. The model ignores the noise magnitude entirely and traces the corrupted data directly back to the owls' true hunting grounds.

---

## 🚀 Running the Project 

### On Molab (Cloud): 
*[☕ Cafe Diffusion: The Blind Barista Challenge!](https://molab.marimo.io/notebooks/nb_A4EMTSqdRtQ6yyv2VXwUQr)
*[🐿️ The Geometry of Noise: Habitat Recovery 🦉](https://molab.marimo.io/notebooks/nb_SxRn5BEnv2BZ5SgYCnTVgr)

### Locally:

**Prerequisites:**
* Python 3.13+
* `git`
* [`uv`](https://docs.astral.sh/uv/) - Required for managing dependencies and the environment

Clone the repo: 
```
git https://github.com/postphotos/noisy-geometry
cd noisy-geometry
``` 

Use `uv` to automatically sync the project's dependencies and boot the interactive editor:

```bash
# 1. Explore the Barista Theory
cd barista
uv sync
uv run marimo edit notebook.py

# 2. Explore the Geographic Proofs
cd ../owl-squirrel
uv sync
uv run marimo edit notebook.py
```

---

## 📄 License

This project is licensed under the GNU General Public License v3.0.
See the `LICENSE` file for full terms.
