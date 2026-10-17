# Master's Application Guide: PhysioEarth-Wildfire
### How to Leverage This Project on Your CV, Statement of Purpose (SOP), and Admissions Interviews

---

## 1. Exact CV / Resume Bullet Points

Place this under **Projects** or **Research Experience** on your academic CV:

### Format A (For MS in Data Science / Artificial Intelligence):
> **PhysioEarth: Physics-Constrained Fourier Neural Operator for Wildfire Spread** | *Python, PyTorch, SciML, PDEs*
> * Formulated a 2D Fourier Neural Operator (FNO) integrating Rothermel's surface fire spread physics and reaction-diffusion combustion dynamics over complex digital elevation models.
> * Engineered a hybrid physics-informed loss function that mathematically penalizes unphysical fuel regeneration ($\partial Y / \partial t \le 0$) and natural barrier penetration.
> * Demonstrated a **39% reduction in temperature field RMSE** (0.3103 vs. 0.5110) and eliminated river barrier breaches (**0.98% vs. 16.55%**) compared to standard unconstrained 2D U-Nets across multi-step autoregressive rollouts.
> * Developed an open-source framework with 100% test coverage (`pytest`) and an interactive 60 FPS disaster digital twin web dashboard. [Code & Demo: GitHub Link]

### Format B (For MS in Computational Science / Applied Math):
> **Spectral Neural Operators for Non-Linear Reaction-Diffusion Combustion Dynamics** | *Partial Differential Equations, SciML*
> * Parameterized infinite-dimensional function mappings using spectral Fourier convolutions, demonstrating resolution-invariant zero-shot generalization across terrain resolutions.
> * Implemented finite-difference solvers for coupled parabolic-hyperbolic equations incorporating anisotropic slope convection and atmospheric advection vectors.
> * Quantified physical violation rates (PVR) across multi-step rollouts, establishing stability bounds under extreme environmental domain shifts.

---

## 2. Statement of Purpose (SOP) Formulation

Incorporate this project into your Statement of Purpose (SOP) to demonstrate high research potential:

### SOP Paragraph (For Top Universities Worldwide):
> *"My research interests lie at the intersection of scientific machine learning (SciML), partial differential equations, and climate disaster resilience (UN SDG #13). Recognizing that conventional deep learning architectures (such as U-Nets or ConvLSTMs) produce physically impossible hallucinations—such as unphysical fuel regeneration and fire spreading across water bodies—I developed **PhysioEarth-Wildfire**.*
> 
> *By combining a 2D Fourier Neural Operator (FNO) with physics-informed conservation constraints, I modeled wildfire propagation as a coupled reaction-diffusion combustion system over digital elevation topographies. My empirical benchmarks demonstrated that enforcing physical monotonic fuel depletion reduced multi-step thermal RMSE by 39% and cut river firebreak breaches from 16.55% down to 0.98% compared to standard spatial vision baselines.*
> 
> *This experience solidified my aspiration to pursue graduate studies in Data Science and AI at **[University Name]**. I am eager to join the research group of **Prof. [Professor's Name]** to advance neural operators for planetary-scale climate forecasting and physics-informed foundation models."*

---

## 3. Admissions / Professor Technical Interview Questions & Answers

### Q1: "Why use a Fourier Neural Operator (FNO) instead of a standard CNN or U-Net?"
* **Your Winning Answer:**
  > *"Standard CNNs and U-Nets learn mappings between finite-dimensional Euclidean grids. If a CNN is trained on a 500-meter satellite grid, it breaks down when evaluated on 30-meter elevation terrain because its learned convolutional kernels are tied to fixed pixel distances. The Fourier Neural Operator parameterizes integral kernel operators directly in the frequency domain via spectral Fourier convolutions ($R \cdot \mathcal{F}(v)$). This makes FNO **mesh-independent and resolution-invariant**: we can train on coarse simulation meshes and evaluate zero-shot on fine satellite rasters without retraining."*

### Q2: "How do you mathematically prevent the model from predicting fire where there is no fuel?"
* **Your Winning Answer:**
  > *"We incorporate a hard physical monotonicity constraint into the loss function: $\mathcal{L}_{\text{fuel}} = \|\text{ReLU}(Y_{t+1} - Y_t)\|_2^2$. In nature, vegetation combustion is strictly irreversible ($\partial Y / \partial t \le 0$). Any positive $\Delta Y > 0$ represents unphysical biomass regeneration and incurs a quadratic loss penalty. We also add a firebreak barrier loss $\mathcal{L}_{\text{barrier}} = \|T \odot M_{\text{barrier}}\|_2^2$ to penalize active thermal energy on water bodies."*

### Q3: "How does your model incorporate Rothermel's surface fire equations?"
* **Your Winning Answer:**
  > *"Rothermel's formulation models rate-of-spread as an isotropic base rate multiplied by slope ($\Phi_S$) and wind ($\Phi_W$) factors. We compute the spatial gradient of digital elevation models $\nabla h = (\partial h/\partial x, \partial h/\partial y)$ and combine it with the ambient wind vector $\mathbf{w}$ to construct an anisotropic advection velocity field $\mathbf{v}_{\text{eff}} = c_{\text{wind}}\mathbf{w} + c_{\text{slope}} \nabla h \|\nabla h\|$. This vector field is fed as an input channel to the FNO, enabling the network to learn directional convection."*
