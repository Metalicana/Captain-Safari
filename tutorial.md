# Understanding Memory Modules in Frontier Video Generation Papers

## Goal of this tutorial

The purpose of this note is to understand how recent memory-based video generation and video world-model papers implement memory.

Rather than treating "memory" as a vague module, we decompose every method into five design choices:

\[
\boxed{
\text{what is stored}
\rightarrow
\text{how it is addressed}
\rightarrow
\text{how it is retrieved}
\rightarrow
\text{how it conditions generation}
\rightarrow
\text{whether it updates/transitions}
}
\]

This lets us compare papers such as **Captain Safari**, **Context-as-Memory**, **WorldMem**, **VRAG**, **MagicWorld**, and others using the same mathematical language.

Our intended contribution will eventually be framed as:

\[
\boxed{
\textbf{Action-equivariant memory for long-horizon aerial video generation}
}
\]

The key idea is that memory should not only retrieve what the world looked like; it should also transform consistently under actions.

---

# 1. Universal notation

Let a video be:

\[
V_{1:T} = \{I_1, I_2, \ldots, I_T\}
\]

where \(I_t\) is the RGB frame at time \(t\).

Let the camera or agent state be:

\[
x_t = (p_t, R_t, v_t)
\]

where:

\[
p_t \in \mathbb{R}^3
\]

is position,

\[
R_t \in SO(3)
\]

is orientation,

and:

\[
v_t \in \mathbb{R}^3
\]

is velocity.

Let the action sequence be:

\[
A_{t:t+K} = \{a_t, a_{t+1}, \ldots, a_{t+K}\}
\]

For our aerial setting:

\[
a_t = (\alpha_t, \omega_t) \in \mathbb{R}^6
\]

where:

\[
\alpha_t \in \mathbb{R}^3
\]

is body-frame translational acceleration, and:

\[
\omega_t \in \mathbb{R}^3
\]

is body-frame angular velocity.

A generic memory bank is:

\[
\mathcal{M}_t = \{m_i\}_{i=1}^{N_t}
\]

where each memory item may contain visual features, pose, time, action history, or other state information.

A general memory item can be written as:

\[
m_i = (z_i, I_i, x_i, h_i, \tau_i)
\]

where:

- \(z_i\): latent or feature token,
- \(I_i\): RGB frame or context frame,
- \(x_i\): camera/agent state,
- \(h_i\): local history or action context,
- \(\tau_i\): timestamp.

A memory retriever is:

\[
W_t = R_\theta(q_t, \mathcal{M}_t)
\]

where \(q_t\) is a query. The query may be derived from pose, time, current image, text, action, or global state.

A video generator is:

\[
\hat V_{t:t+K}
=
G_\phi(I_t, c, A_{t:t+K}, x_{t:t+K}, W_t)
\]

where \(c\) is optional text conditioning and \(W_t\) is retrieved memory.

Our proposed additional operator is a memory transition function:

\[
\bar W_{t+K}
=
T_\theta(W_t, A_{t:t+K})
\]

This predicts what the current memory should become after applying an action sequence.

The central equation of our proposed direction is:

\[
\boxed{
T_\theta(R_\theta(x_t, \mathcal{M}_t), A_{t:t+K})
\approx
R_\theta(g_A(x_t), \mathcal{M}_t)
}
\]

where:

\[
g_A(x_t) = x_{t+K}
\]

is the action-induced future state. In our aerial case, \(g_A\) can be implemented using RK4 integration.

In words:

> Retrieve memory at the current state, transform it by the action, and require it to match memory retrieved at the action-induced future state.

This is the action-equivariant memory principle.

---

# 2. The five design axes of memory

Every memory-based video generation paper can be described using these five axes.

| Axis | Question |
|---|---|
| Storage | What is stored? RGB frames, latent tokens, 3D features, point clouds, poses, timestamps? |
| Addressing | What queries the memory? Pose, FOV overlap, current frame, text, state, action? |
| Retrieval | Is memory selected by nearest neighbor, top-\(K\), FOV overlap, attention, or learned retrieval? |
| Injection | How does retrieved memory enter the generator? Concatenated frames, cross-attention, KV cache, side branch? |
| Transition/update | Does memory just sit there, or does it evolve under time/actions? |

Most frontier methods focus on:

\[
\boxed{
\text{storage} + \text{addressing} + \text{retrieval} + \text{injection}
}
\]

Our planned contribution focuses on:

\[
\boxed{
\text{action-conditioned memory transition}
}
\]

---

# 3. Captain Safari

## 3.1 What Captain Safari stores

Captain Safari stores **pose-aligned world memory**.

Each frame \(I_i\) has a camera pose:

\[
x_i = (R_i, T_i)
\]

A geometry encoder extracts 3D-aware memory tokens:

\[
z_i = G(I_i)
\]

where:

\[
z_i \in \mathbb{R}^{n_m \times d_m}
\]

Each memory item is:

\[
m_i = (x_i, z_i)
\]

Therefore, the memory bank is:

\[
\mathcal{M}
=
\{(x_i, z_i)\}_{i=1}^{N}
\]

This means every memory row is:

\[
\boxed{
\text{where the camera was} + \text{what the world looked like there}
}
\]

Captain Safari’s memory is fundamentally **pose-indexed**.

---

## 3.2 How Captain Safari retrieves memory

Given a target camera pose:

\[
x_t
\]

Captain Safari builds a query:

\[
q_t = \phi_x(x_t)
\]

It appends learnable query tokens:

\[
\hat Q_t = [q_t, r_1, r_2, \ldots, r_{n_q}]
\]

Then:

\[
Q_t = \operatorname{QryEnc}(\hat Q_t)
\]

The memory rows are encoded as:

\[
\hat X_i =
[
\phi_x(x_i),
\phi_z(z_{i,1}),
\ldots,
\phi_z(z_{i,n_m})
]
\]

and:

\[
\tilde X_i = \operatorname{MemEnc}(\hat X_i)
\]

For a local memory window:

\[
\tilde X^{\mathrm{mem}}
=
[
\tilde X_{k_s},
\ldots,
\tilde X_{k_e}
]
\]

Then memory retrieval is cross-attention:

\[
W_t
=
Q_t
+
\operatorname{CrossAttn}
(
Q_t,
\tilde X^{\mathrm{mem}},
\tilde X^{\mathrm{mem}}
)
\]

Expanding a single attention head:

\[
U = Q_t W_Q
\]

\[
K = \tilde X^{\mathrm{mem}} W_K
\]

\[
V = \tilde X^{\mathrm{mem}} W_V
\]

\[
A_{\mathrm{attn}}
=
\operatorname{softmax}
\left(
\frac{UK^\top}{\sqrt{d_h}}
\right)
\]

\[
W_t = Q_t + A_{\mathrm{attn}}V
\]

Thus, in universal notation:

\[
\boxed{
W_t = R_\theta(x_t, \mathcal{M})
}
\]

Captain Safari retrieves memory using the target pose.

---

## 3.3 How retrieved memory enters the generator

The retrieved world tokens \(W_t\) are projected into the video diffusion transformer space:

\[
\bar W_t = \phi_W(W_t)
\]

Let \(H^\ell\) be video latent tokens at transformer layer \(\ell\). Memory enters by cross-attention:

\[
H^{\ell+1}
=
H^\ell
+
\operatorname{CrossAttn}
(
H^\ell,
\bar W_t,
\bar W_t
)
\]

So the video token asks:

\[
\boxed{
\text{What should the world look like from this pose?}
}
\]

The generator can be written as:

\[
\hat V_{1:T}
=
G_\phi(I_0, c, x_{1:T}, W_{1:T})
\]

where:

\[
W_t = R_\theta(x_t, \mathcal{M})
\]

---

## 3.4 Intuition

Captain Safari converts camera pose into a memory address.

A normal temporal model asks:

\[
\text{What happened recently?}
\]

Captain Safari asks:

\[
\text{What stored observations are relevant to this camera pose?}
\]

This is important for FPV video because the camera may leave a region and later revisit it. The relevant memory may not be the most recent frame. It may be an old frame from a physically overlapping viewpoint.

A naive method would retrieve the nearest pose:

\[
i^\star = \arg\min_i d(x_t, x_i)
\]

\[
W_t = z_{i^\star}
\]

Captain Safari instead uses soft cross-attention:

\[
W_t = \sum_i \alpha_i v_i
\]

where:

\[
\alpha_i =
\frac{
\exp(q_t^\top k_i / \sqrt{d})
}{
\sum_j \exp(q_t^\top k_j / \sqrt{d})
}
\]

Thus memory is a learned mixture of multiple pose-aligned observations.

---

## 3.5 What Captain Safari does not explicitly model

Captain Safari gives us:

\[
W_t = R_\theta(x_t, \mathcal{M})
\]

It does not explicitly learn:

\[
T_\theta(W_t, A_{t:t+K})
\approx
R_\theta(x_{t+K}, \mathcal{M})
\]

Its core question is:

\[
\boxed{
\text{Given this pose, what should the world look like?}
}
\]

Our intended question is:

\[
\boxed{
\text{Given this memory and this action, how should memory evolve?}
}
\]

So Captain Safari is:

\[
\boxed{
\text{pose-addressed memory}
}
\]

whereas our proposed method is:

\[
\boxed{
\text{action-equivariant memory}
}
\]

---

# 4. Context-as-Memory

## 4.1 What Context-as-Memory stores

Context-as-Memory stores historical frames directly:

\[
\mathcal{M}_t = \{I_i\}_{i<t}
\]

The memory is not an explicit 3D map. It is not a learned memory bank of geometry tokens. It is simply a set of prior context frames.

Each memory item is approximately:

\[
m_i = (I_i, x_i)
\]

where \(x_i\) is the camera pose used for retrieval.

---

## 4.2 How it retrieves memory

Context-as-Memory retrieves historical frames by field-of-view overlap.

For a target time \(t\), define:

\[
\mathcal{C}_t
=
\operatorname{TopK}_{i<t}
\operatorname{FOVOverlap}(x_t, x_i)
\]

Then:

\[
W_t = \{I_i : i \in \mathcal{C}_t\}
\]

So the retriever is:

\[
\boxed{
W_t = R_{\mathrm{FOV}}(x_t, \mathcal{M}_t)
}
\]

where retrieval is mostly geometric and hard top-\(K\), not learned soft attention.

---

## 4.3 How it conditions generation

The retrieved frames are concatenated with the target generation frames.

A simplified expression:

\[
\hat V_{t:t+K}
=
G_\phi([I_{\mathcal{C}_t}, I_t, \text{target slots}])
\]

So the memory is injected as **extra frames**, not as abstract world tokens.

---

## 4.4 Intuition

Context-as-Memory asks:

\[
\boxed{
\text{Which past frames show the same part of the world?}
}
\]

Then it gives those frames to the generator.

The method is simple and effective because many video generators can use image/video context if the right frames are provided.

It is a passive memory method:

\[
\boxed{
\text{retrieve relevant context, then generate}
}
\]

---

## 4.5 What Context-as-Memory does not explicitly model

It does not learn a memory transition:

\[
T_\theta(W_t, A_{t:t+K})
\]

It retrieves memory at the query pose:

\[
W_t = R_{\mathrm{FOV}}(x_t, \mathcal{M})
\]

but does not enforce:

\[
T_\theta(W_t, A)
\approx
R_{\mathrm{FOV}}(x_{t+K}, \mathcal{M})
\]

Therefore, our modification could be:

\[
\bar W_{t+K}
=
T_\theta(W_t, A_{t:t+K})
\]

\[
\mathcal{L}_{\mathrm{trans}}
=
\left\|
\bar W_{t+K}
-
R_{\mathrm{FOV}}(x_{t+K}, \mathcal{M})
\right\|_2^2
\]

This would convert passive context memory into action-equivariant memory.

---

# 5. WorldMem

## 5.1 What WorldMem stores

WorldMem stores memory units containing frames and states.

A generic memory item is:

\[
m_i = (I_i, x_i, \tau_i)
\]

where:

- \(I_i\): memory frame,
- \(x_i\): state, such as pose,
- \(\tau_i\): timestamp.

So:

\[
\mathcal{M}_t
=
\{(I_i, x_i, \tau_i)\}_{i<t}
\]

This is already more structured than pure frame memory.

---

## 5.2 How it retrieves memory

A generic WorldMem-style retrieval can be written as:

\[
q_t = \phi_q(x_t, \tau_t)
\]

\[
k_i = \phi_k(x_i, \tau_i)
\]

\[
v_i = \phi_v(I_i)
\]

\[
\alpha_i =
\operatorname{softmax}
\left(
\frac{q_t^\top k_i}{\sqrt{d}}
\right)
\]

\[
W_t =
\sum_i \alpha_i v_i
\]

So:

\[
\boxed{
W_t = R_\theta(x_t, \tau_t, \mathcal{M}_t)
}
\]

WorldMem uses both state and time.

---

## 5.3 Intuition

WorldMem is closer to a true world-model memory bank.

It stores:

\[
\boxed{
\text{what was seen} + \text{where/when it was seen}
}
\]

Then it retrieves relevant memory for long-term consistent simulation.

The timestamp allows memory to handle changing worlds, not only static scenes.

---

## 5.4 Why WorldMem is dangerous for our novelty

WorldMem already combines:

\[
\text{memory} + \text{state} + \text{long-term world simulation}
\]

and it is closer to action/world-model settings than simple context retrieval.

Therefore, our novelty cannot be:

\[
\text{memory bank for video generation}
\]

or:

\[
\text{memory + action conditioning}
\]

The novelty must be:

\[
\boxed{
\text{explicit action-equivariant transition in memory space}
}
\]

Our contrast:

\[
\text{WorldMem: } W_t = R_\theta(x_t, \tau_t, \mathcal{M})
\]

\[
\text{Ours: }
T_\theta(W_t, A)
\approx
R_\theta(g_A(x_t), \mathcal{M})
\]

---

# 6. VRAG / Retrieval-Augmented Interactive Video World Models

## 6.1 What VRAG stores

VRAG-style methods use retrieval-augmented generation for interactive video/world modeling.

A generic memory database can be represented as:

\[
\mathcal{M}
=
\{(V_i, s_i, A_i)\}_{i=1}^{N}
\]

where:

- \(V_i\): prior video segment,
- \(s_i\): global or latent state,
- \(A_i\): associated actions.

---

## 6.2 How it retrieves memory

Given current context, state, and possibly actions:

\[
q_t = \phi_q(I_t, s_t, A_{t:t+K})
\]

retrieve:

\[
\mathcal{C}_t =
\operatorname{Retrieve}(q_t, \mathcal{M})
\]

then fuse:

\[
W_t = \operatorname{Fuse}(\mathcal{C}_t, s_t)
\]

Generation:

\[
\hat V_{t:t+K}
=
G_\phi(I_t, A_{t:t+K}, s_t, W_t)
\]

---

## 6.3 Intuition

VRAG-style methods address long-term compounding errors.

Autoregressive video generation often drifts because each generated chunk becomes the input for the next chunk. Retrieval helps by bringing in external or historical context.

The core idea is:

\[
\boxed{
\text{retrieve useful prior examples/context to stabilize interactive video generation}
}
\]

---

## 6.4 Difference from our idea

VRAG retrieves memory and conditions generation on actions/state.

But our proposed method explicitly requires:

\[
T_\theta(W_t, A)
\approx
W_{t+K}^{\mathrm{ret}}
\]

So the contrast is:

\[
\text{VRAG: } \hat V = G(I, A, s, W)
\]

\[
\text{Ours: } W \text{ must satisfy } T(W,A)\approx R(g_A(x),\mathcal{M})
\]

This means memory itself is constrained to obey action-induced transitions.

---

# 7. MagicWorld

## 7.1 What MagicWorld stores

MagicWorld uses historical retrieval plus 3D geometric priors.

A generic memory can be written as:

\[
\mathcal{M}_t = \{(I_i, z_i, x_i)\}_{i<t}
\]

where:

- \(I_i\): historical frame,
- \(z_i\): feature or latent representation,
- \(x_i\): state/pose.

---

## 7.2 Action-guided geometry

MagicWorld uses user actions to construct or predict a 3D geometric prior.

Generic form:

\[
P_t = \operatorname{AG3D}(I_t, A_t)
\]

where \(P_t\) is an action-guided point cloud or geometry representation.

Generation:

\[
\hat I_{t+1}
=
G_\phi(I_t, A_t, P_t, W_t)
\]

where \(W_t\) is retrieved historical context.

---

## 7.3 Intuition

MagicWorld says:

\[
\boxed{
\text{actions should move the scene through geometry, and memory should reduce drift}
}
\]

It combines:

- action control,
- 3D geometry,
- historical retrieval,
- interactive generation.

This makes it close to our conceptual space.

---

## 7.4 Difference from our idea

MagicWorld uses actions to construct geometry and uses retrieval to maintain consistency.

Our method asks for a stricter memory-space relation:

\[
T_\theta(R_\theta(x_t,\mathcal{M}), A)
\approx
R_\theta(g_A(x_t),\mathcal{M})
\]

So:

\[
\text{MagicWorld: } A \rightarrow \text{geometry prior}
\]

\[
\text{Ours: } A \rightarrow \text{memory transition}
\]

This is the distinction.

---

# 8. Video World Models with Long-Term Spatial Memory

## 8.1 What it stores

This line of work introduces long-term spatial memory for video world models.

The memory may be decomposed into:

\[
\mathcal{M}
=
(\mathcal{M}^{spatial}, \mathcal{M}^{working}, \mathcal{M}^{episodic})
\]

where:

- \(\mathcal{M}^{spatial}\): geometry-grounded spatial memory,
- \(\mathcal{M}^{working}\): short-term context,
- \(\mathcal{M}^{episodic}\): stored past observations or events.

---

## 8.2 How it retrieves

A generic retrieval is:

\[
W_t
=
R_\theta(x_t, \mathcal{M}^{spatial}, \mathcal{M}^{working}, \mathcal{M}^{episodic})
\]

The query is usually spatial or pose-based.

---

## 8.3 Intuition

This direction says:

\[
\boxed{
\text{long video generation needs explicit spatial memory}
}
\]

The model should know what parts of the scene have already been observed and maintain them over time.

---

## 8.4 Difference from our idea

Spatial-memory work focuses on:

\[
\boxed{
\text{where things are}
}
\]

Our proposed work focuses on:

\[
\boxed{
\text{how memory should change under actions}
}
\]

Contrast:

\[
\text{Spatial Memory: } W_t = R_\theta(x_t,\mathcal{M}^{3D})
\]

\[
\text{Ours: } T_\theta(W_t,A)\approx R_\theta(g_A(x_t),\mathcal{M})
\]

---

# 9. Decoupled Memory Control

## 9.1 What it stores

Decoupled Memory Control separates memory conditioning from the main frozen video backbone.

It stores hybrid temporal/spatial memory:

\[
\mathcal{M}_t =
\{m_i^{temp}, m_i^{spatial}\}_{i<t}
\]

The goal is to improve long-horizon consistency without heavily modifying the base generator.

---

## 9.2 How it retrieves and injects memory

A generic form:

\[
W_t = R_\theta(x_t, \mathcal{M}_t)
\]

Then the model uses a separate memory branch:

\[
H^{\ell+1}
=
H^\ell
+
\gamma_t
\operatorname{CrossAttn}(H^\ell,W_t,W_t)
\]

where:

\[
\gamma_t=\operatorname{Gate}(x_t,\mathcal{M}_t)
\]

is a camera-aware or relevance-aware gate.

---

## 9.3 Intuition

This method says:

\[
\boxed{
\text{memory should condition the generator only when relevant}
}
\]

This avoids forcing memory into novel views where the past does not help.

---

## 9.4 Difference from our idea

Decoupled Memory Control focuses on memory relevance and modularity.

Our method focuses on action transition:

\[
T_\theta(W_t,A)\approx W_{t+K}
\]

Contrast:

\[
\text{Decoupled Memory: } \gamma(x,\mathcal{M})R(x,\mathcal{M})
\]

\[
\text{Ours: } T(R(x,\mathcal{M}),A)\approx R(g_A(x),\mathcal{M})
\]

---

# 10. RELIC

I do not want to fabricate details here.

To fit RELIC into this tutorial, we need the paper PDF, abstract, or method section.

Once available, we should summarize it using the same template:

1. What is stored?
2. How is memory addressed?
3. How is memory retrieved?
4. How does memory condition generation?
5. Does memory update or transition under actions?

The universal form will be:

\[
W_t = R_\theta(q_t,\mathcal{M})
\]

and we will check whether RELIC has anything equivalent to:

\[
T_\theta(W_t,A)\approx R_\theta(g_A(x_t),\mathcal{M})
\]

If it does, then it is a direct competitor. If it does not, it becomes another passive or state-indexed memory baseline.

---

# 11. Summary table

| Paper | Stored memory | Address/query | Retrieval | Injection | Transition? |
|---|---|---|---|---|---|
| Captain Safari | Pose-tagged 3D/world tokens | Camera pose \(x_t\) | Learned cross-attention | DiT cross-attention | No explicit action transition |
| Context-as-Memory | Historical RGB frames | FOV overlap from pose | Hard top-\(K\) context frames | Frame concatenation | No |
| WorldMem | Frames + states + timestamps | State/time query | Memory attention | Generator conditioning | Time/state-aware, but not necessarily explicit \(T(W,A)\) |
| VRAG | Retrieved video/context + global state | Current state/context/action | Retrieval augmented generation | Fused with generator state | Not primarily memory-transition law |
| MagicWorld | History cache + geometry prior | Action/current scene/history | History retrieval | Geometry + history conditioning | Action affects geometry, not necessarily memory equivariance |
| Long-Term Spatial Memory | Spatial/working/episodic memory | Pose/spatial query | Spatial retrieval | World-model conditioning | Spatial update, not necessarily action-equivariant |
| Decoupled Memory Control | Hybrid temporal/spatial memory | Camera-aware relevance | Per-frame cross-attention + gate | Independent memory branch | Gated memory, not action transition |
| Our planned method | State/action-tagged visual memory | Pose/state + action | Retrieval + action transition | Pose/action/memory conditioning | Yes: \(T(R(x),A)\approx R(g_A(x))\) |

---

# 12. Our intended contribution

Most memory papers do:

\[
W_t = R_\theta(x_t,\mathcal{M}_t)
\]

then:

\[
\hat V = G_\phi(I_t,W_t)
\]

Action-conditioned papers do:

\[
\hat V = G_\phi(I_t,A_{t:t+K})
\]

Memory + action papers often do:

\[
\hat V = G_\phi(I_t,W_t,A_{t:t+K})
\]

Our method should do:

\[
W_t = R_\theta(x_t,\mathcal{M}_t)
\]

\[
\bar W_{t+K} = T_\theta(W_t,A_{t:t+K})
\]

\[
\bar W_{t+K} \approx R_\theta(g_A(x_t),\mathcal{M}_t)
\]

\[
\hat V =
G_\phi(I_t,A_{t:t+K},W_t,\bar W_{t+K})
\]

The main thesis is:

\[
\boxed{
\text{Existing memory retrieves the past; our memory learns how the retrieved past should transform under actions.}
}
\]

Or more mathematically:

\[
\boxed{
T_\theta \circ R_\theta
\approx
R_\theta \circ g_A
}
\]

This is an action-equivariance constraint on memory.

---

# 13. How this connects to our aerial setting

For aerial videos:

\[
x_t=(p_t,R_t,v_t)
\]

and:

\[
A_t=(\alpha_t,\omega_t)
\]

where \(\alpha_t\) is acceleration and \(\omega_t\) is angular velocity.

The action-induced future state is:

\[
g_A(x_t)
=
\operatorname{RK4}(x_t,A_{t:t+K})
\]

So our key constraint becomes:

\[
T_\theta(R_\theta(x_t,\mathcal{M}),A_{t:t+K})
\approx
R_\theta(\operatorname{RK4}(x_t,A_{t:t+K}),\mathcal{M})
\]

This is directly testable using pose, pseudo-IMU, real IMU, and memory pairs.

---

# 14. Why this matters

A passive memory module can preserve visual appearance, but it does not necessarily know how the remembered world should evolve under actions.

An action-conditioned generator can follow local commands, but it may drift or forget the scene over long horizons.

Action-equivariant memory combines both:

\[
\boxed{
\text{scene persistence}
+
\text{action consistency}
}
\]

The memory becomes a controllable state representation, not just a cache of old frames.

This is the conceptual gap we want to explore.