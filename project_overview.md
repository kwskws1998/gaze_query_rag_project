# Reader-Specific Gaze-Query Embeddings for No-Training Personalized RAG

## 1. Introduction

This project asks whether a reader's observed gaze pattern can be used as a
personalized retrieval signal for reading-comprehension QA. The immediate goal
is not to predict private emotion, not to fine-tune an LLM, and not to add a
large model stack. The core hypothesis is:

$$
g_{r,t}\neq g_{r',t}
\Rightarrow
z^g_{r,t,k}\neq z^g_{r',t,k}
\Rightarrow
\mathcal{E}^g_{r,t}\neq \mathcal{E}^g_{r',t}.
$$

That is, two readers can read the same paragraph \(t\), but their different
gaze distributions should produce different personalized evidence chunks for
the same question.

The downstream task is Hugging Face OneStop QA:

$$
\mathcal{D}_{QA}=\texttt{malmaud/onestop\_qa}.
$$

The gaze source is OneStop Eye Movements paragraph-level interest-area data:

$$
\mathcal{D}_{gaze}=\texttt{ia\_Paragraph.csv.zip}.
$$

The concrete frozen models are:

$$
E=\texttt{intfloat/e5-large-v2},
\qquad
G=\texttt{meta-llama/Meta-Llama-3-8B-Instruct}.
$$

No parameter is trained in the first experiment:

$$
\theta_E,\theta_G\ \text{frozen},
\qquad
\Theta_{\mathrm{trainable}}=\varnothing.
$$

The first experiment is therefore a post-reading no-training RAG test:

$$
\mathrm{TRT}_{r,t}
\rightarrow
g_{r,t}
\rightarrow
z^g_{r,t,k}
\rightarrow
\mathcal{E}^g_{r,t}
\rightarrow
\hat y^g_{r,t}.
$$

This is not yet future prediction. The future-prediction version replaces the
observed test gaze \(g_{r,t}\) with a calibration-predicted gaze distribution
\(\hat g_{r,t}\).

## 2. Background

The architecture is closest to heatmap-conditioned representation learning. In
eCLIP, the paper describes the query/key/value role split as:

> "patchified heatmap overlaid images serve as queries"

and:

> "original image's patches act as keys and values"

Our version keeps this role split but changes the modality:

$$
\text{heatmap-overlaid image queries}
\Rightarrow
\text{gaze-weighted text-token queries},
$$

$$
\text{original image patches as keys/values}
\Rightarrow
\text{original text-token embeddings as keys/values}.
$$

Gaze-conditioned language-modeling work also motivates the use of eye-tracking
signals inside text representations. GazeReward states that it:

> "combine the ET features with the text"

and uses a projection step:

> "project these features to the model embedding size"

This project is simpler: no reward model, no LLM fine-tuning, no learned gaze
adapter in the first pass. The reader-specific signal enters only through a
gaze distribution over tokens.

## 3. Method

For paragraph \(t\), let the word sequence be:

$$
x_t=(w_{t,1},\ldots,w_{t,M}).
$$

The E5 tokenizer maps word \(w_{t,m}\) to subtoken index set \(S_{t,m}\). To
preserve word-level mass after tokenization:

$$
\mathrm{TRT}_{r,t,i}
=
\frac{\mathrm{TRT}_{r,t,m}}{|S_{t,m}|},
\qquad
i\in S_{t,m}.
$$

The frozen E5 encoder gives paragraph token embeddings:

$$
H_t
=
E_{\theta_E}(\texttt{passage: }x_t)
\in\mathbb{R}^{L\times d},
\qquad
H_t=[h_{t,1},\ldots,h_{t,L}]^\top.
$$

Raw gaze is log-transformed and normalized:

$$
a_{r,t,i}=\log(1+\mathrm{TRT}_{r,t,i}),
\qquad
g_{r,t,i}
=
\frac{a_{r,t,i}+\epsilon}
{\sum_{j=1}^{L}(a_{r,t,j}+\epsilon)}.
$$

Thus:

$$
g_{r,t}\in\Delta^{L-1},
\qquad
D_{r,t}=\operatorname{diag}(g_{r,t}).
$$

The gaze-weighted token embedding is just row scaling:

$$
H^g_{r,t}=D_{r,t}H_t,
\qquad
h^g_{r,t,i}=g_{r,t,i}h_{t,i}.
$$

No learned \(W_Q,W_K,W_V\) are introduced. The gaze-weighted tokens are
queries, and the original text tokens are keys and values:

$$
Q_{r,t}=D_{r,t}H_t,
\qquad
K_t=H_t,
\qquad
V_t=H_t.
$$

The gaze-query attention is:

$$
A_{r,t}
=
\operatorname{softmax}
\left(
\frac{(D_{r,t}H_t)H_t^\top}{\sqrt d}
\right).
$$

The reader-specific gaze-conditioned token representation is:

$$
\widetilde{H}^{g}_{r,t}
=
A_{r,t}H_t
=
\operatorname{softmax}
\left(
\frac{(D_{r,t}H_t)H_t^\top}{\sqrt d}
\right)
H_t.
$$

For chunk \(k\) with token index set \(I_{t,k}\), the reader-specific
gaze-view chunk embedding is:

$$
z^g_{r,t,k}
=
\operatorname{norm}
\left(
\frac{1}{|I_{t,k}|}
\sum_{i\in I_{t,k}}
\widetilde h^g_{r,t,i}
\right).
$$

The text-only baseline uses the original token embeddings:

$$
z_{t,k}
=
\operatorname{norm}
\left(
\frac{1}{|I_{t,k}|}
\sum_{i\in I_{t,k}}
h_{t,i}
\right).
$$

The full proposed embedding is:

$$
\boxed{
z^g_{r,t,k}
=
\operatorname{norm}
\left(
\frac{1}{|I_{t,k}|}
\sum_{i\in I_{t,k}}
\left[
\operatorname{softmax}
\left(
\frac{(D_{r,t}H_t)H_t^\top}{\sqrt d}
\right)
H_t
\right]_i
\right)
}
$$

The question embedding uses the same frozen E5 model:

$$
q_t
=
\operatorname{norm}
\left(
\operatorname{pool}
\left[
E_{\theta_E}(\texttt{query: }q_t^{raw})
\right]
\right).
$$

Retrieval scores are:

$$
s^{text}_{t,k}=q_t^\top z_{t,k},
\qquad
s^{gaze}_{r,t,k}=q_t^\top z^g_{r,t,k}.
$$

Evidence sets are:

$$
\mathcal{E}^{text}_{t}
=
\operatorname{TopK}_k(s^{text}_{t,k}),
\qquad
\mathcal{E}^{gaze}_{r,t}
=
\operatorname{TopK}_k(s^{gaze}_{r,t,k}).
$$

The frozen LLM answers the multiple-choice question using the same prompt and
decoding settings across conditions:

$$
\hat y^{text}_{t}
=
G(q_t^{raw},\mathcal{Y}_t,\mathcal{E}^{text}_{t}),
\qquad
\hat y^{gaze}_{r,t}
=
G(q_t^{raw},\mathcal{Y}_t,\mathcal{E}^{gaze}_{r,t}).
$$

## 4. Experiments and Expected Results

The minimal comparison set is:

$$
\text{Text-only}
\quad\text{vs.}\quad
\text{Mean-gaze}
\quad\text{vs.}\quad
\text{Reader-specific gaze}
\quad\text{vs.}\quad
\text{Shuffled-reader gaze}.
$$

Mean-gaze uses:

$$
\bar g_t
=
\frac{1}{R_t}
\sum_{r=1}^{R_t}g_{r,t},
\qquad
\bar D_t=\operatorname{diag}(\bar g_t).
$$

Shuffled-reader control uses:

$$
s^{shuf}_{r,t,k}
=
q_t^\top z^g_{\pi(r),t,k},
\qquad
\pi(r)\neq r.
$$

The primary metric is answer accuracy:

$$
\mathrm{Acc}^{gaze}
=
\mathbb{E}_{r,t}
\left[
\mathbf{1}\{\hat y^{gaze}_{r,t}=y_t\}
\right].
$$

If option log-likelihood is available, use it as a secondary metric:

$$
\ell^{gaze}_{r,t}
=
\log p_G(y_t\mid q_t^{raw},\mathcal{Y}_t,\mathcal{E}^{gaze}_{r,t}).
$$

The expected ordering is:

$$
\mathrm{Acc}^{actual}
>
\mathrm{Acc}^{mean}
\geq
\mathrm{Acc}^{text},
\qquad
\mathrm{Acc}^{actual}
>
\mathrm{Acc}^{shuf}.
$$

The first inequality tests whether gaze helps beyond text-only retrieval. The
second tests whether the gain is genuinely reader-specific.

A secondary analysis checks when gaze helps. Define retrieval overlap:

$$
J_{r,t}
=
\frac{
|\mathcal{E}^{gaze}_{r,t}\cap\mathcal{E}^{text}_{t}|
}{
|\mathcal{E}^{gaze}_{r,t}\cup\mathcal{E}^{text}_{t}|
}.
$$

Define gaze concentration:

$$
C^{(p)}_{r,t}
=
\sum_{i\in\operatorname{Top}\ p\%(g_{r,t})}g_{r,t,i}.
$$

The conditional hypothesis is:

$$
\mathbb{E}[\Delta_{r,t}\mid C^{(p)}_{r,t}\ \text{high}]
>
\mathbb{E}[\Delta_{r,t}\mid C^{(p)}_{r,t}\ \text{low}],
$$

where:

$$
\Delta_{r,t}
=
\mathbf{1}\{\hat y^{gaze}_{r,t}=y_t\}
-
\mathbf{1}\{\hat y^{text}_{t}=y_t\}.
$$

The claim after this first experiment should be stated narrowly:

$$
\boxed{
\text{observed reader gaze personalizes post-reading evidence retrieval}
}
$$

not:

$$
\boxed{
\text{future reader response is predicted}
}
$$

The future-prediction version requires calibration:

$$
\hat g_{r,t}
=
C(g_{r,\mathcal{T}_{cal}},H_{\mathcal{T}_{cal}},H_t),
\qquad
|\mathcal{T}_{cal}|\in\{1,2,4,8\}.
$$

Then the RAG system uses:

$$
\hat z^g_{r,t,k}=f(\hat g_{r,t},H_t,I_{t,k})
$$

instead of:

$$
z^g_{r,t,k}=f(g_{r,t},H_t,I_{t,k}).
$$

That second-stage result would support the stronger personalization claim:

$$
\mathrm{Acc}^{\hat g}
>
\mathrm{Acc}^{mean}
>
\mathrm{Acc}^{text}.
$$

