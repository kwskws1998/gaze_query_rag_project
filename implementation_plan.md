# Implementation Plan: Reader-Specific Gaze-Query RAG

## 1. Engineering Goal

Implement a no-training personalized RAG pipeline for OneStop QA:

$$
\mathrm{TRT}_{r,t}
\rightarrow
g_{r,t}
\rightarrow
D_{r,t}H_t
\rightarrow
\widetilde{H}^{g}_{r,t}
\rightarrow
z^g_{r,t,k}
\rightarrow
\operatorname{TopK}
\rightarrow
G
\rightarrow
\hat y_{r,t}.
$$

The first version must be modular, reproducible, and easy to ablate. No model
weights are trained. All personalization comes from reader-specific gaze
vectors.

Primary frozen models:

$$
E=\texttt{intfloat/e5-large-v2}
$$

$$
G=\texttt{meta-llama/Meta-Llama-3-8B-Instruct}
$$

Primary task:

$$
\mathcal{D}_{QA}=\texttt{malmaud/onestop\_qa}
$$

Primary gaze data:

$$
\mathcal{D}_{gaze}=\texttt{ia\_Paragraph.csv.zip}.
$$

## 2. Proposed Directory Layout

```text
gaze_query_rag_project/
  README.md
  project_overview.md
  implementation_plan.md
  pyproject.toml
  configs/
    default.yaml
    smoke.yaml
    llama3.yaml
  src/
    gaze_query_rag/
      __init__.py
      config.py
      schemas.py
      data/
        __init__.py
        hf_onestop_qa.py
        onestop_gaze.py
        alignment.py
        chunking.py
      modeling/
        __init__.py
        encoder.py
        gaze_features.py
        gaze_query_attention.py
      retrieval/
        __init__.py
        index.py
        retriever.py
      generation/
        __init__.py
        prompt.py
        llama.py
        answer_parser.py
      evaluation/
        __init__.py
        metrics.py
        evaluator.py
        analysis.py
      io/
        __init__.py
        cache.py
        artifacts.py
  scripts/
    inspect_onestop_schema.py
    build_aligned_dataset.py
    build_embeddings.py
    run_retrieval.py
    run_generation.py
    run_eval.py
    run_smoke_pipeline.py
  tests/
    test_gaze_features.py
    test_alignment.py
    test_gaze_query_attention.py
    test_retrieval.py
```

## 3. Core Data Schemas

### `schemas.py`

#### `QAExample`

Purpose: canonical representation of a OneStop QA item.

Fields:

- `example_id: str`
- `paragraph_id: str`
- `paragraph_text: str`
- `question: str`
- `choices: list[str]`
- `answer_index: int | None`
- `metadata: dict[str, Any]`

Input source: Hugging Face `malmaud/onestop_qa`.

Output use: retrieval, prompting, evaluation.

#### `GazeRecord`

Purpose: word-level or interest-area-level gaze observation.

Fields:

- `reader_id: str`
- `paragraph_id: str`
- `word_index: int`
- `word: str`
- `trt: float`
- `fixation_count: float | None`
- `skip: float | None`
- `metadata: dict[str, Any]`

Input source: OneStop `ia_Paragraph.csv.zip`.

Output use: gaze distribution construction.

#### `AlignedExample`

Purpose: matched QA item plus all available reader gaze records.

Fields:

- `qa: QAExample`
- `reader_gaze: dict[str, list[GazeRecord]]`

Input: `QAExample` records and `GazeRecord` records.

Output: experiment-ready unit.

#### `Chunk`

Purpose: retrieval unit.

Fields:

- `chunk_id: str`
- `paragraph_id: str`
- `text: str`
- `char_start: int`
- `char_end: int`
- `token_indices: list[int] | None`

Input: paragraph text.

Output: text-only and gaze-view embeddings.

#### `RetrievalResult`

Purpose: top-k retrieved evidence for one query.

Fields:

- `example_id: str`
- `reader_id: str | None`
- `condition: str`
- `ranked_chunks: list[tuple[str, float]]`

Output use: prompting and retrieval analysis.

#### `PredictionRecord`

Purpose: model answer output for one condition.

Fields:

- `example_id: str`
- `reader_id: str | None`
- `condition: str`
- `prompt: str`
- `raw_output: str`
- `predicted_index: int | None`
- `gold_index: int | None`
- `metadata: dict[str, Any]`

Output use: final evaluation.

## 4. Data Loading and Alignment

### `data/hf_onestop_qa.py`

#### `load_onestop_qa(dataset_name: str, split: str | None, cache_dir: Path | None) -> list[QAExample]`

Purpose: load OneStop QA from Hugging Face and convert rows into `QAExample`.

Inputs:

- `dataset_name`: expected default `malmaud/onestop_qa`
- `split`: dataset split, or `None` if the dataset exposes only one split
- `cache_dir`: optional Hugging Face cache directory

Outputs:

- list of normalized `QAExample`

Implementation notes:

- Do not assume the schema is stable.
- Inspect column names at runtime.
- Store the raw row in `metadata`.
- Raise a clear error if paragraph/question/answer fields cannot be inferred.

#### `infer_qa_schema(columns: list[str]) -> dict[str, str]`

Purpose: infer likely paragraph, question, choices, and answer columns.

Inputs:

- list of dataset column names

Outputs:

- mapping such as `{paragraph: ..., question: ..., choices: ..., answer: ...}`

Failure mode:

- raise `SchemaInferenceError` if required fields are ambiguous.

### `data/onestop_gaze.py`

#### `load_onestop_ia(path: Path, columns: list[str] | None = None) -> pd.DataFrame`

Purpose: load OneStop interest-area CSV from a zipped file.

Inputs:

- `path`: path to `ia_Paragraph.csv.zip`
- `columns`: optional column subset

Outputs:

- raw interest-area dataframe

Implementation notes:

- Use pandas `read_csv` directly on zip if possible.
- Preserve original column names.
- Avoid hard-coding the final schema before inspection.

#### `normalize_gaze_schema(df: pd.DataFrame) -> pd.DataFrame`

Purpose: convert raw OneStop column names into canonical names.

Inputs:

- raw OneStop IA dataframe

Outputs:

- dataframe with canonical columns:
  - `reader_id`
  - `paragraph_id`
  - `word_index`
  - `word`
  - `trt`
  - optional `fixation_count`
  - optional `skip`

Failure mode:

- raise `SchemaInferenceError` if reader, paragraph, word, or TRT fields are missing.

#### `to_gaze_records(df: pd.DataFrame) -> list[GazeRecord]`

Purpose: convert canonical gaze dataframe into typed records.

Inputs:

- canonical gaze dataframe

Outputs:

- list of `GazeRecord`

### `data/alignment.py`

#### `align_qa_with_gaze(qa_examples: list[QAExample], gaze_records: list[GazeRecord]) -> list[AlignedExample]`

Purpose: align QA paragraphs with gaze paragraphs.

Inputs:

- normalized QA examples
- normalized gaze records

Outputs:

- list of `AlignedExample`

Implementation notes:

- Prefer exact paragraph IDs if shared.
- If IDs differ, use deterministic text matching or metadata matching.
- Report unmatched QA items and unmatched gaze paragraphs.

#### `build_alignment_report(aligned: list[AlignedExample], qa_examples: list[QAExample], gaze_records: list[GazeRecord]) -> dict[str, Any]`

Purpose: produce a reproducibility report for alignment quality.

Inputs:

- aligned examples
- original QA examples
- original gaze records

Outputs:

- counts, unmatched IDs, reader coverage, paragraph coverage

### `data/chunking.py`

#### `chunk_paragraph(paragraph_id: str, text: str, strategy: str, max_words: int, stride: int = 0) -> list[Chunk]`

Purpose: split paragraph into retrieval units.

Inputs:

- paragraph ID
- paragraph text
- strategy: `sentence`, `fixed_words`, or `whole_paragraph`
- maximum words per chunk
- optional overlap stride

Outputs:

- list of `Chunk`

Initial recommendation:

- start with sentence-level chunking
- keep whole-paragraph as a sanity baseline

## 5. Gaze Feature Construction

### `modeling/gaze_features.py`

#### `word_trt_to_token_trt(word_trt: np.ndarray, word_to_token_indices: list[list[int]], num_tokens: int) -> np.ndarray`

Purpose: map word-level TRT to token-level TRT.

Inputs:

- `word_trt`: shape \((M,)\)
- `word_to_token_indices`: maps each word to tokenizer subtoken indices
- `num_tokens`: total token count \(L\)

Outputs:

- token-level TRT vector, shape \((L,)\)

Formula:

$$
\mathrm{TRT}_{r,t,i}
=
\frac{\mathrm{TRT}_{r,t,m}}{|S_{t,m}|},
\qquad
i\in S_{t,m}.
$$

#### `compute_gaze_distribution(token_trt: np.ndarray, eps: float = 1e-8, transform: str = "log1p") -> np.ndarray`

Purpose: convert token-level TRT into a probability distribution.

Inputs:

- token TRT vector, shape \((L,)\)
- numerical epsilon
- transform type

Outputs:

- gaze distribution \(g_{r,t}\), shape \((L,)\)

Formula:

$$
a_i=\log(1+\mathrm{TRT}_i),
\qquad
g_i=\frac{a_i+\epsilon}{\sum_j(a_j+\epsilon)}.
$$

#### `compute_mean_gaze(gaze_by_reader: dict[str, np.ndarray]) -> np.ndarray`

Purpose: construct mean-gaze baseline.

Inputs:

- reader ID to gaze distribution

Outputs:

- mean gaze vector \(\bar g_t\)

Formula:

$$
\bar g_t=\frac{1}{R_t}\sum_{r=1}^{R_t}g_{r,t}.
$$

#### `shuffle_reader_gaze(gaze_by_reader: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]`

Purpose: construct shuffled-reader null control.

Inputs:

- reader-specific gaze distributions
- random seed

Outputs:

- mapping from reader ID to another reader's gaze vector

Constraint:

$$
\pi(r)\neq r.
$$

## 6. Text Encoding

### `modeling/encoder.py`

#### `load_e5_encoder(model_name: str, device: str, cache_dir: Path | None) -> EncoderBundle`

Purpose: load frozen tokenizer and encoder.

Inputs:

- model name, default `intfloat/e5-large-v2`
- device
- cache directory

Outputs:

- `EncoderBundle(tokenizer, model, device, hidden_size)`

Implementation notes:

- call `model.eval()`
- disable gradients globally for encoding

#### `encode_passage_tokens(bundle: EncoderBundle, text: str, max_length: int) -> TokenEncoding`

Purpose: produce last-hidden-state token embeddings for a paragraph.

Inputs:

- encoder bundle
- paragraph text
- maximum token length

Outputs:

- `TokenEncoding`
  - `input_ids`
  - `attention_mask`
  - `tokens`
  - `hidden_states`, shape \((L,d)\)
  - offset mappings if available

Formula:

$$
H_t=E_{\theta_E}(\texttt{passage: }x_t)\in\mathbb{R}^{L\times d}.
$$

#### `encode_query(bundle: EncoderBundle, question: str, max_length: int) -> np.ndarray`

Purpose: produce normalized E5 query embedding.

Inputs:

- encoder bundle
- question text
- maximum token length

Outputs:

- normalized query vector \(q_t\), shape \((d,)\)

Formula:

$$
q_t=\operatorname{norm}(\operatorname{pool}(E_{\theta_E}(\texttt{query: }q_t^{raw}))).
$$

#### `build_word_to_token_alignment(text: str, words: list[str], token_offsets: list[tuple[int, int]]) -> list[list[int]]`

Purpose: align OneStop word-level gaze records to tokenizer subtokens.

Inputs:

- original paragraph text
- OneStop word sequence
- tokenizer character offsets

Outputs:

- list mapping each word index to subtoken indices

Failure mode:

- emit alignment diagnostics if a word cannot be matched.

## 7. Gaze-Query Attention

### `modeling/gaze_query_attention.py`

#### `gaze_weight_tokens(hidden: np.ndarray, gaze: np.ndarray) -> np.ndarray`

Purpose: compute \(D_{r,t}H_t\) without instantiating \(D_{r,t}\).

Inputs:

- token embeddings \(H_t\), shape \((L,d)\)
- gaze vector \(g_{r,t}\), shape \((L,)\)

Outputs:

- gaze-weighted token embeddings, shape \((L,d)\)

Formula:

$$
H^g_{r,t}=g_{r,t}[:,None]\odot H_t.
$$

#### `gaze_query_attention(hidden: np.ndarray, gaze: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray`

Purpose: compute no-training gaze-query attention.

Inputs:

- original token embeddings \(H_t\), shape \((L,d)\)
- gaze vector \(g_{r,t}\), shape \((L,)\)
- optional attention mask

Outputs:

- gaze-conditioned token embeddings \(\widetilde{H}^{g}_{r,t}\), shape \((L,d)\)

Formula:

$$
\widetilde{H}^{g}_{r,t}
=
\operatorname{softmax}
\left(
\frac{(D_{r,t}H_t)H_t^\top}{\sqrt d}
\right)
H_t.
$$

Implementation notes:

- use stable softmax
- mask padding tokens
- run in torch for GPU efficiency

#### `pool_chunk_embedding(hidden: np.ndarray, token_indices: list[int], normalize: bool = True) -> np.ndarray`

Purpose: pool token states into a chunk embedding.

Inputs:

- token embeddings, shape \((L,d)\)
- token indices for chunk \(I_{t,k}\)
- normalization flag

Outputs:

- chunk embedding, shape \((d,)\)

Formula:

$$
z_{t,k}
=
\operatorname{norm}
\left(
\frac{1}{|I_{t,k}|}
\sum_{i\in I_{t,k}}h_{t,i}
\right).
$$

#### `build_gaze_view_chunk_embeddings(hidden: np.ndarray, gaze: np.ndarray, chunks: list[Chunk]) -> dict[str, np.ndarray]`

Purpose: build reader-specific gaze-view embeddings for all chunks in one paragraph.

Inputs:

- original token embeddings
- reader gaze vector
- chunks with token index sets

Outputs:

- mapping from `chunk_id` to \(z^g_{r,t,k}\)

Formula:

$$
z^g_{r,t,k}
=
\operatorname{pool}_{i\in I_{t,k}}
\left(
\widetilde h^g_{r,t,i}
\right).
$$

#### `build_text_chunk_embeddings(hidden: np.ndarray, chunks: list[Chunk]) -> dict[str, np.ndarray]`

Purpose: build text-only baseline chunk embeddings.

Inputs:

- original token embeddings
- chunks

Outputs:

- mapping from `chunk_id` to \(z_{t,k}\)

## 8. Retrieval

### `retrieval/index.py`

#### `build_in_memory_index(chunk_embeddings: dict[str, np.ndarray]) -> EmbeddingIndex`

Purpose: create a simple cosine-search index.

Inputs:

- chunk ID to normalized embedding

Outputs:

- `EmbeddingIndex(ids, matrix)`

Implementation notes:

- first version can use matrix multiplication
- FAISS can be added later if needed

#### `search_index(index: EmbeddingIndex, query: np.ndarray, top_k: int) -> list[tuple[str, float]]`

Purpose: retrieve top-k chunks.

Inputs:

- embedding index
- normalized query embedding
- top-k

Outputs:

- ranked list of `(chunk_id, score)`

Formula:

$$
s_{t,k}=q_t^\top z_{t,k}.
$$

### `retrieval/retriever.py`

#### `retrieve_text_only(example: QAExample, chunks: list[Chunk], text_embeddings: dict[str, np.ndarray], query: np.ndarray, top_k: int) -> RetrievalResult`

Purpose: run baseline retrieval.

Inputs:

- QA example
- chunks
- text-only chunk embeddings
- query embedding
- top-k

Outputs:

- `RetrievalResult(condition="text")`

#### `retrieve_gaze_view(example: QAExample, reader_id: str, chunks: list[Chunk], gaze_embeddings: dict[str, np.ndarray], query: np.ndarray, top_k: int, condition: str) -> RetrievalResult`

Purpose: run gaze-conditioned retrieval for actual, mean, or shuffled gaze.

Inputs:

- QA example
- reader ID
- chunks
- gaze-view chunk embeddings
- query embedding
- top-k
- condition label

Outputs:

- `RetrievalResult(condition="actual" | "mean" | "shuffled")`

## 9. Generation

### `generation/prompt.py`

#### `format_mcqa_prompt(question: str, choices: list[str], evidence_chunks: list[Chunk]) -> str`

Purpose: format a deterministic multiple-choice QA prompt.

Inputs:

- question
- answer choices
- retrieved evidence chunks

Outputs:

- prompt string

Constraint:

- same prompt template for all conditions
- only retrieved evidence differs

#### `format_choice_block(choices: list[str]) -> str`

Purpose: convert choices into stable labels.

Inputs:

- list of choices

Outputs:

- formatted choices such as `A. ...`, `B. ...`

### `generation/llama.py`

#### `load_llama_generator(model_name: str, device: str, cache_dir: Path | None, dtype: str) -> GeneratorBundle`

Purpose: load frozen Llama 3 Instruct generator.

Inputs:

- model name, default `meta-llama/Meta-Llama-3-8B-Instruct`
- device
- cache directory
- dtype

Outputs:

- `GeneratorBundle(tokenizer, model, device)`

#### `generate_answer(bundle: GeneratorBundle, prompt: str, max_new_tokens: int, temperature: float) -> str`

Purpose: generate answer text.

Inputs:

- generator bundle
- prompt
- decoding parameters

Outputs:

- raw decoded output

Initial decoding:

- temperature \(=0\)
- short answer format

#### `score_answer_options(bundle: GeneratorBundle, prompt_prefix: str, choices: list[str]) -> np.ndarray`

Purpose: optionally compute option log-likelihoods instead of free-form generation.

Inputs:

- generator bundle
- prompt without answer
- answer choices

Outputs:

- log-likelihood vector, shape \((|\mathcal{Y}_t|,)\)

Preferred use:

- more stable evaluation than parsing free-form generations.

### `generation/answer_parser.py`

#### `parse_choice(raw_output: str, num_choices: int) -> int | None`

Purpose: parse generated answer into choice index.

Inputs:

- raw LLM output
- number of choices

Outputs:

- predicted choice index or `None`

## 10. Evaluation and Analysis

### `evaluation/metrics.py`

#### `accuracy(predictions: list[PredictionRecord]) -> float`

Purpose: compute MCQA accuracy.

Inputs:

- prediction records with parsed and gold labels

Outputs:

- scalar accuracy

Formula:

$$
\mathrm{Acc}
=
\mathbb{E}
\left[
\mathbf{1}\{\hat y=y\}
\right].
$$

#### `paired_accuracy_delta(a: list[PredictionRecord], b: list[PredictionRecord]) -> dict[str, float]`

Purpose: compute paired condition difference.

Inputs:

- predictions from condition A
- predictions from condition B

Outputs:

- mean delta, standard error, bootstrap confidence interval

Formula:

$$
\Delta
=
\mathbb{E}
\left[
\mathbf{1}\{\hat y^a=y\}
-
\mathbf{1}\{\hat y^b=y\}
\right].
$$

#### `retrieval_jaccard(a: RetrievalResult, b: RetrievalResult) -> float`

Purpose: measure retrieval-set shift.

Inputs:

- two retrieval results for the same QA item

Outputs:

- Jaccard overlap

Formula:

$$
J
=
\frac{|A\cap B|}{|A\cup B|}.
$$

#### `gaze_concentration(gaze: np.ndarray, top_percent: float) -> float`

Purpose: compute top-mass concentration.

Inputs:

- gaze vector
- top percentage

Outputs:

- scalar concentration

Formula:

$$
C^{(p)}
=
\sum_{i\in \operatorname{Top}\ p\%(g)}g_i.
$$

### `evaluation/evaluator.py`

#### `evaluate_predictions(predictions: list[PredictionRecord]) -> pd.DataFrame`

Purpose: aggregate results by condition and reader.

Inputs:

- prediction records

Outputs:

- dataframe with accuracy, count, and missing parse rate

#### `compare_conditions(predictions: list[PredictionRecord], baseline: str, treatment: str) -> dict[str, Any]`

Purpose: paired comparison between baseline and treatment.

Inputs:

- all predictions
- baseline condition name
- treatment condition name

Outputs:

- paired delta summary

### `evaluation/analysis.py`

#### `analyze_when_gaze_helps(predictions: list[PredictionRecord], retrievals: list[RetrievalResult], gaze_stats: pd.DataFrame) -> pd.DataFrame`

Purpose: test conditional gain by gaze concentration and retrieval shift.

Inputs:

- predictions
- retrieval results
- gaze statistics

Outputs:

- dataframe of conditional deltas

Primary analysis:

$$
\mathbb{E}[\Delta_{r,t}\mid C^{(p)}_{r,t}\ \text{high}]
>
\mathbb{E}[\Delta_{r,t}\mid C^{(p)}_{r,t}\ \text{low}].
$$

## 11. Pipeline Scripts

### `scripts/inspect_onestop_schema.py`

Purpose: inspect Hugging Face QA schema and OneStop IA schema.

Outputs:

- `artifacts/schema/onestop_qa_columns.json`
- `artifacts/schema/onestop_ia_columns.json`
- human-readable schema report

### `scripts/build_aligned_dataset.py`

Purpose: build aligned QA-plus-gaze dataset.

Inputs:

- HF dataset name
- OneStop IA zip path
- output path

Outputs:

- `artifacts/data/aligned_examples.jsonl`
- `artifacts/data/alignment_report.json`

### `scripts/build_embeddings.py`

Purpose: encode paragraphs, build chunks, construct text-only and gaze-view embeddings.

Outputs:

- `artifacts/embeddings/text_chunks.npz`
- `artifacts/embeddings/gaze_chunks_actual.npz`
- `artifacts/embeddings/gaze_chunks_mean.npz`
- `artifacts/embeddings/gaze_chunks_shuffled.npz`

### `scripts/run_retrieval.py`

Purpose: run top-k retrieval for all conditions.

Outputs:

- `artifacts/retrieval/retrieval_results.jsonl`

### `scripts/run_generation.py`

Purpose: call frozen LLM using retrieved evidence.

Outputs:

- `artifacts/predictions/predictions.jsonl`

### `scripts/run_eval.py`

Purpose: compute final metrics and condition comparisons.

Outputs:

- `artifacts/results/summary.json`
- `artifacts/results/by_condition.csv`
- `artifacts/results/paired_deltas.csv`

### `scripts/run_smoke_pipeline.py`

Purpose: execute a small end-to-end run.

Inputs:

- `--limit-examples`
- `--limit-readers`
- `--top-k`

Outputs:

- smoke-test artifacts under `artifacts/smoke/`

## 12. Implementation Milestones

### Milestone 1: Schema and Alignment

Deliverables:

- load Hugging Face QA examples
- load OneStop IA data
- normalize gaze schema
- align QA paragraphs with gaze paragraphs
- produce alignment report

Exit criteria:

- at least one aligned QA paragraph with multiple readers
- deterministic alignment report

### Milestone 2: No-Training Embeddings

Deliverables:

- E5 token encoder wrapper
- word-to-token alignment
- token-level gaze vector construction
- text-only chunk embeddings
- actual, mean, shuffled gaze-view chunk embeddings

Exit criteria:

- embeddings have consistent dimension \(d\)
- gaze vectors sum to 1
- no NaNs in embeddings
- shuffled-reader control satisfies \(\pi(r)\neq r\)

### Milestone 3: Retrieval

Deliverables:

- text-only retrieval
- actual gaze retrieval
- mean gaze retrieval
- shuffled gaze retrieval
- retrieval overlap analysis

Exit criteria:

- top-k evidence is produced for every condition
- actual gaze retrieval differs from text-only for nontrivial cases

### Milestone 4: Generation and QA Evaluation

Deliverables:

- deterministic MCQA prompt
- frozen Llama generation or option scoring
- parsed predictions
- accuracy by condition
- paired deltas

Exit criteria:

- valid prediction records for all conditions
- missing parse rate reported
- paired text vs actual and actual vs shuffled comparisons reported

### Milestone 5: Conditional Analysis

Deliverables:

- gaze concentration statistics
- retrieval Jaccard overlap
- conditional gain by gaze concentration
- reader-level gain distribution

Exit criteria:

- result table identifies whether gaze helps globally or only in high-concentration cases

## 13. Initial Experimental Configuration

Recommended smoke config:

```yaml
dataset:
  qa_name: malmaud/onestop_qa
  gaze_path: /Users/wansookim/Documents/OneStop-Eye-Movements/data/OneStop/ia_Paragraph.csv.zip
  max_examples: 20
  max_readers_per_example: 5

models:
  encoder: intfloat/e5-large-v2
  generator: meta-llama/Meta-Llama-3-8B-Instruct
  encoder_max_length: 512
  generator_max_new_tokens: 16

retrieval:
  chunk_strategy: sentence
  top_k: 3

conditions:
  - text
  - mean_gaze
  - actual_gaze
  - shuffled_gaze

gaze:
  measure: TRT
  transform: log1p
  eps: 1.0e-8
```

Recommended first full-run config:

```yaml
dataset:
  max_examples: null
  max_readers_per_example: null

retrieval:
  chunk_strategy: sentence
  top_k: 3

generation:
  mode: option_loglikelihood
  temperature: 0.0
```

## 14. Key Design Constraints

1. Target gaze is allowed only for the post-reading RAG experiment.
2. Do not claim future prediction until \(g_{r,t}\) is replaced by \(\hat g_{r,t}\).
3. Keep the frozen encoder and frozen generator identical across conditions.
4. Keep the prompt identical across conditions.
5. Only evidence retrieval may differ across conditions.
6. Always report shuffled-reader control.
7. Always report mean-gaze baseline.
8. Report both global gain and conditional gain.

