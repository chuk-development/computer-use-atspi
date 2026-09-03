# Computer-use: context, memory, and tool-routing decisions

Durable notes for this project (kept in the repo, not in any agent memory store).
These are the design calls behind the demo loop.

## 1. The image cap is a context problem, not a model or Deep Infra problem

Every step in a naive computer-use loop appends a screenshot to the request. The
provider caps images per request (Deep Infra GLM-5.2 vision ~10; other models
differ; the "30 image" wall is the same class). Fix is provider-independent:
keep only the last N screenshots as real images, drop older ones. Implemented in
`agent_loop.py:prune_images` (N=3). Merging two screenshots into one is rejected:
it halves resolution (worse click grounding) and only postpones the cap.

## 2. Adaptive per-round compaction with a judge (the preferred design)

Plain compaction summarizes the whole history into one blob and is lossy: the
reason a fix failed, or how a component behaves, gets dropped. Better:

- Each round, a cheap second model produces a "pre-compaction" of the running
  context (a running summary / notes), separate from the raw history.
- Each round, decide per message whether to send the full raw context or the
  pre-compacted version. Recent + relevant stays raw; the rest goes as summary.
- The decision is made every turn, not once at a fixed threshold.

Cost of the judge model is acceptable here: the driver model we use is cheap, so
spending a small model on routing/compaction each round is worth it. Note the
cache trade-off: rewriting the context every round breaks prompt-cache reuse, so
the judge should prefer stable prefixes (keep the head verbatim, only rewrite the
tail) to keep as much cache hit as possible.

## 3. What OpenAI Astra and Mem0 do differently from plain embeddings

The hard part is not summarizing, it is not losing what you summarized away.

- Plain compaction: lossy, single summary, no way back to the original detail.
- Embeddings / Mem0: extract salient facts, store as vectors, retrieve top-k by
  semantic similarity each turn. Two weaknesses: what the extractor did not pull
  out is gone, and vector recall is fuzzy (near-miss, wrong chunk).
- Astra (per the GPT-6 launch post): the model writes its own notes across
  context windows AND the earlier context windows stay searchable. So nothing is
  actually deleted; it just leaves the active window and can be retrieved exactly
  (literal search over its own past messages and tool outputs), not only through
  a fuzzy vector match. That is the difference: self-authored notes + exact,
  searchable raw archive, instead of one lossy summary or one fuzzy vector index.

Practical target for us: keep a searchable transcript log the agent can grep
(exact retrieval) on top of the sliding image window and the running summary.
This is the next build step after the base loop runs.

### The design in one line (as stated)

Front stays verbatim, back gets compacted. After each tool round a second cheap
model runs in the background and writes down the important facts, so we only ever
compact ~200k tokens of tail instead of the whole 1M. Then, per next question,
the main model decides whether to send the full tail or its compacted version.
Oldest turns are the ones that go compacted; the front (still likely relevant)
goes in full. This breaks prompt-cache reuse on the tail, which is an accepted
cost. The "model just writes notes that are always shipped / searchable" variant
is basically the same thing from the other end.

## 5. Coding-specific instance: the pseudocode/skeleton repo map

The idea "keep a copy of the whole repo, have a cheaper model write a tiny
pseudocode/skeleton describing every file and function, ship that as the map so
the agent knows exactly which real files to open" is real and already built:

- Aider's **repo map**: tree-sitter parses every file, extracts the public
  symbols (functions, classes, signatures), ranks them with a PageRank-style
  score over the symbol reference graph, and emits a token-budgeted skeleton
  (default ~1k tokens), cached in SQLite. Exactly "architecture as a compact map,
  read the real file only when editing." (aider.chat/2023/10/22/repomap.html)
- Standalone ports: RepoMapper (pdavis68 / Cryect) reproduce it outside Aider.
- LLM-summary variant (closer to the "second dumber model writes pseudocode"
  framing): codebase-summarizer generates summaries at symbol -> file -> folder
  -> repo level plus machine-readable dependency/call/route maps. Academic work
  on code summarization and "semantic compression with LLMs" covers the same.

So the instinct is right and validated. The one gap in most tools: the map is
static signature extraction (tree-sitter), not a living LLM-written pseudocode
that updates and that the agent greps against — the LLM-summary tools exist but
are less mature. That living, grep-able pseudocode map is the part worth building.

## 4. Tool routing: use the API/MCP, fall back to pixels only when forced

Pixel computer-use (screenshot -> click) is the LAST resort, for apps with no
programmatic interface. Where a structured interface exists, use it — it is an
order of magnitude cheaper and does not touch the screenshot cap at all.

- Blender: do NOT drive it by clicking. Use the Blender MCP server (one
  `execute_blender_code` call builds the whole model). The demo that models a
  rocket by clicking in Blender is a showcase, not the efficient path. The other
  model's failure at "model it in 3D" was exactly this: it tried pixels and hit
  the 30-screenshot cap. Route the 3D task to the Blender MCP instead.
- Anything with a CLI, API, or MCP (files, git, HTTP, DB, editors) -> code path.
- Only GUI-only apps with no API -> pixel computer-use.

So the agent should have a router: prefer code/MCP tools, use the computer tool
only when nothing else can reach the target. That keeps the expensive, cap-bound
screenshot loop for the few cases that truly need eyes and a mouse.
