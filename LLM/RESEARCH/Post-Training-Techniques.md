### Key Points on Post-Training Techniques

- Research suggests Kimi K2 emphasizes verifiable rewards and self-critique in RL for robust agentic alignment, potentially offering stronger generalization in open-ended tasks compared to GLM-4.6's RLHF focus on human preferences and MiniMax M2's CISPO for gradient stability.
- GLM-4.6 appears to prioritize efficiency in token usage and bilingual data during SFT, which may enhance practical coding and reasoning, though it lags in explicit long-horizon agentic synthesis seen in the others.
- MiniMax M2's interleaved thinking and curriculum-based RL likely provide an edge in dynamic, multi-step workflows, but all three models share a core reliance on SFT followed by RL variants to balance scalability and performance without clear evidence of one universally outperforming the rest.

#### Comparative Overview

Post-training for these models—Kimi K2 (Moonshot AI), GLM-4.6 (Zhipu AI), and MiniMax M2 (MiniMax AI)—builds on pre-trained MoE architectures to infuse agentic, reasoning, and coding capabilities. Kimi K2 uses a multi-stage pipeline with heavy synthetic data synthesis for tool use, GLM-4.6 focuses on RLHF for polished alignment, and MiniMax M2 integrates CISPO for efficient RL scaling. These choices reflect trade-offs in data efficiency, stability, and generalization, with no model showing definitive superiority across all benchmarks.

#### Strengths and Trade-Offs

Kimi K2 excels in verifiable RL for safety and tool orchestration, ideal for complex agents but potentially compute-intensive. GLM-4.6 offers balanced bilingual support and tool integration, suiting global coding tasks, while MiniMax M2's interleaved approach enhances real-time adaptability at lower costs. Evidence leans toward context-dependent performance, with users encouraged to test via platforms like Hugging Face (e.g., [Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Thinking), [GLM-4.6](https://huggingface.co/zai-org/GLM-4.6), [MiniMax M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)).

---

As large language models (LLMs) evolve toward agentic intelligence—capabilities enabling autonomous planning, tool use, and multi-step reasoning—the post-training phase becomes pivotal in transforming raw pre-trained parameters into practical, aligned systems. This comprehensive analysis compares the post-training techniques of three leading open-weight MoE models: Moonshot AI's Kimi K2 (1.04T total parameters, 32.6B active), Zhipu AI's GLM-4.6 (355B total, ~32B active), and MiniMax AI's MiniMax M2 (230B total, 10B active). Released in 2025, these models represent a shift in open-source AI, emphasizing efficiency, stability, and generalization for agentic tasks like coding and long-horizon workflows. Drawing from technical reports, papers, and evaluations, we dissect each component—supervised fine-tuning (SFT), reinforcement learning (RL), data synthesis, alignment methods, and specialized enhancements—with first-principles reasoning on why these choices were made, grounded in machine learning fundamentals such as gradient stability, reward hacking prevention, and scaling laws.

Post-training, occurring after initial pre-training on vast corpora (e.g., 15T+ tokens for each), refines models through targeted optimization to align with human-like behaviors. All three employ a sequential SFT-to-RL pipeline, but diverge in data strategies, optimizers, and agentic focuses. Kimi K2 prioritizes synthetic agentic trajectories for verifiable alignment, GLM-4.6 leverages RLHF for preference-based polishing, and MiniMax M2 uses CISPO for efficient, stable RL scaling. These reflect principled trade-offs: Kimi seeks robustness in uncertain environments, GLM efficiency in bilingual coding, and MiniMax low-cost generalization for deployment.

#### Architecture and Pre-Training Context

Before delving into post-training, note that all are sparse MoE Transformers, activating subsets of parameters for efficiency (e.g., 8-10 experts per token). Pre-training establishes foundational priors, but post-training injects agentics. Kimi K2's MuonClip optimizer ensures zero instability over 15.5T tokens, GLM-4.6's 15T corpus includes bilingual data for broader coverage, and MiniMax M2's hybrid attention enables long-context (512K tokens) with near-linear compute. These bases influence post-training: sparse MoEs require careful gradient handling to avoid expert silos, motivating stability-focused methods like clipping.

#### Supervised Fine-Tuning (SFT): Building Instructional Foundations

SFT aligns models to follow instructions via demonstration, maximizing likelihood on curated prompt-response pairs. From first principles, SFT exploits supervised learning's efficiency in transferring patterns, but risks mode collapse without diversity—hence the emphasis on high-quality, varied data.

- **Kimi K2**: Employs Muon optimizer (consistent with pre-training) on diverse prompts spanning web, code, and agentic domains. Data is synthesized via expert models (e.g., K1.5) with automated filtering by LLMs or humans for quality. Agentic focus: ~20K synthetic tools and multi-turn trajectories simulate real interactions, augmented with sandboxes for coding authenticity.

  - **Why?** Muon maximizes token utility (20-30% better than AdamW per scaling laws), preventing overfitting in sparse MoEs where parameters exceed data volume. Diversity hedges against distribution shifts, ensuring generalization to unseen tasks—a principle from information theory where entropy boosts learning signals.

- **GLM-4.6**: Fine-tuned on 7T additional tokens emphasizing code and reasoning, building on a 15T pre-training corpus with bilingual (English-Chinese) content. Includes hybrid reasoning modes (fast vs. thinking) toggled via parameters.

  - **Why?** Bilingual data addresses data scarcity in non-English domains, improving cross-lingual transfer per compositionality principles. Token efficiency (30% better than GLM-4.5) stems from focused curation, aligning with compute constraints in MoEs where active params must scale without proportional costs.

- **MiniMax M2**: SFT injects CoT patterns on ~60% math/coding data, plus STEM, writing, and chats. Curated for long CoT responses to prime RL.
  - **Why?** CoT data bootstraps reflective thinking, rooted in chain decomposition for complex problems—reducing error propagation in long-horizon tasks. Curriculum balances difficulty, preventing catastrophic forgetting per stability-plasticity dilemma.

| Model          | SFT Data Scale                        | Key Focus                               | Optimizer                            | First-Principles Rationale                                              |
| -------------- | ------------------------------------- | --------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| **Kimi K2**    | Diverse prompts + 20K synthetic tools | Agentic trajectories, quality filtering | Muon                                 | Token efficiency via high-entropy data; avoids silos in sparse MoEs.    |
| **GLM-4.6**    | 7T code/reasoning + bilingual         | Hybrid modes, tool integration          | Not specified (likely AdamW variant) | Bilingual coverage for generalization; efficiency in token consumption. |
| **MiniMax M2** | CoT-heavy (60% math/code)             | Reasoning priming for RL                | Custom AdamW (β1=0.9, β2=0.95)       | Curriculum for stability; CoT decomposition for error reduction.        |

#### Reinforcement Learning (RL): Refining with Feedback

RL optimizes policies via rewards, addressing SFT's limitations in subjective or dynamic tasks. Principles: Verifiable signals prevent hacking; clipping stabilizes off-policy variance.

- **Kimi K2**: RLVR (verifiable rewards) + self-critique rubrics. Domains: Math/logic (~50K problems), coding (GitHub PRs), safety (adversarial). Equation: Policy optimization with pairwise comparisons, PTX loss for forgetting prevention, budget controls, temperature decay.

  - **Why?** Verifiable rewards ground alignment in objective outcomes, mitigating hacking per Goodhart's law. Self-critique extends to subjective tasks via closed-loop refinement, ensuring coherence across horizons—vital for agentics where feedback loops amplify errors.

- **GLM-4.6**: RLHF aligns with human preferences, incorporating tool calls (e.g., APIs) in RL tasks for agentic behavior.

  - **Why?** RLHF simplifies preference modeling (Bradley-Terry), enhancing fluency and safety without explicit rewards—efficient for bilingual polishing, per distribution matching to human data.

- **MiniMax M2**: CISPO clips importance sampling weights, unifying GRPO/DAPO. Verifiable (math ~50K, coding ~30K, SE sandboxes) + non-verifiable (GenRM for pairwise comparisons). Curriculum: Verifiable first, then general.
  - **Why?** CISPO preserves gradients from reflective tokens, reducing variance in off-policy RL (2x speedup)—rooted in variance reduction for stability. Curriculum fosters transfer learning, balancing specialization and breadth per plasticity principles.

| Model          | RL Method            | Reward Types                                 | Enhancements               | First-Principles Rationale                                                       |
| -------------- | -------------------- | -------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------- |
| **Kimi K2**    | RLVR + Self-Critique | Verifiable (objective) + Rubric (subjective) | PTX loss, budget control   | Objective grounding prevents hacking; closed-loop for subjective generalization. |
| **GLM-4.6**    | RLHF                 | Human preferences                            | Tool integration in tasks  | Preference-based simplicity; efficient for style alignment.                      |
| **MiniMax M2** | CISPO                | Rule-based (verifiable) + GenRM (pairwise)   | Curriculum, value clipping | Gradient preservation for stability; variance reduction in long contexts.        |

#### Data Synthesis and Alignment: Scaling Quality

Data drives post-training; synthesis addresses scarcity.

- **Kimi K2**: Agentic pipeline: Tool specs (GitHub + synthetic), tasks with rubrics, hybrid simulations. QAT for INT4 during RL.

  - **Why?** Synthesis maximizes signal density (5% QA gains), QAT simulates quantization noise early—per information quantization theory for deployment efficiency.

- **GLM-4.6**: Bilingual corpora + code-focused tokens; adversarial for safety.

  - **Why?** Diversity counters bias, adversarial hardens robustness—principles from game theory for equilibrium alignment.

- **MiniMax M2**: Full-trajectory generalization: Perturbations across workflows. Interleaved thinking preserves context in agents.
  - **Why?** Trajectories simulate real disturbances, interleaved enables dynamic adaptation—per control theory for closed-loop stability.

#### Specialized Enhancements: Agentic and Efficiency Focus

- **Kimi K2**: Interleaved tool calls (200-300), self-critique for creativity.

  - **Why?** Interleaving maintains coherence in feedback loops, critique bridges verifiable-subjective gaps.

- **GLM-4.6**: 200K context, hybrid modes for reasoning depth.

  - **Why?** Long context via YaRN extrapolation preserves priors, modes optimize compute trade-offs.

- **MiniMax M2**: Lightning attention for 80K+ tokens, early truncation.
  - **Why?** Linear attention scales FLOPs sub-quadratically, truncation prevents loops for efficiency.

#### Benchmark Insights and Trade-Offs

All achieve SOTA in agentics (e.g., Kimi: 60.2% BrowseComp; GLM: Strong CC-Bench; MiniMax: 69.4% SWE-Bench). Kimi's synthesis excels in tools, GLM in bilingual coding, MiniMax in speed (2x Claude). Costs: Kimi ~$1M extra post-training, MiniMax $0.53M RL.

| Benchmark          | Kimi K2 | GLM-4.6 | MiniMax M2 | Insight                                    |
| ------------------ | ------- | ------- | ---------- | ------------------------------------------ |
| SWE-Bench Verified | 71.3%   | ~67%    | 69.4%      | Agentic coding edge from RL verifiability. |
| BrowseComp         | 60.2%   | Lower   | 44-48%     | Interleaving boosts tool orchestration.    |
| AIME (tools)       | 99.1%   | 98.6%   | 78%        | Synthesis enhances math augmentation.      |

In conclusion, these techniques embody principled advancements: Kimi for verifiable robustness, GLM for efficient alignment, MiniMax for stable scaling. Future iterations may hybridize, but current evidence highlights context-specific strengths in agentic AI.

#### Key Citations

- [arXiv: Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534)
- [arXiv: MiniMax-M1 Paper](https://arxiv.org/abs/2506.13585)
- [IntuitionLabs: GLM-4.6 Report](https://intuitionlabs.ai/pdf-data/pdfs/glm-4-6-an-open-source-ai-for-coding-vs-sonnet-gpt-5.pdf)
- [Hugging Face: MiniMax-M2 Model Card](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [Zhipu AI: GLM-4.6 Docs](https://docs.z.ai/guides/llm/glm-4.6)
- [Interconnects: Kimi K2 Analysis](https://www.interconnects.ai/p/kimi-k2-thinking-what-it-means)
- [MarkTechPost: GLM-4.6 Release](https://www.marktechpost.com/2025/09/30/zhipu-ai-releases-glm-4-6-achieving-enhancements-in-real-world-coding-long-context-processing-reasoning-searching-and-agentic-ai/)
- [YouTube: MiniMax M2 Interleaved Reasoning](https://www.youtube.com/watch?v=KPawqoyzk7U)
- [Zhihu Frontier: MiniMax M2 Blog](https://zhuanlan.zhihu.com/p/728456789)

### Key Insights on Post-Training Data Examples

- Research indicates that while specific real-world data examples from the models' training are not publicly disclosed, illustrative examples can be derived from described processes in technical reports, showing how post-training enhances agentic and coding capabilities.
- Kimi K2's post-training emphasizes synthetic agentic trajectories for tool use, potentially leading to more robust multi-step reasoning compared to GLM-4.6's focus on bilingual coding data and RLHF, or MiniMax M2's interleaved thinking via CISPO for efficiency.
- Evidence suggests all models use SFT followed by RL variants, but exact templates vary; generated examples here are plausible reconstructions based on documented methods, with hedging for proprietary details.

### Overview of Processes

Post-training for these models involves SFT to align with instructions and RL to refine behaviors, using synthetic and curated data. Variants like "Instruct" or "Thinking" modes adapt for specific tasks. Below, we outline generated data examples, templates, and prompts grounded in available technical details.

### Model-Specific Examples

For each model, examples illustrate the entire process, including variants.

#### Kimi K2 (Moonshot AI)

Post-training includes SFT on diverse prompts and RL with verifiable rewards (RLVR) and self-critique. Data synthesis creates tool-use trajectories.

- **SFT Stage Example**: A prompt-response pair for instruction following.

  - **Template**: {"prompt": "User query with context", "response": "Detailed answer with steps"}
  - **Specific Data Example**: Prompt: "Calculate the area of a circle with radius 5." Response: "First, recall the formula: πr². For r=5, area=3.1416\*25=78.54."

- **RL Stage Example (Verifiable Domain)**: Math task with reward based on accuracy.

  - **Template**: {"task": "Solve equation", "trajectory": "Step-by-step solution", "reward": "1 if correct"}
  - **Specific Data Example**: Task: "Solve 2x + 3 = 7." Trajectory: "Subtract 3: 2x=4. Divide by 2: x=2." Reward: 1 (verified via interpreter).

- **Agentic Data Synthesis Example**: Multi-turn tool-use trajectory.

  - **Template**: {"tools": ["calculator", "search"], "user": "Query", "agent": "Tool calls and reasoning"}
  - **Specific Data Example**: Tools: ["weather_api"]. User: "What's the weather in Tokyo?" Agent: "Call weather_api(location='Tokyo'). Response: Sunny, 25°C."

- **Variants**:
  - **Kimi-K2-Instruct**: Optimized for chat; example prompt: "Explain quantum computing simply." Response: "Quantum bits can be 0 and 1 simultaneously..."
  - **Kimi-K2-Thinking**: For deep reasoning; template with budget control: {"prompt": "Complex problem", "think_steps": "Interleaved thoughts", "output": "Final answer"}. Example: Prompt: "Plan a trip to Mars." Think_steps: "Step 1: Fuel needs. Step 2: Trajectory calc."

#### GLM-4.6 (Zhipu AI)

Focuses on SFT with code/reasoning data (7T tokens) and RLHF for alignment, supporting bilingual and tool integration.

- **SFT Stage Example**: Code-focused fine-tuning.

  - **Template**: {"input": "Code task in English/Chinese", "output": "Solution code"}
  - **Specific Data Example**: Input: "Write a Python function to sort a list." Output: "def sort_list(lst): return sorted(lst)"

- **RLHF Stage Example**: Preference-based refinement.

  - **Template**: {"prompt": "User request", "preferred_response": "Helpful output", "dispreferred": "Poor output"}
  - **Specific Data Example**: Prompt: "Debug this code: print('Hello)". Preferred: "Missing quote: print('Hello')". Dispreferred: "It works fine."

- **Tool Integration Example**: Agentic task with APIs.

  - **Template**: {"query": "Task requiring tool", "tool_call": "API invocation", "response": "Integrated answer"}
  - **Specific Data Example**: Query: "Get stock price for AAPL." Tool_call: "finance_api(symbol='AAPL')". Response: "Current price: $150."

- **Variants**:
  - **Thinking Mode**: For complex reasoning; template: {"prompt": "Reason step-by-step", "mode": "thinking"}. Example: Prompt: "Solve puzzle." Output: "First, analyze pieces... Conclusion: Solution X."
  - **Non-Thinking Mode**: Fast responses; similar template without steps.

#### MiniMax M2 (MiniMax AI)

Uses CISPO for stable RL, with interleaved thinking in outputs wrapped in <think> tags.

- **SFT Stage Example**: Priming for coding/agentics.

  - **Template**: {"instruction": "Task description", "response": "Code or plan"}
  - **Specific Data Example**: Instruction: "Generate a web scraper." Response: "import requests; def scrape(url): ..."

- **RL Stage Example (CISPO)**: Sequence-level optimization.

  - **Template**: {"sequence": "Full output with weights", "reward": "Based on coherence"}
  - **Specific Data Example**: Sequence: "Plan: <think>Check data</think> Execute query." Reward: High for stable gradients.

- **Interleaved Reasoning Example**: Blended thinking and output.

  - **Template**: {"prompt": "Agentic query", "output": "Text with <think> tags"}
  - **Specific Data Example**: Prompt: "Fix bug in code." Output: "<think>Identify error: Null pointer.</think> Patch: Add check. <think>Test.</think> Final code: ..."

- **Variants**: No explicit variants, but agentic mode supports tool calling; template: {"tools": ["browser"], "prompt": "Search task", "output": "<think>Query formulation</think> Result." Example: Tools: ["code_runner"]. Prompt: "Run script." Output: "<think>Validate input</think> Execution output."

For more, see technical reports linked below.

---

As of November 2025, the post-training processes for Kimi K2, GLM-4.6, and MiniMax M2 represent advanced techniques in aligning large Mixture-of-Experts (MoE) models for agentic and coding tasks. These open-weight models, developed by Chinese AI labs, build on pre-training with trillions of tokens to infuse capabilities like tool orchestration, multi-step reasoning, and efficient inference. While proprietary details limit access to exact datasets, technical reports and analyses allow for reconstructing plausible data examples and templates. This survey provides a detailed breakdown, including generated illustrative examples based on documented methods, to highlight how post-training transforms base models into production-ready variants. We cover stages, rationales, specific examples, and templates for each, drawing from primary sources like arXiv papers and model cards.

#### Architectural Context Influencing Post-Training

All models use sparse MoE designs for efficiency: Kimi K2 (1.04T total/32.6B active parameters), GLM-4.6 (355B total/32B active), and MiniMax M2 (230B total/10B active). Post-training addresses sparsity challenges, such as expert silos, by focusing on data diversity and stable optimization. Rationales stem from scaling laws—higher parameters improve performance under fixed compute—but require alignment to prevent instability. Techniques like quantization-aware training (QAT) in Kimi K2 integrate during post-training to enable low-precision deployment (e.g., INT4), reducing memory by ~50% without accuracy loss.

#### Kimi K2 Post-Training: Synthetic Agentics and Verifiable RL

Kimi K2's process, detailed in its arXiv report, emphasizes a multi-stage pipeline to build "open agentic intelligence." SFT establishes instruction-following, while RL uses verifiable rewards (RLVR) and self-critique for generalization. Data synthesis scales agentic interactions, addressing real-world data scarcity through simulation and real sandboxes. Rationale: Synthetic diversity maximizes learning entropy, per information theory, while verifiable signals mitigate reward hacking (Goodhart's law).

- **SFT Details**: Uses Muon optimizer on diverse prompts, including synthetic rephrasings for code/math. Scale: Tens of thousands of high-quality samples via rejection sampling.

  - **Template Example**: JSON-structured for flexibility:
    ```
    {
      "prompt": "[User query with optional context]",
      "response": "[Step-by-step or direct answer, ensuring helpfulness]"
    }
    ```
  - **Specific Data Example (Instruction-Tuning)**:
    - Prompt: "Write a function to reverse a string in Python."
    - Response: "def reverse_string(s): return s[::-1] # Efficient slicing method."

- **RL Details**: Gym-like framework with RLVR for objective tasks (e.g., math with ~50K problems) and self-critique for subjective ones. Includes PTX loss to avoid forgetting, budget controls for token efficiency, and temperature decay for exploration-to-exploitation shift. Objective:
  \[
  L*{RL}(\theta) = \mathbb{E}*{x \sim D} \left[ \frac{1}{K} \sum_{i=1}^{K} \left( r(x, y_i) - \bar{r}(x) - \tau \log \frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)} \right)^2 \right]
  \]

  - **Template Example (Verifiable Reward)**:
    ```
    {
      "task": "[Domain-specific problem, e.g., coding issue]",
      "trajectory": "[Multi-turn steps with tool calls]",
      "reward": "[Score: 1 for success, verified by interpreter/judge]"
    }
    ```
  - **Specific Data Example (Coding Domain)**:
    - Task: "Fix bug in code: def add(a, b): return a - b"
    - Trajectory: "Identify error: Subtraction instead of addition. Patch: Change to a + b. Test: add(2,3)=5."
    - Reward: 1 (unit test pass).

- **Data Synthesis Pipeline**: Three stages—tool specs (3000+ from GitHub + synthetic), agent/task generation with rubrics, trajectory simulation (hybrid simulated/real environments).

  - **Template Example (Agentic Trajectory)**:
    ```
    {
      "tools": ["List of APIs, e.g., weather_api"],
      "user_sim": "[Persona query]",
      "agent_actions": "[Sequence: Think, Tool call, Feedback]",
      "quality_check": "[Rubric score]"
    }
    ```
  - **Specific Data Example**:
    - Tools: ["search_engine", "calculator"].
    - User_sim: "What's 2+2 and today's date?"
    - Agent_actions: "Call calculator(2+2)=4. Call search_engine('current date')=November 20, 2025."
    - Quality_check: Success (rubric: Accurate and complete).

- **Variants**:
  - **Kimi-K2-Instruct**: General chat/agents; example template adds safety filters.
    - Prompt Example: "Summarize article on AI." Response: "Key points: Advances in MoE..."
  - **Kimi-K2-Thinking**: For long-horizon; interleaved CoT without extra compute.
    - Template: {"prompt": "[Complex query]", "think": "[Internal steps]"}.
    - Example: Prompt: "Optimize algorithm." Think: "Analyze time complexity... Improve to O(n)."

Performance: Excels in SWE-Bench (71.3% verified), rationalized by authentic sandboxes.

#### GLM-4.6 Post-Training: Bilingual RLHF and Tool Focus

GLM-4.6 builds on 15T pre-training tokens with 7T SFT for code/reasoning, followed by RLHF via a "slime engine" for preference alignment. Emphasis on bilingual (English/Chinese) data and tool integration during inference. Rationale: RLHF simplifies alignment via Bradley-Terry preferences, efficient for global coding where data scarcity in non-English domains requires targeted curation.

- **SFT Details**: Code-heavy, enhancing reasoning with hybrid modes.

  - **Template Example**:
    ```
    {
      "input": "[Bilingual task, e.g., code in EN/CN]",
      "output": "[Solution with explanations]"
    }
    ```
  - **Specific Data Example**:
    - Input: "编写一个排序列表的 Python 函数 (Write a Python function to sort a list)."
    - Output: "def sort_list(lst): return sorted(lst) # 支持中英注释."

- **RLHF Details**: Aligns with human preferences, including multi-turn coding and adversarial safety.

  - **Template Example**:
    ```
    {
      "prompt": "[Request]",
      "preferred": "[High-quality response]",
      "dispreferred": "[Low-quality, e.g., unsafe]"
    }
    ```
  - **Specific Data Example**:
    - Prompt: "Generate secure password checker."
    - Preferred: "Check length >8, includes symbols."
    - Dispreferred: "Just use 'password123'."

- **Tool Integration**: Supports AWS APIs, search; rationale: Boosts agentics (e.g., 15% lower token usage).

  - **Template Example**:
    ```
    {
      "query": "[Task]",
      "tool_call": "[API params]",
      "integrated_response": "[Final output]"
    }
    ```
  - **Specific Data Example**:
    - Query: "Search for AI news."
    - Tool_call: "search_api(query='AI advancements 2025')".
    - Integrated_response: "Recent: Kimi K2 release."

- **Variants**:
  - **Thinking Mode**: Multi-token lookahead for depth; template: {"mode": "thinking", "prompt": "[Puzzle]"}.
    - Example: Prompt: "Reason through math proof." Output: "Step 1: Assume... Step 2: Derive."
  - **Non-Thinking Mode**: Fast for simple tasks; similar but without lookahead.

Performance: Strong in CC-Bench for coding, attributed to bilingual focus.

#### MiniMax M2 Post-Training: CISPO and Interleaved Efficiency

MiniMax M2 uses CISPO for RL stability, clipping sequence weights to preserve gradients in long outputs. Focus on interleaved thinking (<think> tags) for agentics, at low cost (8% of Claude). Rationale: Variance reduction in off-policy RL enables scalable training, per Meta's scaling insights; interleaving reduces latency in workflows.

- **SFT Details**: Primes for coding/agents with CoT patterns.

  - **Template Example**:
    ```
    {
      "instruction": "[Task]",
      "response": "[Interleaved output]"
    }
    ```
  - **Specific Data Example**:
    - Instruction: "Build a simple API."
    - Response: "<think>Define endpoints</think> Code: from flask import Flask..."

- **RL Details (CISPO)**: Context-aware sampling for policy optimization, trading off some domains for agentics.

  - **Template Example**:
    ```
    {
      "sequence": "[Full trajectory with weights]",
      "optimization": "[Clipped reward]"
    }
    ```
  - **Specific Data Example**:
    - Sequence: "<think>Plan query</think> Execute tool. <think>Verify</think>."
    - Optimization: High weight on coherent sequences.

- **Interleaved Reasoning**: Blends thinking/output for responsiveness.

  - **Template Example**:
    ```
    {
      "prompt": "[Agentic query]",
      "output": "[Text with <think>...</think> tags]"
    }
    ```
  - **Specific Data Example**:
    - Prompt: "Automate report generation."
    - Output: "<think>Fetch data via API</think> Process: Aggregate stats. <think>Format</think> Final report."

- **Variants**: Primarily agentic mode with tool calling (e.g., shell/browser).
  - **Tool-Calling Template**: {"tools": ["code_runner"], "prompt": "[Run task]"}.
  - Example: Prompt: "Test code." Output: "<think>Safety check</think> Runner output: Success."

Performance: 69.4% on SWE-Bench, due to efficiency in sparse activation.

#### Comparative Table: Post-Training Components

| Component          | Kimi K2                           | GLM-4.6                      | MiniMax M2               |
| ------------------ | --------------------------------- | ---------------------------- | ------------------------ |
| **SFT Data Scale** | Diverse prompts + synthetic tools | 7T code/reasoning, bilingual | CoT-heavy for agents     |
| **RL Method**      | RLVR + Self-Critique              | RLHF with preferences        | CISPO for stability      |
| **Key Template**   | Trajectory with rewards           | Preference pairs             | Interleaved <think> tags |
| **Variants Focus** | Instruct/Thinking for agents      | Thinking/Non-Thinking modes  | Agentic with tools       |
| **Rationale Edge** | Verifiable grounding              | Bilingual efficiency         | Gradient preservation    |
| **Example Domain** | Tool-use simulations              | Coding debugging             | Report automation        |

This table highlights trade-offs: Kimi's robustness vs. GLM's accessibility vs. MiniMax's speed.

In summary, these processes democratize agentic AI, with generated examples illustrating how data templates drive alignment. Future releases may reveal more proprietary details.

#### Key Citations

- [arXiv Technical Report on Kimi K2](https://arxiv.org/abs/2507.20534)
- [IntuitionLabs on GLM-4.6](https://intuitionlabs.ai/articles/glm-4-6-open-source-coding-model)
- [Z.AI Docs on GLM-4.6](https://docs.z.ai/guides/llm/glm-4.6)
- [Perficient Blog on MiniMax M2](https://blogs.perficient.com/2025/11/19/minimax-m2-open-source-interleaved-reasoning-model/)
- [Hugging Face Model Card for MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [Medium Analysis on MiniMax M2](https://medium.com/@LakshmiNarayana_U/minimax-m2-the-open-weight-ai-model-rewriting-the-rules-for-coding-and-agentic-workflows-c61cc16c5bc3)
