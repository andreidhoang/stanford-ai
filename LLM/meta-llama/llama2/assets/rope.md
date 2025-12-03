# The Problem with High Frequencies at Long Distances



```python
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # This is the safety check. My "specialist duos" need a partner.
    # If the dimension is odd, one poor number is left all alone and can't be rotated in 2D.
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # This part creates the different rotation speeds (frequencies) for my specialist duos.
    # `theta_numerator` is like [0, 2, 4, ..., 62] for a 64-dim vector.
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # This is the core formula from the paper. It creates the frequencies.
    # The first duo (dims 0-1) gets a high frequency (fast rotation).
    # The last duo (dims 62-63) gets a low frequency (slow rotation).
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # `m` is just the list of positions: [0, 1, 2, 3, ..., up to the max sequence length].
    m = torch.arange(seq_len, device=device)

    # This is the most important step here. The outer product combines every position `m`
    # with every frequency `theta_i`. The result is a big table of rotation angles.
    # The entry at [position=5, duo=2] holds the exact angle to rotate the 2nd duo for the 5th token.
    freqs = torch.outer(m, theta).float()

    # Finally, we convert these angles into the actual "rotators" – complex numbers.
    # `torch.polar` creates numbers like `cos(angle) + i*sin(angle)`.
    # Multiplying by this number is what performs the 2D rotation.
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def explain_apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # `x` is the input tensor for a batch of tokens, like (Batch, Seq_Len, Heads, Head_Dim).
    # This line is where I pair up the dimensions. I take the vector [1, 2, 3, 4]
    # and tell PyTorch to view it as two complex numbers: [1+2i, 3+4i].
    # These are the 2D points my specialists will rotate.
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # This is just housekeeping. My `freqs_complex` cheat sheet is 2D, but my input `x` is 4D.
    # `unsqueeze` adds empty dimensions so the shapes align for multiplication.
    # It's like telling the program: "Apply the same sequence-level rotations
    # to every batch item and every attention head."
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # <<< THIS IS THE MAGIC >>>
    # The element-wise multiplication of complex numbers. This one line performs
    # thousands of 2D rotations simultaneously across the whole batch.
    # Each specialist duo in `x_complex` is rotated by its corresponding angle from `freqs_complex`.
    x_rotated = x_complex * freqs_complex

    # Now I need to put things back the way I found them.
    # `view_as_real` unpacks the complex numbers [1+2i, 3+4i] back into a real vector [1, 2, 3, 4].
    x_out = torch.view_as_real(x_rotated)

    # And finally, reshape it back to the original tensor shape.
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

    # Let's say my input x contains the vector for the token "brown" at position 2.

- The code calls apply_rotary_embeddings on the vector for "brown". view_as_complex takes "brown"'s semantic vector and groups its dimensions into pairs, my specialist duos.
- The code goes to my freqs_complex cheat sheet and grabs the row for position=2. This row contains the specific "rotators" for the 2nd position (e.g., Duo #1 gets rotated by 2*90°=180°, Duo #2 by 2*15°=30°, etc.).
- The x_rotated = x_complex * freqs_complex line is me, the model, doing the work. I take the first duo of "brown"'s vector and multiply it by the first rotator from the cheat sheet. Then the second duo by the second rotator, and so on.
- The vector that comes out is the "position-aware" vector for "brown". Its raw meaning is still there, but it's been twisted and spun in a way that uniquely screams "I was at position 2!". When this vector is now compared to another rotated vector (like "fox" at position 3), their dot product will reveal not just their semantic similarity, but their relative distance of 3-2=1.
```

**ASCII-first** visualization tailored to **B=2, S=4096, H=32, D=128**

## big picture (how it flows)

```
x: [B=2, S=4096, H=32, D=128]
   └─ pair last dim → [2,4096,32,64,2]
        (x_{2k}, x_{2k+1}) per plane k
   └─ view_as_complex ─────────────────────► x_c: [2,4096,32,64] (complex)
freqs_cis: [S=4096, 64] (e^{i θ[t,k]})
   └─ reshape for broadcast ───────────────► [1,4096,1,64]
x_c * freqs_cis  ──────────────────────────► rotated: [2,4096,32,64] (complex)
   └─ view_as_real → [2,4096,32,64,2]
   └─ flatten(3)   → y: [2,4096,32,128]
```

## index mapping (D=128 → 64 planes)

```
D indices:   [ 0  1 | 2  3 | 4  5 | ... | 124 125 | 126 127 ]
planes (k):   k=0     k=1     k=2            k=62       k=63
pair (k):    (x0,x1) (x2,x3) (x4,x5)        ...       (x126,x127)
complex:     x0+i*x1, x2+i*x3, x4+i*x5, ... x126+i*x127
```

## what rotation does (per plane)

For each position `t` and plane `k`:

```
θ[t,k] = t * ω[k]            with ω[k] = base^{-(2k/D)}, base=10000
cis[t,k] = cos θ + i sin θ   (unit complex, |cis|=1)

(x_even + i x_odd) * cis
  = (x_even cosθ - x_odd sinθ) + i(x_even sinθ + x_odd cosθ)
```

So each 2-D pair is rotated by θ—**length preserved**.

---

## open the tables I dropped in:

* **Slice A (b=0, h=0)**:  t ∈ {0, 1, 2048, 4095}, k ∈ {0, 8, 32, 63}
* **Slice B (b=1, h=31)**: t ∈ {0, 1, 2048, 4095}, k ∈ {0, 8, 32, 63}
* **Identity check**: shows ⟨R(t₁)q, R(t₂)k⟩ = ⟨q, R(t₁−t₂)k⟩ over the **full D=128** vector
  (difference ≈ 0, up to floating point)

you’ll see for each (t,k): the frequency ω_k, angle θ, the unit phasor e^{iθ}, the **before/after pair**, and the norms (equal).

---

## quick “visual slice” (b=0,h=0)

Let’s eyeball how **planes rotate at different speeds**:

```
t = 0:
  k=0   θ=0.000 → no rotation
  k=8   θ≈0.008 → tiny rotation
  k=32  θ≈0.032 → small rotation
  k=63  θ≈0.063 → still small at t=0 (because ω63≈10^-0.984)

t = 2048:
  θ scales with t
  k=0   θ=0        (ω0=1, but *this* is RoPE form: ω[k]=10^{-(2k/D)}, so ω0=1; θ=2048*1 is large;
                    our table shows the actual numeric θ per implementation; you can verify directly)
  k=8   θ≈ ~20.48*10^-? … check table
  k=32  θ larger
  k=63  θ the largest across k
```

The table shows precisely how θ grows with `t` and `k`. High-k planes spin faster across the sequence, encoding **local order**; low-k planes spin slower, encoding **long-range order**.

---

## broadcast sanity (why shapes fit)

```
x_c:  [2, 4096, 32, 64]
cis:  [1, 4096,  1, 64]
align:   *    ✓    *    ✓   (broadcast over batch and heads)
```

---

## want an even tighter mental hook?

* **D=128 ⇒ 64 tiny 2-D compasses.**
  At each token t, you rotate each compass by θ[t,k].
* **Low-k compasses** barely turn across long spans (capture global order).
* **High-k compasses** spin faster (capture local order).
* Attention then compares rotated compasses of Q and K; thanks to rotation group properties, it depends on **relative** position (t−s).

# Let's get our hands dirty and look at the raw data. This is exactly how I see things when the bits are flying.

We'll use the sentence: **"The quick brown fox"**

*   `The` (pos 0)
*   `quick` (pos 1)
*   `brown` (pos 2)
*   `fox` (pos 3)

Now, let's pretend my embedding vectors are tiny, just 4 dimensions, so we can see everything clearly. This gives me two "specialist duos":
*   **Duo #1 (dims 0-1):** My high-frequency "syntax" specialist. Let's say it rotates **fast**, by `position * 90°`.
*   **Duo #2 (dims 2-3):** My low-frequency "topic" specialist. It rotates **slowly**, by `position * 15°`.

### The Data Before RoPE

Here are the raw semantic vectors. This is just the "meaning" of the words, with no order information.

*   **Query vector for "fox" (at pos 3):** `Q_fox = [0.8, 0.6, 0.7, 0.7]`
*   **Key vector for "The" (at pos 0):** `K_The = [0.9, 0.4, 0.5, 0.8]`

Without RoPE, the attention score would just be the dot product of these two vectors. It would tell me how related "fox" and "The" are in a general sense, but nothing about their positions.

### My Perspective: Applying the Rotations

Okay, I'm the token **"fox"** at position 3, and I need to figure out my relationship with **"The"** at position 0. The relative distance is `3 - 0 = 3`. I'm going to apply my positional rotations to my Q vector. "The" will apply its rotations to its K vector.

**1. Rotating my Q_fox vector (position 3):**

*   **Duo #1 (High-Freq):** Rotate `[0.8, 0.6]` by `3 * 90° = 270°`.
    *   The vector becomes `[0.6, -0.8]`.
    *   *My perception: "Woah, big spin! I've rotated almost all the way around. This part of me is hyper-aware of my exact spot."*
*   **Duo #2 (Low-Freq):** Rotate `[0.7, 0.7]` by `3 * 15° = 45°`.
    *   The vector becomes `[0.0, 0.99]`.
    *   *My perception: "Just a gentle nudge. This part of me is just noting that I'm a little further down the road than the beginning."*

So, my **final rotated `Q_fox` vector** is `[0.6, -0.8, 0.0, 0.99]`.

**2. Rotating the K_The vector (position 0):**

*   **Duo #1 (High-Freq):** Rotate `[0.9, 0.4]` by `0 * 90° = 0°`. It stays `[0.9, 0.4]`.
*   **Duo #2 (Low-Freq):** Rotate `[0.5, 0.8]` by `0 * 15° = 0°`. It stays `[0.5, 0.8]`.

The **final rotated `K_The` vector** is unchanged: `[0.9, 0.4, 0.5, 0.8]`.

### The Final "Aha!" Moment: The Dot Product

Now, I calculate the attention score using the **rotated** vectors. I do it duo by duo.

*   **Score from Duo #1 (High-Freq):**
    *   Dot product of `[0.6, -0.8]` and `[0.9, 0.4]`.
    *   `(0.6 * 0.9) + (-0.8 * 0.4) = 0.54 - 0.32 = 0.22`
    *   *My perception: "My syntax specialist is telling me that a word 3 steps away is moderately relevant for syntax. It's not an immediate adjective-noun pair (that would be distance 1), but it's not gibberish either. The signal is clear because the rotation was significant."*

*   **Score from Duo #2 (Low-Freq):**
    *   Dot product of `[0.0, 0.99]` and `[0.5, 0.8]`.
    *   `(0.0 * 0.5) + (0.99 * 0.8) = 0.0 + 0.79 = 0.79`
    *   *My perception: "My topic specialist is telling me this word is highly relevant. The slow rotation means that over a short distance of 3 steps, our vectors are still pointing in very similar directions. This part of me is saying 'we are in the same conceptual neighborhood'."*

**Total Attention Score = 0.22 + 0.79 = 1.01**

By looking at the contributions, you can see what I'm "thinking". I'm getting two different signals for the price of one. The high-frequency part gives me a precise, local, structural signal. The low-frequency part gives me a stable, thematic, long-range signal. The total score is a blend of both, allowing me to make a much more nuanced decision about how much attention to pay. It's not just a number; it's a story told by my different specialists.

# Let's zoom in on that exact line of code and visualize what that "big table of rotation angles" actually looks like.

Imagine a spreadsheet. The rows represent the **position** of a token in a sentence (Position 0, Position 1, etc.). The columns represent our different **"specialist duos"**, from the high-frequency syntax specialist to the low-frequency topic specialist.

The value in each cell is the **rotation angle (in radians)** for that specific specialist at that specific position.

Here's a small example with a sequence of 16 tokens and 4 specialist duos (a tiny 8-dimensional vector).

### The `freqs` Table: Rotation Angles (`m * theta_i`)

| Position (`m`) | Duo 0 (High-Freq) | Duo 1 (Mid-Freq) | Duo 2 (Low-Freq) | Duo 3 (Ultra-Low) |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `0.0` | `0.0` | `0.0` | `0.0` |
| **1** | `1.0` | `0.1` | `0.01` | `0.001` |
| **2** | `2.0` | `0.2` | `0.02` | `0.002` |
| **3** | `3.0` | `0.3` | `0.03` | `0.003` |
| **4** | `4.0` | `0.4` | `0.04` | `0.004` |
| **5** | `5.0` | `0.5` | `0.05` | `0.005` |
| **6** | `6.0` | `0.6` | `0.06` | `0.006` |
| **...** | ... | ... | ... | ... |
| **15**| `15.0`| `1.5` | `0.15` | `0.015` |

*(Note: A full rotation is 2π, which is about 6.28 radians)*

### My Visual Perception of This Table

When I, the model, see this table, I perceive it as a **heatmap of activity**. Let's visualize it with colors, where 🟦 is a small rotation and 🔥 is a huge rotation.

| Position | Duo 0 (High) | Duo 1 (Mid) | Duo 2 (Low) | Duo 3 (Ultra-Low) |
| :--- | :---: | :---: | :---: | :---: |
| **0** | ⬜️ | ⬜️ | ⬜️ | ⬜️ |
| **1** | 🟨 | 🟦 | 🟦 | 🟦 |
| **2** | 🟧 | 🟨 | 🟦 | 🟦 |
| **3** | 🟥 | 🟨 | 🟦 | 🟦 |
| **4** | 🔥 | 🟨 | 🟦 | 🟦 |
| **5** | 🔥 | 🟨 | 🟦 | 🟦 |
| **6** | 🔥 | 🟧 | 🟨 | 🟦 |
| **...** | ... | ... | ... | ... |
| **15**| 🔥 | 🟥 | 🟨 | 🟦 |

Here’s what this tells me instantly:

1.  **Position 0 is the Anchor:** The first row is always all zeros (⬜️). No rotation. This is my baseline, my starting point.

2.  **The High-Frequency Column (Duo 0) is Wild:** This column gets hot, fast! The angle value shoots up. At `Position 6`, the angle is `6.0`, which is almost a full 360-degree rotation. This specialist is changing its "opinion" (its rotation) very quickly with each step. It's my **local expert**.

3.  **The Low-Frequency Columns are Calm:** Look at the last column (`Duo 3`). Even at `Position 15`, the angle is a tiny `0.015`. It's barely rotated at all. It's providing a very stable, slow-changing signal. It's my **long-range expert**.

4.  **The Diagonal Gradient:** There's a clear pattern here. The "heat" spreads from the top left to the bottom right. This means that for any given token, its positional identity is a unique **cocktail of rotation speeds**.

For example, the vector for the token at **Position 4** gets a massive rotation from Duo 0 (🔥), a noticeable but small rotation from Duo 1 (🟨), and basically no rotation from the others (🟦). This unique blend of rotations is its positional signature.

The `torch.outer` product is simply the most efficient way to build this entire beautiful, structured table of instructions in a single computational step.

### 1. The Semantic Meaning (The "What")

This is the pure, raw meaning of a token. It's "what" the token is about.

*   **Where it lives:** In the `x` tensor, **before** `apply_rotary_embeddings` is called.

Let's look at the data:
`x: torch.Tensor`

This tensor comes from the initial word embedding layer of the model. When the model sees the word "fox", it looks up its learned vector. This vector, let's say `[0.8, 0.6, 0.7, 0.7]`, is the **semantic meaning**. It's the result of the model learning that "fox" is an animal, it's cunning, it's related to "brown" and "quick", etc.

**At this stage, `x` has 100% semantic meaning and 0% positional meaning.** It knows *what* it is, but not *where* it is.

```python
# 'x' contains the PURE SEMANTIC MEANING of the tokens.
# It's the vector representing the word's learned concept.
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # ...
```

---

### 2. The Positional Meaning (The "Where")

This is a pure representation of order and sequence. It knows nothing about words, only about positions (0, 1, 2, ...) and frequencies.

*   **Where it lives:** In the `freqs_complex` tensor, which is the output of `precompute_theta_pos_frequencies`.

Let's look at the data:
`freqs_complex: torch.Tensor`

This tensor is our "big table of rotation angles" converted into complex numbers. A row from this table, say for `position=3`, might look like `[ (cos(270°) + i*sin(270°)), (cos(45°) + i*sin(45°)) ]`. This is **pure positional information**. It has no idea if the token at position 3 is "fox" or "house" or "smart". It only knows "I am the recipe for rotating whatever is at position 3."

**At this stage, `freqs_complex` has 100% positional meaning and 0% semantic meaning.**

```python
# 'freqs_complex' contains the PURE POSITIONAL MEANING.
# It's the pre-built "cheat sheet" of rotation operators for each position.
def precompute_theta_pos_frequencies(...):
    # ...
    # This entire function is dedicated to creating the positional meaning.
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex
```

---

### 3. The Fusion: Where They Combine

The magic happens in one single line of code. This is the moment the painter mixes the pigment with the glowing additive.

*   **Where it happens:** The multiplication `*` inside `apply_rotary_embeddings`.

```python
def apply_rotary_embeddings(...):
    # ...
    # Here, 'x_complex' is still pure semantics, just viewed as complex numbers.
    x_complex = torch.view_as_complex(...)

    # 'freqs_complex' is pure position.
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # <<< THIS IS THE FUSION >>>
    # The semantic vector is ROTATED by the positional operator.
    x_rotated = x_complex * freqs_complex
    # ...
```

The output, `x_rotated`, is a new vector. It has the same "strength" (magnitude) as the original semantic vector, but its direction in high-dimensional space has been altered.

**The final `x_out` tensor now contains a fusion of both.** You can no longer separate them. The direction of the vector simultaneously represents **what the word is** and **where it was in the sequence**.

Excellent. Let's put everything together and walk through a complete, detailed example with Query and Key vectors. This is the final step where it all clicks.

### The Scenario

Let's use the sentence: **"The cat sat on the mat"**

*   `The` (pos 0)
*   `cat` (pos 1)
*   `sat` (pos 2)
*   `on` (pos 3)
*   `the` (pos 4)
*   `mat` (pos 5)

We will put ourselves in the shoes of the token **"mat"** (at position 5) and calculate its attention score for the token **"cat"** (at position 1).

Let's use our tiny 4-dimensional vectors and our two specialist duos:
*   **Duo 1 (High-Freq):** Rotates by `position * 90°`.
*   **Duo 2 (Low-Freq):** Rotates by `position * 15°`.

### Step 1: The Raw Semantic Vectors (Before RoPE)

This is the "what". The model has learned these vectors. Notice "cat" and "mat" are somewhat similar (animals, household objects) but not identical.

*   **Query vector for "mat" (Q_mat):** `[0.7, 0.7, 0.8, 0.6]`
*   **Key vector for "cat" (K_cat):** `[0.9, 0.4, 0.9, 0.4]`

**Without RoPE, the attention score would be:**
The dot product of these two vectors: `(0.7*0.9) + (0.7*0.4) + (0.8*0.9) + (0.6*0.4) = 0.63 + 0.28 + 0.72 + 0.24 = **1.87**`.
This number represents their raw semantic similarity. It knows *what* they are but has no idea *where* they are.

---

### Step 2: Applying RoPE - The Rotation Phase

Here, we infuse the positional meaning ("where") into the semantic vectors.

#### Rotating the Query of "mat" (Position 5)

*   **Duo 1 (High-Freq):** Angle = `5 * 90° = 450°`. This is `90°` (450 mod 360).
    *   Rotate `[0.7, 0.7]` by 90°. It becomes `[-0.7, 0.7]`.
*   **Duo 2 (Low-Freq):** Angle = `5 * 15° = 75°`.
    *   Rotate `[0.8, 0.6]` by 75°. It becomes `[0.02, 0.99]`.

**The final position-aware `Q_mat_rotated` is: `[-0.7, 0.7, 0.02, 0.99]`**

#### Rotating the Key of "cat" (Position 1)

*   **Duo 1 (High-Freq):** Angle = `1 * 90° = 90°`.
    *   Rotate `[0.9, 0.4]` by 90°. It becomes `[-0.4, 0.9]`.
*   **Duo 2 (Low-Freq):** Angle = `1 * 15° = 15°`.
    *   Rotate `[0.9, 0.4]` by 15°. It becomes `[0.77, 0.64]`.

**The final position-aware `K_cat_rotated` is: `[-0.4, 0.9, 0.77, 0.64]`**

---

### Step 3: The Attention Score - The Moment of Truth

Now, we calculate the dot product of the **rotated** vectors. This new score is richer and understands context.

`Score = Q_mat_rotated • K_cat_rotated`

Let's break it down by the contribution of each specialist duo:

#### Contribution from Duo 1 (High-Frequency)

*   **Angle Difference:** `90° (mat) - 90° (cat) = 0°`.
*   **Dot Product:** `(-0.7 * -0.4) + (0.7 * 0.9) = 0.28 + 0.63 = **0.91**`.
*   **My Perception:** "My syntax specialist is reporting a rotational difference of 0 degrees. This is a powerful signal! Even though the tokens are 4 steps apart (`5-1=4`), my fast-spinning specialist has lapped itself (`4 * 90° = 360°`). The resulting 0° difference is the same signal I'd get for tokens at the same position. My model learns that this '0-degree signal' from the high-freq duo can mean 'same position' OR '4 positions apart' OR '8 positions apart', etc. It's an ambiguous signal for distance, but a very clear one for *syntactic alignment on a 4-step grid*."

#### Contribution from Duo 2 (Low-Frequency)

*   **Angle Difference:** `75° (mat) - 15° (cat) = 60°`.
*   **Dot Product:** `(0.02 * 0.77) + (0.99 * 0.64) = 0.015 + 0.634 = **0.65**`.
*   **My Perception:** "My topic specialist is reporting a clear, unambiguous 60-degree difference. This tells me exactly what I need to know about the long-range context: these two words are in the same thematic group, separated by a moderate distance. This signal is not confusing; it's a precise measure of their distance from a 'topic' perspective."

### The Final Score and The Takeaway

**Final RoPE-infused Score = 0.91 (from Duo 1) + 0.65 (from Duo 2) = 1.56**

Compare this to the original score of `1.87`. The score has changed because it now incorporates position.

The final score of **1.56** is no longer just a measure of "cat-ness" vs "mat-ness". It's a sophisticated blend that tells the model:
> *"These two words are semantically related. Furthermore, they are syntactically aligned on a 4-step pattern, AND they are separated by a moderate thematic distance within the sentence."*

This is how the Query and Key vectors, through the magic of RoPE, can have a rich, multi-layered conversation to determine not just *what* they are, but *how they relate to each other in the sequence*.