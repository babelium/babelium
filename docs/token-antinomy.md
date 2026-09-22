# The Antinomy of the Token

## Prologue: A Conversation Cut Off Mid-Word

Suppose you are using a code completion tool. You are halfway through typing, and the cursor sits in the middle of a word — say you have just typed `the reader is intri`, without finishing `intrigued`, and the tool has to continue from there.

Normally, if you hand the complete sentence `the reader is intrigued` to a language model, the model first cuts it into a string of word fragments (the technical term is *token*; what that actually is will be made precise below). The word `intrigued` will most likely be cut into two pieces such as `intrig` + `ued`, or perhaps stay whole — which cut you get depends on how often the word appears in the training data and on the tokenizer's merge rules.

But right now you have typed only `intri`. The tool cannot foresee that you are about to type `gued`; it can only tokenize the five letters it has. And the cut that a tokenizer gives for `intri` alone is very likely **not the same cut** as the prefix of the cut it gives for the complete word. If the complete word becomes `intrig` + `ued`, then the bare `intri`, unable to assemble the fragment `intrig`, may come out as something entirely different, like `int` + `ri`.

What follows from that? During training, the language model has only ever seen token sequences produced by cutting complete words the way complete words get cut. It has never seen `int` + `ri` appear in the position occupied by `intrigued`. So when you feed it this unusual cut, it reads the current position as one where a word has just ended and a new one should begin — it will very likely insert a space and start a fresh word, rather than finishing `intrigued`.

This is not a hypothetical anecdote. It is a real engineering problem with a real name: **token healing**. The standard fix is to throw away the last one or two fragments before the cursor and regenerate that stretch under a more conservative, byte-constrained scheme, so the model can "heal" and pick up what should have been there.

The patch itself is not hard; it is routine engineering by now. But behind it sits a question that has never been asked all the way down. A tokenizer cuts text into a string of discrete pieces, and the model interprets those pieces back into text or continues from them — over that round trip, what guarantees that "cut apart and reassembled" and "left as it was" are the same thing? And further: if that guarantee can fail in the most ordinary situation imaginable — typing half a sentence — then what kind of object is a token, such that its behavior is this unstable?

What this article describes is the process of digging into that question. Dig far enough and you find that the claim almost everyone would unreflectively accept — *a token is a piece that text or a signal has been cut into* — describes only half of what a token is.

## Chapter 1: The Assumption Nobody Ever Stated

Nearly every discussion of tokens, whether about language model tokenizers or about how image models cut a picture into patches, carries the same subtext: **a token is a slice of the signal**. Text is cut into substrings, an image into a grid of pixel blocks, audio into frames. The claim is so natural that almost nobody treats it as a hypothesis in need of testing.

But move your gaze away from text tokenization and look at how other fields *manufacture* their own tokens, and the slicing story starts to run out.

In vector quantization (VQ, covered in detail in a later case study), the situation is nearly the reverse. The model first computes a continuous, high-dimensional representation, then finds the nearest entry in a codebook prepared in advance and takes that entry as the token for this piece of content. That is not "cutting the signal apart"; it is **snapping** the signal onto an anchor fixed beforehand. Slicing goes from whole to part; snapping goes from continuous to discrete. The direction and the mechanism differ.

So step back and split the word *token* into the two capacities it actually carries, rather than rushing to define it.

One capacity is **identity**: a token has to be countable, retrievable by index, computable once and storable for reuse, recognizable as "the same one" across different occasions, and composable like a building block into longer sequences. These requirements sound abstract but are concrete in practice — token number 5 in a vocabulary is still number 5 whether it appears at the start of one sentence or the middle of another. That is identity at work. It is exactly this property that lets a language model use a KV cache: compute the earlier content once and reuse it, instead of recomputing every time, because the address "number 5" is itself stable and does not vary with context.

The other capacity is **similarity**: being able to compare how alike two things are, to interpolate between two points, and, on encountering an unseen example, to infer how to handle it from what it resembles. This demands a continuous space that can be differentiated and moved around in — the high-dimensional space where embedding vectors live.

These two capacities correspond to two entirely different mathematical structures, discrete and continuous. And the sentence "a token is a slice of the signal" emphasizes only the identity half — every piece cut out can be counted and numbered. It says nothing about the similarity half: whether two tokens can be compared, interpolated, judged alike.

So picture a token as a two-faced object: one face turned toward *countable, addressable, cacheable, composable*, the other toward *comparable, interpolable, generalizable to unseen examples*. The two faces look unrelated — one is set-theoretic and discrete, the other geometric and continuous — yet every time a language model runs, both are in use at once. The input text is first cut into discrete tokens (identity face), each token is translated into a continuous embedding vector for computation (similarity face), and the result must be translated back into discrete tokens to become output text (identity face again).

The **conversion** between the two faces is the thing actually worth watching. And that conversion necessarily contains one decisive act — picking a discrete item out of the continuous, or placing a discrete item into the continuous. That act will be given a name of its own later: **commitment**. Remember the word; nearly every case study below turns around it.

## Chapter 2: Five Case Studies, Five Ways of Getting Back

The abstractions are in place; now for actual examples. The five cases below come from five entirely different model families, but all of them are doing the same thing: converting back and forth between the continuous and the discrete. Working through them shows how the same conversion grows completely different symptoms in different places.

### Case A: Autoregressive Text Generation — The Perpetually Misread First Word

Autoregressive language models (AR from here on — models that emit one token after another) are the example most often reached for in any discussion of tokens, and precisely because they are so familiar, some genuinely structural facts about them tend to get waved past.

Start with a phenomenon everyone has noticed but whose cause has never been stated clearly: **why does the first generated token always look slightly odd, almost like a placeholder?**

To answer that, a word about what a language model looks like inside. For every new token, the model runs the whole text written so far through an **attention** mechanism — roughly, it lets the current position look at every preceding position and decide which to weigh and which to ignore, the way a speaker glances over everyone else's notes before contributing. And because generation proceeds left to right, when writing token *t* the model may look only at positions before *t*, never ahead. That rule is the **causal mask**: causes may be seen, effects may not.

The causal mask has an immediate corollary: the first position in a sequence, whenever you look at it, can see nothing but itself. This is not a strategy the model learned; it is an arithmetic consequence of a hard rule. And because that position can only ever see itself, the attention weights it produces follow a very fixed pattern that does not vary with content, and the layers above shape it into a special **anchor**. Many papers call this an **attention sink**: other positions, when computing attention, pour inexplicably large amounts of weight into this first position, as if information were draining into a sink.

One popular explanation has been that this position performs *semantic* placeholding, or marks some boundary of high certainty where no work is needed. Following the thread down gives a more accurate account: **the position's specialness is, first of all, purely structural.** The causal mask alone, with no semantics involved, already locks the first position into an island that sees no one and only itself. That the model later learns to use it as a sink is a natural consequence of this structural property, not evidence that some semantic judgment came first and produced the specialness.

That conclusion is not itself new — the attention-sink literature has clean circuit-level evidence for it. But it raises a deeper question. Slots forced into existence by structure appear widely across models: the **register** slots in vision models discussed later, `[MASK]` in masked language models, `padding` at the end of a sequence. These slots share one feature: nothing in the original input signal corresponds to them. No character in a passage of text corresponds to `[MASK]`; no patch of pixels in an image corresponds to a `register`. Such slots are bolted on, not grown out of the signal. Case E returns to this.

So much for the first token. The more dramatic part of this case is a self-refutation.

An early stage of the theory judged AR to be a structural **outlier**. In text generation, the model first scores every candidate in the vocabulary (those scores, stacked together, live in a space whose dimension equals the vocabulary size — tens of thousands of dimensions if the vocabulary has tens of thousands of entries), then selects the highest-scoring token and converts it into a vector of a few hundred or few thousand dimensions (the *embedding space*, where the model's actual computation happens), which is passed to the next step. At first glance the scoring space (tens of thousands of dimensions) and the space fed forward (a few hundred) have different dimensions, so this conversion would seem to require a bridge between two entirely different spaces — whereas in vector quantization, scoring and feeding forward happen in the same space. Hence the early judgment: AR is the only placement that must cross between two different spaces; the others need no crossing.

That judgment turned out to be wrong, and the manner of the error is worth recounting, because the way it went wrong points at an easily missed trap.

The mistake was treating the **score table** as the space where the carrier lives. The model scores each candidate by taking the inner product of the current hidden state with a vector associated with each vocabulary entry — the inner product being the sum of elementwise products, larger when the two vectors point in more nearly the same direction. The current hidden state and the embedding vector of the selected token **have always lived in the same few-hundred-dimensional space**; they were never apart. The "tens of thousands of scores" are just a table of numbers produced by this inner product, not the place where the hidden state or the selected vector *resides* — much as ranking a room of people by height gives you a table with a hundred ranks, but each person's height is a single number, not an element of "rank space." The ranking table and the height are not the same kind of object.

Once that is clear, AR is doing the **same operation** as the vector quantization model below: take a continuous vector, compare it against a batch of anchor vectors, select the closest, hand that anchor onward. The only difference is the comparison — AR uses an inner product, VQ uses Euclidean distance — and those two agree on the ranking only when all anchor vectors have equal norm. That is the one residual difference, a purely numerical detail, not a gulf between spaces.

**The lesson left by this self-correction: AR went from "the one structural outlier in the whole theory" to an ordinary instance isomorphic to every other placement.** And the reason "AR is an outlier" seemed plausible at first is precisely that AR is the example nearly everyone meets first — an atypical case that became the default mental picture behind the word *token*. That may partly explain why discussion of tokenization and text generation has stayed so long on empirical analogies about *compression* and *boundaries*: everyone has been arguing over the interface hardest to compare directly, while the other interfaces, where the gap is far easier to measure, went unexamined.
