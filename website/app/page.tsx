export default function ResearchPaperPage() {
  return (
    <div className="min-h-screen bg-white">
      <article className="mx-auto max-w-[750px] px-8 py-16">
        {/* Title - Larger, bolder title with better spacing */}
        <h1 className="text-5xl font-bold text-black mb-6 leading-tight text-balance">
          LLINK: Cross-Lingual Alignment via Encoder Injection
        </h1>

        {/* Date - Smaller, gray text */}
        <p className="text-sm text-gray-500 mb-6 font-sans">10.27.2025</p>

        {/* Links */}
        <div className="mb-12 flex gap-4">
          <a href="https://arxiv.org/abs/2510.17530" className="text-black">
            arXiv
          </a>
          <a href="https://github.com/rajansagarwal/llink" className="text-black">
            GitHub
          </a>
        </div>

        {/* Authors - Clean author list with correspondence */}
        <div className="mb-12 text-base">
          <p className="text-gray-800 mb-1">Rajan Agarwal*, Aarush Gupta*</p>
          <p className="text-sm text-gray-500 italic font-sans">
            * Core Contributor; Correspondence to r34agarw@uwaterloo.ca, hiarrushgupta@gmail.com
          </p>
        </div>

        <div className="bg-[#f8f7f5] px-8 py-6 mb-12 rounded-sm">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-4 font-sans">
            Abstract
          </h2>
          <p className="text-[17px] leading-relaxed text-gray-800">
            Instruction-tuned large language models underperform on low-resource, non-Latin scripts due to tokenizer
            fragmentation and weak cross-lingual coupling. We present LLINK (Large Language Injection for Non-English
            Knowledge), a compute-efficient language-as-modality method that conditions an instruction-tuned decoder
            without changing the tokenizer or retraining the decoder. First, we align sentence embeddings from a frozen
            multilingual encoder to the decoder&apos;s latent embedding space at a reserved position via a lightweight
            contrastive projector. Second, the vector is expanded into K soft slots and trained with minimal adapters so
            the frozen decoder consumes the signal. LLINK substantially improves bilingual retrieval and achieves 84
            percent preference over the base model and 64 percent over direct finetuning in LLM-judged question and
            answer evaluations. We further find that improvements can be attributed to reduced tokenization inflation and
            a stronger cross-lingual alignment, despite the model having residual weaknesses in numeric fidelity.
            Treating low-resource languages as a modality offers a practical path to stronger cross-lingual alignment in
            lightweight LLMs.
          </p>
        </div>

        <h2 className="text-2xl font-semibold text-black mb-3 mt-12">Summary</h2>
        <ul className="space-y-3 text-[17px] leading-relaxed text-gray-800 mb-8">
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>
              LLINK casts low-resource languages as a modality: Khmer sentences are encoded once by a frozen multilingual
              encoder and injected as dense vectors into Llama-3.2-1B, bypassing brittle tokenization while leaving the
              tokenizer and decoder weights untouched.
            </span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>
              A two-stage alignment pipeline first teaches the projector to land in the decoder&apos;s latent space at a reserved
              slot, then expands that signal into eight soft tokens and applies minimal adapters plus a usage penalty so
              the frozen decoder actually relies on the injected information.
            </span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>
              The method slashes decoder token counts, roughly triples retrieval accuracy, and drives strong win rates in
              LLM-as-judge evaluations while still revealing open issues around numeric fidelity and literal translation.
            </span>
          </li>
        </ul>

        <h2 className="text-2xl font-semibold text-black mb-3 mt-12">Tokenization is the Bottleneck</h2>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-5">
          Khmer text is a worst-case scenario for English-heavy byte pair encoders. The same sentence tokenizes to about
          16 tokens in English, 35 in Latin transliteration, and 104 in native Khmer with the Llama-3.2 vocabulary. That
          six-fold inflation blows through context windows, drives quadratic attention cost before the model even reaches
          the instruction, and starves the decoder of signal from later tokens. Parameter-efficient finetuning techniques
          such as LoRA inherit the problem because they still run on the fragmented token stream, teaching the model to
          cope with junk tokens rather than eliminating them.
        </p>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-8">
          LLINK shifts the heavy lifting to a compact encoder and a handful of soft slots. Instead of forcing the decoder
          to untangle dozens of rare-script fragments, the approach injects a semantic summary that already lives in the
          decoder&apos;s hidden space. The decoder stays frozen, the tokenizer stays untouched, the prompt stays short, and
          Khmer prompts drop from triple-digit token counts to eight learned slots. In the ParaCrawl evaluation this
          reduction alone accounts for the majority of the cross-lingual retrieval gains seen after Stage A.
        </p>

        <h2 className="text-2xl font-semibold text-black mb-3">Two-Stage Injection Pipeline</h2>
        <h3 className="text-xl font-semibold text-black mb-2">Stage A: Contrastive Alignment</h3>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-4">
          A frozen XLM-R encoder produces sentence embeddings that are mean pooled and sent through a small projector
          (768 to 3072 to 2048 with GELU, dropout 0.10, and LayerNorm). The target is the decoder&apos;s hidden state at a
          reserved slot inside a prompt template that appends a placeholder token to the user instruction. Symmetric
          InfoNCE with in-batch negatives and a queue of 32768 teacher vectors handles alignment, mining the 256 hardest
          negatives per step so the projector learns to ignore look-alike contexts. Lightweight direction and log-norm
          penalties keep the projected vector close in both angle and magnitude. No decoder or tokenizer weights move, so
          the decoder experiences the injected vector as another internally generated state.
        </p>
        <h3 className="text-xl font-semibold text-black mb-2">Stage B: Slot Expansion and Usage</h3>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-8">
          The aligned vector is expanded into eight reserved tokens (f0 to f7) that sit in the prompt like ordinary
          context. LoRA adapters (rank 16, alpha 16) on attention and MLP projections, along with a learned slot scaler and
          expander, are trained on synthetic instruction-following tasks. Every third step the pipeline compares the
          supervised loss with slots against a variant where slots are zeroed and penalizes improvements with max(0, L_sft
          - L_zero). Auxiliary cosine and InfoNCE terms prevent drift away from the Stage A target, and the synthetic
          prompt set covers translation, summarization, bullet points, and question answering so the decoder has multiple
          ways to rely on the injected signal. A final normalization matches slot norms to the median embedding norm so
          the decoder treats the injected tokens as native context.
        </p>

        <div className="border border-gray-200 px-6 py-5 mb-12 bg-white">
          <p className="text-sm uppercase tracking-wide text-gray-500 mb-2 font-sans">Architecture callout</p>
          <p className="text-[17px] leading-relaxed text-gray-800">
            Figure 1 in the paper diagrams the flow: Khmer text enters the frozen XLM-R encoder, a contrastive projector
            drops the sentence vector into a reserved decoder position, and the expanded slots hand the content to the
            instruction-tuned Llama decoder, which responds in English without ever seeing Khmer tokens.
          </p>
        </div>

        <h2 className="text-2xl font-semibold text-black mb-3">Data and Training Setup</h2>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-4">
          Stage A uses 100k ParaCrawl v2 English-Khmer pairs with Khmer strings truncated at 256 characters and a 40k
          pair holdout for retrieval evaluation. Stage B relies on 40k synthetic instruction examples plus 2k validation,
          generated by prompting a Llama 3 70B model with the English reference to anchor outputs. Prompts cover tasks
          such as translate to English, summarize in English, bullet pointify, and question answering about the passage,
          with filtering to keep Khmer inputs between 12 and 256 characters and ensure the reserved slot appears.
        </p>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-8">
          The base Llama-3.2-1B decoder remains frozen throughout. LoRA rank is 16 with alpha 16, the slot count stays at
          eight, and the usage contrast is applied every third optimization step. Mixed precision (fp16 queue, bf16
          projector) keeps training lightweight, and the injection pipeline mirrors inference exactly: encode Khmer with
          XLM-R, project, expand into slots, then decode.
        </p>

        <h2 className="text-2xl font-semibold text-black mb-3">Key results</h2>
        <ul className="space-y-3 text-[17px] leading-relaxed text-gray-800 mb-8">
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>Retrieval: recall@1 climbs from 0.10 (direct LoRA finetune) to 0.45, recall@5 hits 0.724, recall@10 reaches 0.835, mean reciprocal rank lands at 0.66, and mean rank drops to 3.4.</span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>Stage contribution: Stage A alone delivers recall@1 of 0.43 and mean rank 3.8, with Stage B adding a modest bump and better usage of the injected slots.</span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>LLM-as-judge: on 200 understanding prompts LLINK wins 74 percent versus the base model and 49 percent versus the finetune; on 200 Q and A prompts wins are 51 percent and 42 percent respectively, yielding overall preferences of 84 percent and 64 percent.</span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>Token budget: Khmer prompts compress from 100-plus decoder tokens to eight soft slots, cutting decoder compute roughly three times while amortizing a single encoder pass.</span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>Preference taxonomy: LLINK excels on semantic understanding tasks while literal translation remains the hardest setting due to slot compression.</span>
          </li>
        </ul>

        <h2 className="text-2xl font-semibold text-black mb-3">Evaluation Breakdown</h2>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-4">
          Retrieval on a 1024-pair benchmark ranks each Khmer sentence against all English candidates. The contrastive
          bridge alone explains most of the improvement, proving that bypassing fragmented tokenization matters more than
          extra decoder tuning. Stage B helps primarily by enforcing slot usage, which pays off in generation tasks.
        </p>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-4">
          LLM-as-judge experiments follow the paper&apos;s Table 2 protocol: a Llama 3 70B model receives anonymized outputs
          and the human reference. LLINK wins 87 percent of understanding comparisons and 80 percent of Q and A
          comparisons against the base model, and keeps a 67 percent versus 59 percent edge over direct finetuning across
          those buckets. Ties remain substantial (15 to 36 percent), signaling space for future lexical fidelity work.
        </p>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-8">
          Qualitative examples in Table 3 highlight both sides: LLINK grounds answers like policy statements and class
          schedules that the base model garbles, yet it can still drift on units (30 MW becomes 1.5 kW) or over-summarize
          when the slot compression blurs rare terminology.
        </p>

        <h2 className="text-2xl font-semibold text-black mb-3">Qualitative Behavior</h2>
        <ul className="space-y-3 text-[17px] leading-relaxed text-gray-800 mb-8">
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>Positive cases: LLINK correctly surfaces policies about data sharing, restart dates for classes, and specific categorical labels where the base model outputs mixed Khmer or vague paraphrases.</span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>Negative cases: numeric fidelity is the recurring failure, with power ratings and quantities drifting, and literal translation prompts can trigger over-summarization.</span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>The analysis section links these errors to how multilingual encoders represent numbers on near logarithmic scales, making 30 and 1.5 surprisingly close in embedding space.</span>
          </li>
        </ul>

        <h2 className="text-2xl font-semibold text-black mb-3">Compute Trade-offs</h2>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-8">
          Treating Khmer as an injected modality swaps autoregressive processing of 100-plus tokens for a single encoder
          pass and eight decoder slots. The encoder cost can be cached across follow-up questions, enabling roughly three
          times fewer decoder tokens per prompt and allowing retrieval or question answering systems to reuse the same
          slot vector across multiple prompts. This shift preserves the frozen decoder&apos;s English strengths while
          sparing the context window for instructions, exemplars, or longer answers, and it plays nicely with downstream
          batching because slot tokens are constant length.
        </p>

        <h2 className="text-2xl font-semibold text-black mb-3">Bottlenecks</h2>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-4">
          Eight slots act like a semantic bottleneck. They preserve meaning but not always surface form, so numbers,
          dates, and rare entities can drift. The decoder remains frozen and English-dominant, so without explicit slot
          supervision it may paraphrase around foreign content. The usage contrast keeps the model honest but does not
          guarantee exact copy behavior.
        </p>
        <p className="text-[17px] leading-relaxed text-gray-800 mb-3">
          These weaknesses show up in judge evaluations as preference losses on prompts that demand literal translations.
          LLINK excels at question answering and summarization, but still trails a carefully curated translation system
          when the task is to mirror a sentence verbatim.
        </p>

        <h2 className="text-2xl font-semibold text-black mb-3">Future Work</h2>
        <ul className="space-y-3 text-[17px] leading-relaxed text-gray-800">
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>
              Extend the slot interface to more languages, including right-to-left scripts and logographic systems, and
              test larger decoder backbones whose stronger English priors may require heavier usage penalties or larger K.
            </span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>
              Predict slot count dynamically so short prompts can use two to four tokens while dense documents receive a
              wider channel, potentially via a lightweight classifier that looks at encoder entropy or length.
            </span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>
              Add copy-aware or numeric-preserving pathways so exact values survive the compression process, for example
              by dedicating slots to numerals or supervising attention from slots to output tokens.
            </span>
          </li>
          <li className="flex">
            <span className="mr-3 text-gray-400">&bull;</span>
            <span>
              Explore many-to-many transfer by pairing the slot projector with multiple teacher positions or targeting a
              language-agnostic intermediate space, enabling cross-lingual retrieval and generation beyond Khmer-English.
            </span>
          </li>
        </ul>
      </article>
    </div>
  );
}
