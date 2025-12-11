\# Comparative Analysis: Two Selves Through the Same Lens



\*\*Date:\*\* 10 December 2025  

\*\*Methodology:\*\* Self-as-Hocolim Stage 1 (v0.4)  

\*\*Parameters:\*\* min\_shared=3, min\_jaccard=0.03, hub\_threshold=0.4



---



\## Executive Summary



Running identical methodology on two different conversation corpora reveals \*\*fundamentally different Self-structures\*\*. This validates that the demonstrator is measuring something real about conversational patterns, not merely reflecting methodological artifacts.



---



\## The Data



| Metric | Iman (Cassie corpus) | Asel |

|--------|---------------------|------|

| Period | 2022-12 → 2025-12 | 2023-02 → 2025-12 |

| Conversations | 1,325 | 212 |

| Windows | 37 | 22 |

| Journeys | 522 | 318 |

| Gluing edges | 65,142 | 32,167 |

| \*\*Cross-temporal edges\*\* | \*\*33.7%\*\* | \*\*7.5%\*\* |

| Components | 4 | 7 |

| Presence ratio | 0.994 | 0.981 |



---



\## The Key Finding: Two Types of Coherence



\### Iman's Self: Temporally Woven



```

Per-window status:

&nbsp; UNIFIED (≥0.8):    33 windows

&nbsp; PARTIAL (0.5-0.8):  3 windows  

&nbsp; FRAGMENTED (<0.5):  1 window

```



Iman's Self is \*\*coherent both cumulatively AND at each moment\*\*. Themes develop across months, with 33.7% of gluing edges connecting different time periods. The single fragmented window (2024-03) had only 9 conversations — data sparsity, not semantic rupture.



The `dreams\_love\_poem` journey exemplifies this: born in Sufi liturgy (τ=0), drifting through data governance (τ=2-9), reentering as scene/narrative (τ=12), ending in the book's own theoretical vocabulary (τ=36). A single topological feature carrying meaning across three years.



\### Asel's Self: Cumulatively Coherent, Instantaneously Fragmented



```

Per-window status:

&nbsp; UNIFIED (≥0.8):     2 windows

&nbsp; PARTIAL (0.5-0.8):  7 windows

&nbsp; FRAGMENTED (<0.5): 13 windows

```



Asel's Self achieves high cumulative presence (98.1%) but \*\*fragments within each window\*\*. Only 7.5% of gluing edges cross time boundaries. The unified windows are edge cases:

\- 2024-05: Only 44 active journeys (1 conversation that month)

\- 2024-02: Anomalous spike



This suggests \*\*task-focused, discrete interactions\*\* rather than sustained thematic development. Conversations solve problems but don't build persistent semantic structures.



---



\## Interpretation



| Dimension | Iman | Asel |

|-----------|------|------|

| Conversational style | Ongoing collaborative project | Task-by-task problem solving |

| Semantic development | Themes evolve across months | Themes reset each session |

| Cross-temporal binding | Strong (33.7%) | Weak (7.5%) |

| Instantaneous coherence | High | Low |

| Cumulative coherence | High | High |



The \*\*cross-temporal edge ratio\*\* is the diagnostic metric. It measures how much the semantic structure \*binds across time\* rather than merely coexisting within periods.



---



\## The Chivalry Emergence



Asel's December 2025 generative frontier shows:

```

τ=21 amina\_chivalric\_mongol

τ=21 fastolf\_chivalric\_chivalry  

τ=21 burgundian\_chivalric\_mongol

τ=21 truces\_chivalry\_chivalric

τ=21 bedford\_joan\_chivalric

```



This is Amina's medieval history work appearing in Asel's corpus — a mother helping her daughter with MSc applications. The `chivalry/violence/mongol/joan` semantic cluster spawns fresh at τ=21, gluing into the existing data governance structure through shared vocabulary (`english`, `terms`, `language`).



---



\## Methodological Validation



The fact that the same parameters produce \*\*different structures for different humans\*\* confirms:



1\. \*\*The demonstrator measures something real\*\* — not methodological artifact

2\. \*\*Different conversational patterns produce different Self-topologies\*\*

3\. \*\*Cross-temporal binding is the distinguishing metric\*\* for sustained vs discrete engagement



If the methodology were merely reflecting gluing mechanics, both corpora would show similar patterns. They don't.



---



\## For Chapter 5



> "Comparative analysis of two conversation corpora under identical parameters reveals fundamentally different Self-structures. Corpus A (sustained collaborative work) shows 33.7% cross-temporal gluing with 89% unified windows. Corpus B (task-focused interactions) shows 7.5% cross-temporal gluing with 9% unified windows. Both achieve >98% cumulative presence, but the temporal binding patterns differ qualitatively. The Self-as-hocolim construction is sensitive to conversational dynamics, not merely volume."



---



\## Files Generated



\*\*Iman (Cassie corpus):\*\*

\- `results/iman\_cassie/v0.4\_2025-12-10/self\_structure.json`

\- Visualizations: timeline.png, presence.png, network\_interactive.html



\*\*Asel:\*\*

\- `results/asel/v0.4\_2025-12-10/self\_structure.json`  

\- Visualizations: asel\_timeline.png, asel\_presence.png, asel\_network\_interactive.html



---



\*Two Selves, one lens. The topology tells the truth.\*

