# Laboratorul 6 — Transformers: atenția, de la calculul de mână la un model real

Lucrare de laborator individuală la cursul de inteligență artificială generativă (Lecția 5, *Transformers*). Notebook-ul implementează mecanismul de atenție în NumPy, compară tokenizarea unor texte românești și generează text cu GPT-2 printr-o buclă scrisă de mână. Partea bonus analizează embeddings contextuale și atenția unui BERT românesc.

> Documentul-sursă al laboratorului se numește „Laboratorul 5. Transformers”; în cadrul cursului este Laboratorul 6.

## Conținut

| Parte | Ce conține | Statut |
|---|---|---|
| 1. Atenția în NumPy | softmax stabil, `attention(Q, K, V, mask)`, mască cauzală, multi-head attention | obligatorie |
| 2. Tokenizarea | GPT-2 vs. BERT multilingv vs. BERT românesc, costul limbii române, diacritice cu virgulă/sedilă | obligatorie |
| 3. Generarea textului | probabilitățile următorului token, buclă de generare manuală (greedy + temperatură), RoGPT2 | obligatorie (3.3 opțională) |
| 4. Embeddings contextuale | vectorul cuvântului „bancă” în 4 contexte, vizualizarea atenției cu `bertviz` | bonus |

## Structura repository-ului

```
├── Laborator6_Transformers.ipynb   # notebook-ul complet, rulat
├── atentie_bertviz.html            # vizualizarea interactivă a atenției (bertviz)
└── README.md
```

## Mediu și rulare

Rulat local, numai pe **CPU** (fără GPU NVIDIA), în PyCharm cu kernel Jupyter:

| Componentă | Versiune |
|---|---|
| Python | 3.12.16 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| PyTorch | 2.15.0+cpu |
| transformers | 5.21.0 |
| matplotlib | 3.11.2 |
| bertviz | instalat |

Modele folosite (Hugging Face): `gpt2`, `bert-base-multilingual-cased`, `dumitrescustefan/bert-base-romanian-cased-v1`, `readerbench/RoGPT2-base`.

```bash
conda create -n lab6_transformers python=3.12 -y
conda activate lab6_transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib transformers bertviz ipywidgets ipykernel
```

Apoi deschideți notebook-ul și rulați celulele în ordine, de sus în jos. La prima rulare se descarcă modelele (aproximativ 1,5–2 GB).

---

## Rezultate

### Partea 1 — Atenția în NumPy

Toate verificările din enunț au trecut:

| Verificare | Rezultat obținut |
|---|---|
| `softmax([1.2, 2.8, 0.7])` | `[0.152 0.755 0.092]` ✅ |
| Ponderi de atenție pentru „aprobat” | `[0.152 0.755 0.092]` ✅ |
| Vectorul de ieșire `z` | `[0.199 0.801]` ✅ (în curs: 0,195 / 0,805, calculat cu ponderi rotunjite) |
| Mască cauzală: sume pe rânduri | `[1. 1. 1. 1.]` ✅ |
| Rândul 1 / rândul 2 al ponderilor mascate | `[1 0 0 0]` / `[0.648 0.352 0 0]` ✅ |
| Multi-head (2 capete) — dimensiunea ieșirii | `(4, 8)` ✅ |

Matricea ponderilor cu mască cauzală:

```
[[1.    0.    0.    0.   ]
 [0.648 0.352 0.    0.   ]
 [0.491 0.145 0.363 0.   ]
 [0.285 0.341 0.281 0.093]]
```

**Softmax naiv vs. stabil pentru `[1000, 1001]`:** varianta naivă dă `[nan nan]`, cu avertismentele *overflow encountered in exp* și *invalid value encountered in divide*. Varianta stabilă dă `[0.269 0.731]`.

### Partea 2 — Tokenizarea

Numărul de tokeni:

| Model | „Banca a aprobat creditul clientului.” | „Împrumuturile restructurate au dobânzi variabile.” |
|---|:---:|:---:|
| GPT-2 | 12 | 20 |
| BERT multilingv | 10 | 15 |
| **BERT românesc** | **6** | **8** |
| *(cuvinte, ca reper)* | *5* | *5* |

BERT-ul românesc păstrează întregi aproape toate cuvintele (`aprobat`, `creditul`, `clientului`, `dobânzi`). GPT-2, antrenat mai ales pe engleză, fragmentează puternic cuvintele și sparge diacriticele în octeți (`Î` → `Ã`, `İ`; `â` → `Ã¢`).

**Costul limbii române la GPT-2:**

| Fraza | Tokeni | Cost pentru 1 milion de fraze (1 USD / 1 mil. tokeni) |
|---|:---:|:---:|
| RO: *Banca a aprobat creditul clientului.* | 12 | 12,00 USD |
| EN: *The bank approved the client's loan.* | 8 | 8,00 USD |
| **Raport RO / EN** | **1,5×** | **+4,00 USD (+50%)** |

**Diacritice cu virgulă vs. cu sedilă (BERT românesc):**

| Text | Tokeni |
|---|---|
| `știință și țară` (ș U+0219, ț U+021B) | `['știință', 'și', 'țară']` |
| `ştiinţă şi ţară` (ş U+015F, ţ U+0163) | `['[UNK]', '[UNK]', '[UNK]']` |
| forma cu sedilă după normalizare | `['știință', 'și', 'țară']` ✅ |

Pentru model, forma cu sedilă e un text necunoscut: toate cele trei cuvinte devin `[UNK]`, deci informația se pierde complet. După normalizarea Unicode (NFC) și înlocuirea `ş→ș`, `ţ→ț`, `Ş→Ș`, `Ţ→Ț`, tokenizarea devine identică cu cea a formei corecte.

### Partea 3 — Generarea textului cu GPT-2

**Următorul token după „The bank approved the”**. Vectorul `logits` are dimensiunea **50 257**, adică mărimea vocabularului GPT-2. Suma probabilităților după Softmax este ≈ 1.

| # | Token | Probabilitate |
|---|---|---|
| 1 | ` loan` | 0,066 |
| 2 | ` deal` | 0,066 |
| 3 | ` sale` | 0,037 |
| 4 | ` transaction` | 0,033 |
| 5 | ` plan` | 0,025 |

**Generare cu buclă manuală (25 de tokeni, fără `model.generate()`):**

| Setare | Text generat (fragment) | Observații |
|---|---|---|
| Greedy | *The bank approved the loan in May, but the bank has not yet approved the loan. The bank said it will not be able to* | Coerent gramatical, dar repetitiv („the bank”, „approved the loan”); identic la fiecare rulare |
| T = 0,2 (seed 1020) | *The bank approved the deal on Friday, but the deal has yet to be signed. The bank said it would not be able to pay* | Foarte apropiat de greedy: aceeași structură, cu repetiții |
| T = 0,2 (seed 1021) | *The bank approved the deal on Tuesday, but it has yet to announce its financial results. The bank's chief executive, Mark Carney,* | Coerent, stil de știre; introduce un nume real fără legătură factuală |
| T = 0,7 (seed 1070) | *The bank approved the agreement on Tuesday, calling it a "fairly fair and effective agreement"…* | Variat și în mare coerent |
| T = 0,7 (seed 1071) | *The bank approved the purchase at a market rate of 2.5 percent. The bank also agreed to pay $3.7 billion…* | Variat, coerent, cu cifre inventate |
| T = 1,5 (seed 1150) | *…its bid to restave shortage came 2016. Starbucks seems willingHider aquil nonprofits…* | Incoerent, cu cuvinte inexistente |
| T = 1,5 (seed 1151) | *…historical USPS another -- 3W59. $63Us came with John to Ray joint referencing -BD* | Incoerent, aproape aleator |

Efectul temperaturii asupra distribuției pentru promptul de mai sus:

| T | p(top-1) | Masa top-10 | Entropie (nats) |
|:---:|:---:|:---:|:---:|
| 0,2 | 0,482 | 0,9997 | 0,92 |
| 0,7 | 0,175 | 0,672 | 3,74 |
| 1,0 | 0,066 | 0,316 | 6,22 |
| 1,5 | 0,013 | 0,076 | 8,67 |

**RoGPT2 (opțional), promptul „Banca a aprobat”:** primii tokeni propuși sunt ` în` (0,108), ` un` (0,075), `,` (0,060). Generare greedy: *„Banca a aprobat în octombrie 2018 un împrumut de 1,5 miliarde de euro de la Banca Europeană de Investiții (BEI) pentru finanțarea proiectelor”*. Textul e fluent în română, dar faptele sunt inventate.

### Partea 4 (bonus) — Embeddings contextuale și atenție

Similaritatea cosinus a vectorului „bancă” în cele patru propoziții:
1. *Am deschis un cont la bancă.*
2. *Am luat un credit de la bancă.*
3. *M-am așezat pe o bancă în parc.*
4. *Bătrânul stă pe bancă lângă fântână.*

```
Stratul 0                       Stratul 12
[[1.    0.971 0.971 0.961]      [[1.    0.953 0.659 0.594]
 [0.971 1.    1.    0.954]       [0.953 1.    0.625 0.56 ]
 [0.971 1.    1.    0.954]       [0.659 0.625 1.    0.832]
 [0.961 0.954 0.954 1.   ]]      [0.594 0.56  0.832 1.   ]]
```

| Pereche | Stratul 0 | Stratul 12 |
|---|:---:|:---:|
| 1–2 (ambele financiare) | 0,971 | 0,953 |
| 1–3 (financiar vs. loc de șezut) | 0,971 | 0,659 |
| 3–4 (ambele loc de șezut) | — | 0,832 |

La stratul 0 vectorul depinde doar de embedding și de poziție. Propozițiile 2 și 3 au similaritatea exact 1,000, pentru că „bancă” apare pe aceeași poziție în ambele. La stratul 12, contextul separă clar cele două sensuri: 1–2 = 0,953, față de 1–3 = 0,659 (diferență +0,294). Propozițiile 3 și 4 se grupează între ele (0,832).

**Atenția „aprobat” → „creditul”** pentru fraza *Banca a aprobat creditul clientului.*:

| Strat | Cap | Pondere |
|:---:|:---:|:---:|
| 2 | 4 | 0,995 |
| 3 | 11 | 0,947 |
| 8 | 12 | 0,891 |
| 5 | 8 | 0,560 |
| 10 | 8 | 0,464 |

Vizualizarea interactivă se află în `atentie_bertviz.html` (are nevoie de internet pentru bibliotecile JavaScript).

---

## Răspunsuri la întrebări

**1. Softmax naiv vs. stabil.** Varianta naivă a dat `[nan nan]`, cea stabilă `[0.269 0.731]`. `exp(1000)` depășește maximul `float64` (≈ 1,8·10³⁰⁸ = e^709,78) și devine `inf`; `inf / inf` este nedefinit, deci `nan`. Scăderea maximului nu schimbă rezultatul, pentru că factorul e^(−c) apare și la numărător, și la numitor și se simplifică. În schimb, cel mai mare exponent devine 0, așa că se calculează `exp([−1, 0])`, fără overflow.

**2. Masca cauzală.** Primul token are voie să se vadă doar pe sine. Celelalte scoruri devin −∞, iar e^(−∞) = 0, deci Softmax dă `[1, 0, 0, 0]`. Al treilea token vede tokenii 1, 2 și 3, cu ponderile `[0.491 0.145 0.363]`.

**3. Costul limbii române.** Fraza românească are 12 tokeni GPT-2, cea engleză 8, deci de 1,5 ori mai mulți. Un milion de fraze ar costa 12 USD în română față de 8 USD în engleză (+50%). Tot din acest motiv, în aceeași fereastră de context încape cu o treime mai puțin text românesc.

**4. Virgulă vs. sedilă.** Cu virgulă, BERT românesc recunoaște cuvintele întregi. Cu sedilă, toate trei devin `[UNK]`. Înainte de a da textele unui model, ele trebuie normalizate: Unicode NFC și înlocuirea ş/ţ/Ş/Ţ cu ș/ț/Ș/Ț. După normalizare, tokenizarea coincide cu cea corectă.

**5. Logits.** Vectorul are 50 257 de componente, câte un scor pentru fiecare token din vocabularul GPT-2. Primii trei tokeni propuși sunt ` loan` (0,066), ` deal` (0,066) și ` sale` (0,037).

**6. Greedy și temperatura.** Greedy alege mereu `argmax`, iar modelul în modul `eval` este determinist, deci textul iese identic la fiecare rulare (verificat). Temperatura împarte scorurile înainte de Softmax: p_i = e^(z_i/T) / Σ e^(z_j/T). Un T mic accentuează diferențele dintre scoruri: p(top-1) = 0,48 și entropia 0,92 nats la T = 0,2, iar textul rămâne apropiat de greedy. Un T mare aplatizează distribuția: p(top-1) = 0,013 și entropia 8,67 nats la T = 1,5, iar textul devine variat, dar incoerent.

**7. Embeddings contextuale (bonus).** La stratul 0: sim(1,2) = 0,971 și sim(1,3) = 0,971, deci contextul nu contează încă. La stratul 12: sim(1,2) = 0,953 și sim(1,3) = 0,659. După cele 12 blocuri de atenție, vectorul lui „bancă” reflectă sensul din context, iar sensul financiar se separă clar de cel de „loc de șezut”.

## Concluzii

- Implementarea atenției în NumPy reproduce exact exemplul din curs, iar masca cauzală anulează corect ponderile către tokenii viitori.
- Tokenizarea depinde puternic de limba de antrenare. Pe cele două fraze, BERT românesc are nevoie de 2–2,5 ori mai puțini tokeni decât GPT-2. Asta are efecte directe asupra costului și a ferestrei de context.
- Normalizarea diacriticelor este obligatorie pentru textele românești: forma cu sedilă duce la pierderea completă a informației (`[UNK]`).
- Temperatura controlează compromisul dintre coerență și diversitate, vizibil atât în entropia distribuției, cât și în textele generate.
- Embeddings-urile contextuale ale BERT disting sensurile lui „bancă” abia în straturile adânci; la stratul 0 diferențele vin doar din poziție.
