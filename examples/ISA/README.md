# Interpretable Subgraph Attention (ISA) network 
ISA network is intended to provide an interpretable analysis based on attention mechanism. This model mainly introduces two approaches to enhance interpretability, compared to conventional attention models.   
### Subgroup segmentation
Firstly, it supports versatile scale of subgroups using segmentation rule, and the researcher can customized the way of segmentation by varing the segmentation index parameter. Because the subgroups which are chemically related to the given property or which are intersting to researchers can be varied by the property and tasks, there needs to be a choice of subgroups for researchers.
![Segmentation samples](assets/images/img1.png)

### Positive & Negative (PN) attention
Conventional attention has no capacity to inform how the feature effects on the prediction result. PN attention was introduced to solve this limitation. In its two way of streams, the positive and negative contribution of subgroups are seperately trained. Then, the attention scores given by each stream (named as positive attention score (PAS) and negative attention score (NAS)) indicate which groups contribute positively or negatively on the property.

![Architecture of ISA](assets/images/img2.png)

### Score Analysis
For effective analysis, it will be better to see the statistical score distribution, rather than the score of single moleucle. ISA Analyzer supports the simple statistical analysis through the dataset.

![Example of statistical analysis of ISA analyzer](assets/images/img3.png)

### Aligned Analyzer API

Load the saved training-time fragmentation rules and keep fragment/atom indices
with every result:

```python
from D4CMPP2 import Analyzer

analyzer = Analyzer(
    "path/to/saved/ISA/model",
    save_result=False,
    device="cpu",
)
analysis = analyzer.analyze_rows(
    ["CCOC(=O)c1ccccc1"],
    include_features=False,
)
row = analysis[0]
print(row.fragment_atom_indices)
print(row.score_mode, row.scores["positive"])
analyzer.plot_analysis(row, score="positive")
```

`Analyzer(model_path)` returns `ISAAnalyzer_v2` for positive-only ISA/GC models
and `ISAPNAnalyzer_v2` for ISATPN models. ISAT scores are atom-aligned;
GC/ISATPN scores and ISATPN hidden features are
fragment-aligned. `row.atom_scores("positive")` performs an explicit
fragment-to-atom expansion for visualization.
