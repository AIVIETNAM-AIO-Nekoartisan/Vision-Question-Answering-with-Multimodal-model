# Results

| config | split | n | acc | group | unparsed | L1 | L2 | L3 |
|---|---|---|---|---|---|---|---|---|
| +asr-gt | dev | 200 | 0.315 | 0.166 | 0.000 | 0.517 | 0.275 | 0.283 |
| +asr-gt | test | 600 | 0.218 | 0.100 | 0.000 | 0.314 | 0.143 | 0.207 |
| +ocr | dev | 200 | 0.325 | 0.179 | 0.000 | 0.552 | 0.294 | 0.283 |
| +ocr | test | 600 | 0.227 | 0.101 | 0.000 | 0.314 | 0.143 | 0.225 |
| baseline-uniform | dev | 200 | 0.255 | 0.146 | 0.000 | 0.483 | 0.196 | 0.225 |
| baseline-uniform | test | 600 | 0.208 | 0.094 | 0.000 | 0.314 | 0.124 | 0.196 |
| full | dev | 200 | 0.300 | 0.163 | 0.000 | 0.552 | 0.294 | 0.242 |
| full | test | 600 | 0.223 | 0.099 | 0.000 | 0.296 | 0.174 | 0.211 |
| visual-only | dev | 200 | 0.310 | 0.165 | 0.000 | 0.483 | 0.294 | 0.275 |
| visual-only | test | 600 | 0.215 | 0.099 | 0.000 | 0.314 | 0.124 | 0.211 |

Random baseline is 12.8% (8 options).
Accuracy alone is not a result: use the paired McNemar test, since the
configs answer the same questions.
