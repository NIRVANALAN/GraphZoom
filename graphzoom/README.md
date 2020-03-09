## experiments
| dataset  | method        | level | shape     | acc   |
| -------- | ------------- | ----- | --------- | ----- |
| cora     | gcn           | -     |           | 81.1  |
| citeseer | gcn           | -     |           | 65.2  |
| pubmed   | gcn           | -     |           | 79.8  |
| cora     | zoom+deepwalk | 1     | 1169, 128 | 79.6  |
| cora     | zoom+gcn      | 1     | 1169, 128 | 83.8  |
| citeseer | zoom+deepwalk | 1     | 1402, 128 | 0.51  |
| citeseer | zoom+gcn      | 1     | 1402, 128 | 0.635 |
| pubmed   | zoom+deepwalk | 1     | 7903, 128 | 75.3  |
| pubmed   | zoom+gcn      | 1     | 7903, 128 | 83.8  |
| reddit   |
| amazon2m |