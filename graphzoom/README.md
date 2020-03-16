## experiments
| dataset             | method              | shape(*128) | Acc      |
| ------------------- | ------------------- | ----------- | -------- |
| cora                | gcn                 |             | 81.1     |
| cora                | gat                 |             | 81.3     |
| cora                | gat(DGL feature)    |             | 81.8     |
| citeseer            | gcn                 |             | 65.2     |
| citeseer            | gat                 |             | 63.6     |
| citeseer            | gat(DGL feature)    |             | 68.0     |
| pubmed              | gcn                 |             | 79.8     |
| pubmed              | gat                 |             | 77.8     |
| pubmed              | gat(DGL feature)    |             | 81.1     |
| reddit              | gcn                 |             | 94.48    |
| reddit              | gat                 |             |
| amazon2m            | gcn                 |             | 84.33    |
| cora                | zoom+deepwalk       | 1169        | 79.6     |
| cora(no fusion)     | zoom+deepwalk       | 1208        | 78.6     |
| cora                | zoom+gcn            | 1169        | **83.8** |
| cora                | zoom+gcn(train ids) | 1169        | *81.90*  |
| cora                | zoom+gat            | 1169        | 82.2     |
| citeseer            | zoom+deepwalk       | 1402        | 51.0     |
| citeseer(no fusion) | zoom+deepwalk       | 1548        | 49.3     |
| citeseer            | zoom+gcn            | 1402        | 64.6     |
| citeseer            | zoom+gat            | 1402        | **69.0** |
| pubmed              | zoom+deepwalk       | 7903        | 75.9     |
| pubmed(no fusion)   | zoom+deepwalk       | 7523        | *77.5*   |
| pubmed              | zoom+gat            | 7903        | 82.7     |

## propagation exps
| dataset  | method                           | time  | shape(*128) | Acc      |
| -------- | -------------------------------- | ----- | ----------- | -------- |
| pubmed   | zoom+gcn(train labels - all ids) | 1.5s  | 7903        | *38.7*   |
| pubmed   | zoom+gcn(all labels - all ids)   | 1.5s  | 7903        | **85.2** |
| pubmed   | zoom+gcn(all labels - 100 ids)   | 1.5s  | 7903        | **81.5** |
| citeseer | zoom+gcn(train labels)           |       | 1402        | 36.20    |
| citeseer | zoom+gcn(all labels - 60 ids)    |       | 1402        | 58.20    |
| citeseer | zoom+gcn(all labels - all ids)   |       | 1402        | 74.20    |
| reddit   | zoom+gcn(train labels - all ids) | 24s   | 84526       | 92.89    |
| amazon2m | zoom+gcn(train labels - all ids) | 63.8s | 924691      | 83.65    |