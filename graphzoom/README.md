## experiments
| dataset             | method        | level | shape(*128) | Acc      |
| ------------------- | ------------- | ----- | ----------- | -------- |
| cora                | gcn           | -     |             | 81.1     |
| citeseer            | gcn           | -     |             | 65.2     |
| pubmed              | gcn           | -     |             | 79.8     |
| cora                | zoom+deepwalk | 1     | 1169        | 79.6     |
| cora(no fusion)     | zoom+deepwalk | 1     | 1208        | 78.6     |
| cora                | zoom+gcn      | 1     | 1169        | **83.8** |
| citeseer            | zoom+deepwalk | 1     | 1402        | 51.0     |
| citeseer(no fusion) | zoom+deepwalk | 1     | 1548        | 49.3     |
| citeseer            | zoom+gcn      | 1     | 1402        | 0.635    |
| pubmed              | zoom+deepwalk | 1     | 7903        | 75.3     |
| pubmed(no fusion)   | zoom+deepwalk | 1     | 7523        | 77.5     |
| pubmed              | zoom+gcn      | 1     | 7903        | **83.8** |
| reddit              |
| amazon2m            |