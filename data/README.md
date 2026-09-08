# Dataset setup

Dataset files are deliberately excluded from Git because they are large and
distributed under their own terms. Download each dataset from its official
source and place it under `data/raw/`.

| Dataset | Expected location | Official source |
|---|---|---|
| FEI Faces | `data/raw/FEI/` | [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html) |
| KSDD2 | `data/raw/KolektorSDD2/` | [Kolektor Surface-Defect Dataset 2](https://www.vicos.si/resources/kolektorsdd2/) |
| MVTec AD | `data/raw/mvtec_anomaly_detection/` | [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) |
| STL-10 | `data/raw/stl10_binary/` | [Stanford STL-10](https://cs.stanford.edu/~acoates/stl10/) |

The KSDD2 and MVTec AD providers specify non-commercial dataset terms. Review
the provider pages before redistributing or reusing the data. This repository
does not redistribute any images.

The loaders in `src/datasets/` validate their expected structures and report
the missing location when setup is incomplete.
