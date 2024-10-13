# Model Ensemble for Medical Image Segmentation
In this project, you'll dive into the idea of using multiple models together, known as model ensembles, to make our deep learning solutions more accurate. They are a reliable approach to improve accuracy of a deep learning solution for the added cost of running multiple networks. Using ensembles is a trick that's widely used by the winners of AI competitions.

Task of the students: explore approaches to model ensemble construction for semantic segmentation, select a dataset (preferentially cardiac MRI segmentation, but others also allowed), find an open-source segmentation solution as a baseline for the selected dataset and test it. Train multiple models and construct an ensemble from them. Analyse the improvements, benefits and added costs of using an ensemble.

## Csapat
- Csapat: Solo
- JGOB5J - Pongrácz Ádám

## Adathalmaz ACDC:
A házi feladat elkészítéséhez a 2017-es MICCAI kihíváshoz készűlt ACDC adathalmazt választottam. Ez a kihívás szív MR képek szegmentálásával foglalkozik. Bővebb információ az adathalmazról a következő linken érhető el: https://acdc.creatis.insa-lyon.fr/

Az adathalmaz a következő oldaltól tölthető le: [ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb/folder/637218e573e9f0047faa00fc)

## Fájlok és futtatás
MedImgEnsemble_DataPrep.ipynb: A notebook tartalmazza az adathalmaz letöltését és az adatok előkészítését. 

## Referenciák

### Related GitHub repositories:
- [divergent-nets](https://github.com/vlbthambawita/divergent-nets)
- [SenFormer](https://github.com/WalBouss/SenFormer)

### Related papers:
- [DivergentNets: Medical Image Segmentation by Network Ensemble](https://arxiv.org/abs/2107.00283)
- [Efficient Self-Ensemble for Semantic Segmentation](https://arxiv.org/abs/2111.13280)
- [Diversity-Promoting Ensemble for Medical Image Segmentation](https://dl.acm.org/doi/abs/10.1145/3555776.3577682)
