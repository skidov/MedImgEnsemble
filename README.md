# Model Ensemble for Medical Image Segmentation
In this project, you'll dive into the idea of using multiple models together, known as model ensembles, to make our deep learning solutions more accurate. They are a reliable approach to improve accuracy of a deep learning solution for the added cost of running multiple networks. Using ensembles is a trick that's widely used by the winners of AI competitions.

Task of the students: explore approaches to model ensemble construction for semantic segmentation, select a dataset (preferentially cardiac MRI segmentation, but others also allowed), find an open-source segmentation solution as a baseline for the selected dataset and test it. Train multiple models and construct an ensemble from them. Analyse the improvements, benefits and added costs of using an ensemble.

## Csapat
- Csapat: Solo
- JGOB5J - Pongrácz Ádám

## Adathalmaz Kvasir SEG:
A Kvasir-SEG adathalmaz egy orvosi képadatbázis, amelyet főként a vastagbél endoszkópiás képeinek szegmentálására használnak. Az adathalmaz 1000 endoszkópos képet, melyek a vastagbél különböző szakaszait ábrázolják, valamint a hozzájuk tartozó maszkot tartalmaz, amik kijelölik a polipok és egyéb elváltozások helyét a képeken. 

Az adathalmaz a következő oldaltól tölthető le: [Kvasir SEG](https://datasets.simula.no/kvasir-seg/)

## Fájlok
- **data**: Adatok betöltésért és feldolgozásáért felelős modulok találhatóak benne.
  - **kvasir_seg_datamodule.py**: A Kvasir-SEG adathalmaz betöltéséért felelős LightningDataModule-t tartalmazza
  - **kvasir_seg_dataset.py**: A Kvasir-SEG adathalmaz betöltéséért felelős Dataset-et tartalmazza
- **utils**: Kiértékelésért felelős modulok találhatóak benne.
  - **kvasir_seg.py**: A model checkpoint betöltéséért és a különböző plotok megjelenítéséért felelős függvényeket tartalmazza
  - **metrics.py**: A tanításnál használt metrikákat tartalmaz
- **model**: A modelleket tartalmazó modulok találhatóak benne
  - **model_baseline.py**: A Baseline modellt tartalmazza
  - **model_ensemble.py**:A Enesemble modellt tartalmazza
  - **model_fcn.py**: A FCN modellt tartalmazza
  - **model_tri_unet.py**: A TriUNet modellt tartalmazza
  - **model_unet.py**: A UNet modellt tartalmazza
- **train.py**: A tanítás indításáért felelős modul. Tanítás után autómatikusan elindítja a test adathalmazon való tesztelést
- **main.py**: A futtatásért felelős modul. Elindíthatóak benne a különböző tanítások, valamint a plotok megjelnítése

## Futtatás
A main.py futtatásával lehet elindítani a különböző tanításokat.

A main.py egy menüt tartalmaz, amivel kiválaszthatóak a modellek. A következő parancsokkal választhatóak ki a menüben:
- **train_baseline** -- Baseline model tanítása
- **train_fcn** -- FCN model tanítása
- **train_unet** -- UNet model tanítása
- **train_tri_unet** -- TriUNet model tanítása
- **train_ensemble** -- Ensemble model tanítása
- **show_masked_img** *rows* -- Plot megjelenítése az adathalmazból vett pédákkal
  - *rows* -- Megjelenítendő képek száma
- **show_predicted_img** *model_name* *rows* -- Plot megjelenítése a model által prediktált maszkkal
  - *model_name* -- A model checkpoint neve, a trained_models mappában (.ckpt kiterjeszés nélkül)
  - *rows* -- Megjelenítendő képek száma

### Docker
- docker build . -t med_img_ensemble
- docker run -it med_img_ensemble

## Referenciák

### Related GitHub repositories:
- [divergent-nets](https://github.com/vlbthambawita/divergent-nets)

### Related papers:
- [DivergentNets: Medical Image Segmentation by Network Ensemble](https://arxiv.org/pdf/2107.00283)
- [A multi-centre polyp detection and segmentation dataset for generalisability assessment](https://arxiv.org/pdf/2106.04463)
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597v1)
- [Kvasir-SEG: A Segmented Polyp Dataset](https://arxiv.org/pdf/1911.07069v1)
