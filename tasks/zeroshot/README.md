# Zero-shot assessments
Following review [paper](https://arxiv.org/abs/2505.09087), we carry out zero-shot homology classification (task2 for Rfam, task3 for ArchiveII) and zero-shot secondary structure prediction (task1). The origin data and feature embeddings of other models can be obtained at [https://zenodo.org/records/14430869](https://zenodo.org/records/14430869).


Download data and place them in the folder of each task as
- task\*\_\*
    - data
        - \*.pkl
        - ...

Then run the following comannds to generate structRFM embeddings:
```shell
python3 demo_task2_rfam.py
python3 demo_task3_archiveII.py
python3 demo_task1.py
```
