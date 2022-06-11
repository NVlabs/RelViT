Data Preparation
===

1. Download the HICO dataset from [here](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view) and extract it to `./assets/data/hico`

2. Download our customized HICO annotations from [here](https://drive.google.com/file/d/11RvCM0KIBB4pFw0NuaX6Z6IPvnkrb83H/view?usp=sharing) and extract them to `./assets/data/hico/hico_20160224_det`

3. Download the images of GQA from [here](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) and extract them to `./assets/data/gqa`

4. Download our customized GQA annotations and meta files from [here](https://drive.google.com/file/d/1hKlRiikRkfZNB-St4kOzuMdLv8N5xk15/view?usp=sharing) and extract them to `./assets/data/gqa`

5. Download the pretrained vision backbones and other files from [here](https://drive.google.com/file/d/1pxmUxkk5t8Bg_cS_jdaQgugCqYddZInE/view?usp=sharing) and extract them to `./cache`

6. The file structure should look like
    ```plain
    data
    ├── gqa
    │   ├── dicts.json
    │   ├── gqa_dic.pkl
    │   ├── images
    │   └── raw
    │       └── questions1.2
    │           ├── test_balanced_questions.json
    │           ├── testdev_balanced_questions.json
    │           ├── train_balanced_concepts.json
    │           ├── train_balanced_questions.json
    │           ├── train_sys_reduced_concepts.json
    │           ├── train_sys_reduced_questions.json
    │           ├── val_balanced_questions.json
    │           └── val_sys_reduced_questions.json
    └── hico
        └── hico_20160224_det
            ├── images
            │   ├── test2015
            │   └── train2015
            ├── instances_test2015.json
            ├── instances_train2015.json
            ├── sys_vcl_nonrare_instances_test2015.json
            ├── sys_vcl_nonrare_instances_train2015.json
            ├── sys_vcl_rare_instances_test2015.json
            └── sys_vcl_rare_instances_train2015.json
    ```