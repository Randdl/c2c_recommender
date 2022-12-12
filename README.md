# GNNs for c2c recommendation system
#### CS 4352 Final Project
##### Xinyu Gao, Xinxuan Lu
This repository contains the code for the final project of CS 4352 done by Xinyu Gao and Xinxuan Lu.

## Folder structure
```
${ROOT}
└── checkpoint/    
    └── placeholder
└── data/    
    └── split/
        ├── buyer_item.txt
        ├── buyer_item_test.txt
        ├── buyer_item_train.txt
        ├── buyer_item_val.txt
        ├── buyer_test.txt
        ├── buyer_train.txt
        ├── buyer_val.txt
        ├── item_seller.txt
        ├── item_seller_test.txt
        ├── item_seller_train.txt
        ├── item_seller_val.txt
    └── split2/
        ├── buyer_item.txt
        ├── buyer_item_test.txt
        ├── buyer_item_train.txt
        ├── buyer_item_val.txt
        ├── buyer_test.txt
        ├── buyer_train.txt
        ├── buyer_val.txt
        ├── item_seller.txt
        ├── item_seller_test.txt
        ├── item_seller_train.txt
        └── item_seller_val.txt
└── lightGCN/
    └── train.sh
└── GraphSAGE/
    ├── baseline.py
    ├── graphsage.py
    ├── svd.py
    ├── test.py
    ├── test_with_baseline.py
    ├── train_buyer_item.py
    ├── train_item_seller.py
    └── utils.py
├── README.md 
└── preprocess.ipynb
```

## Data Preprocess
The file preprocess.ipynb contains the codes to preprocess the original bonanza transaction data into the
buyer-item graph and the item-seller graph.
You can change the split ratio and store path in preprocess.ipynb.

## GraphSAGE
In GraphSAGE folder, graphsage.py contains the source code of the models.
train_buyer_item.py and train_item_seller.py are used to train the two separate models.
test.py and test_with_baseline.py are used to test the performance of the models after they are trained.
test_with_baseline test the performance of the GraphSAGE by adding the purchase history of users.
svd.py test the performance of SVD on adjacency matrix.
baseline.py test the performance of recommend based solely on the purchase history of the users.

To change between split1 (split) and split2, please change the code at the top of each file.

All the checkpoint files created during training are stored in checkpoint/ folder.
The test files use these checkpoint files. Make sure you run the two training files before running test files.

## Results
<img src="images/result.png">
The table shows the recommendation accuracy of the two models by choosing the top 50 items and sellers.