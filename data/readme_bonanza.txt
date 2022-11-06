num: buyer,item, seller: 8922 3119 4056
num_data 39100 39100
num train, test, val:  31280 3910 3910


num: buyer,item, seller: 8922 3119 4056
num_data 39100 39100
num train, test, val:  31280 3910 3910

print("num train, test, val: ", train_list.__len__(), test_list.__len__(), val_list.__len__())
    pickle.dump((train_list, val_list, test_list, item_seller_set,
                 num_buyer, num_item, num_seller,
                 buyer_set, item_set, seller_set,
                 buyer_bought_dict, buyer_negative_dict,
                 buyer_item_seller_filering, buyer_count_dict,
                 item_count_dict, seller_count_dict),
                open(base_dir + "bonanza_" + str(num_buyer_filtering) + ".pickle", 'wb'))


bonanza_buyer_item_seller_id_R_2.txt
user, item, seller, itemstamp