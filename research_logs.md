With undersampling the pre-classifier trained on labeled small data, the precision for fakes (-1) fropped from 0.4 to 0.18, but the precision for not fakes(+1) increqsed from 0.9 to 0.94 so we are going to try the oversampling with SMOTE technique.

Result for SMOTE: precision for -1s:0.2 (droped from 0.4) and +1s: 0.92

Adter ML, we tried a neurl classifier, the precision for -1s: and for +1s:

Note: if we use business_id in our neural method, since the new dataset business_id is totally different and there is not any way to understand what is the mapped id of new businesses in labeled original dataset, maybe it's better just to consider review texts.

We labeled the small_unlabeled_dataset (which contain the social relatioships), then trained a simple link classifier (not binary link prediction)