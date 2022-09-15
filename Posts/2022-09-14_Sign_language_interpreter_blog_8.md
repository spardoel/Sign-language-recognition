# Developing a sign language interpreter using machine learning Part 8 - Training the model on the new dataset

So the first model was terrible. But the dataset has been reorganized and augmented. It is now time to try again. 

## Modifications to load the new dataset format

Aug10

```
Frame features in train set: (670, 50, 1280)
Frame masks in train set: (670, 50)
['before', 'book', 'chair', 'cousin', 'deaf', 'drink', 'fine', 'like', 'now', 'walk']
Epoch 1/150
21/21 [==============================] - ETA: 0s - loss: 2.3195 - accuracy: 0.1507  
Epoch 1: val_loss improved from inf to 2.27121, saving model to video_classifier.h5
21/21 [==============================] - 11s 385ms/step - loss: 2.3195 - accuracy: 0.1507 - val_loss: 2.2712 - val_accuracy: 0.2143 - lr: 5.0000e-04
Epoch 2/150
21/21 [==============================] - ETA: 0s - loss: 2.3026 - accuracy: 0.1612
Epoch 2: val_loss did not improve from 2.27121
21/21 [==============================] - 7s 335ms/step - loss: 2.3026 - accuracy: 0.1612 - val_loss: 2.2820 - val_accuracy: 0.2143 - lr: 5.0000e-04
Epoch 3/150
21/21 [==============================] - ETA: 0s - loss: 2.2831 - accuracy: 0.1761
Epoch 3: val_loss did not improve from 2.27121
21/21 [==============================] - 8s 382ms/step - loss: 2.2831 - accuracy: 0.1761 - val_loss: 2.2757 - val_accuracy: 0.2143 - lr: 5.0000e-04
Epoch 4/150
21/21 [==============================] - ETA: 0s - loss: 2.2809 - accuracy: 0.1851
Epoch 4: val_loss improved from 2.27121 to 2.24690, saving model to video_classifier.h5
21/21 [==============================] - 8s 389ms/step - loss: 2.2809 - accuracy: 0.1851 - val_loss: 2.2469 - val_accuracy: 0.2143 - lr: 5.0000e-04
Epoch 5/150
21/21 [==============================] - ETA: 0s - loss: 2.2763 - accuracy: 0.1970
Epoch 5: val_loss improved from 2.24690 to 2.21383, saving model to video_classifier.h5
21/21 [==============================] - 8s 379ms/step - loss: 2.2763 - accuracy: 0.1970 - val_loss: 2.2138 - val_accuracy: 0.2143 - lr: 5.0000e-04
Epoch 6/150
21/21 [==============================] - ETA: 0s - loss: 2.2231 - accuracy: 0.1970
Epoch 6: val_loss improved from 2.21383 to 2.13433, saving model to video_classifier.h5
21/21 [==============================] - 8s 383ms/step - loss: 2.2231 - accuracy: 0.1970 - val_loss: 2.1343 - val_accuracy: 0.2143 - lr: 5.0000e-04
Epoch 7/150
21/21 [==============================] - ETA: 0s - loss: 2.1209 - accuracy: 0.2075
Epoch 7: val_loss improved from 2.13433 to 1.97499, saving model to video_classifier.h5
21/21 [==============================] - 8s 391ms/step - loss: 2.1209 - accuracy: 0.2075 - val_loss: 1.9750 - val_accuracy: 0.3259 - lr: 5.0000e-04
Epoch 8/150
21/21 [==============================] - ETA: 0s - loss: 2.1002 - accuracy: 0.2149
Epoch 8: val_loss did not improve from 1.97499
21/21 [==============================] - 8s 373ms/step - loss: 2.1002 - accuracy: 0.2149 - val_loss: 2.1330 - val_accuracy: 0.2411 - lr: 5.0000e-04
Epoch 9/150
21/21 [==============================] - ETA: 0s - loss: 2.0560 - accuracy: 0.2463
Epoch 9: val_loss improved from 1.97499 to 1.90246, saving model to video_classifier.h5
21/21 [==============================] - 7s 354ms/step - loss: 2.0560 - accuracy: 0.2463 - val_loss: 1.9025 - val_accuracy: 0.3348 - lr: 5.0000e-04
Epoch 10/150
21/21 [==============================] - ETA: 0s - loss: 1.9461 - accuracy: 0.2910
Epoch 10: val_loss improved from 1.90246 to 1.77267, saving model to video_classifier.h5
21/21 [==============================] - 8s 362ms/step - loss: 1.9461 - accuracy: 0.2910 - val_loss: 1.7727 - val_accuracy: 0.3393 - lr: 5.0000e-04
Epoch 11/150
21/21 [==============================] - ETA: 0s - loss: 1.7881 - accuracy: 0.3284
Epoch 11: val_loss improved from 1.77267 to 1.57947, saving model to video_classifier.h5
21/21 [==============================] - 8s 362ms/step - loss: 1.7881 - accuracy: 0.3284 - val_loss: 1.5795 - val_accuracy: 0.3884 - lr: 5.0000e-04
Epoch 12/150
21/21 [==============================] - ETA: 0s - loss: 1.6665 - accuracy: 0.3612
Epoch 12: val_loss improved from 1.57947 to 1.44990, saving model to video_classifier.h5
21/21 [==============================] - 8s 367ms/step - loss: 1.6665 - accuracy: 0.3612 - val_loss: 1.4499 - val_accuracy: 0.4643 - lr: 5.0000e-04
Epoch 13/150
21/21 [==============================] - ETA: 0s - loss: 1.5835 - accuracy: 0.3776
Epoch 13: val_loss improved from 1.44990 to 1.37599, saving model to video_classifier.h5
21/21 [==============================] - 8s 368ms/step - loss: 1.5835 - accuracy: 0.3776 - val_loss: 1.3760 - val_accuracy: 0.4911 - lr: 5.0000e-04
Epoch 14/150
21/21 [==============================] - ETA: 0s - loss: 1.4194 - accuracy: 0.4313
Epoch 14: val_loss improved from 1.37599 to 1.28251, saving model to video_classifier.h5
21/21 [==============================] - 7s 355ms/step - loss: 1.4194 - accuracy: 0.4313 - val_loss: 1.2825 - val_accuracy: 0.5625 - lr: 5.0000e-04
Epoch 15/150
21/21 [==============================] - ETA: 0s - loss: 1.5863 - accuracy: 0.4015
Epoch 15: val_loss did not improve from 1.28251
21/21 [==============================] - 7s 349ms/step - loss: 1.5863 - accuracy: 0.4015 - val_loss: 1.3830 - val_accuracy: 0.4821 - lr: 5.0000e-04
Epoch 16/150
21/21 [==============================] - ETA: 0s - loss: 1.4913 - accuracy: 0.4373
Epoch 16: val_loss improved from 1.28251 to 1.22984, saving model to video_classifier.h5
21/21 [==============================] - 8s 364ms/step - loss: 1.4913 - accuracy: 0.4373 - val_loss: 1.2298 - val_accuracy: 0.5759 - lr: 5.0000e-04
Epoch 17/150
21/21 [==============================] - ETA: 0s - loss: 1.2958 - accuracy: 0.5090
Epoch 17: val_loss improved from 1.22984 to 1.12092, saving model to video_classifier.h5
21/21 [==============================] - 7s 358ms/step - loss: 1.2958 - accuracy: 0.5090 - val_loss: 1.1209 - val_accuracy: 0.6161 - lr: 5.0000e-04
Epoch 18/150
21/21 [==============================] - ETA: 0s - loss: 1.1247 - accuracy: 0.5612
Epoch 18: val_loss improved from 1.12092 to 0.95124, saving model to video_classifier.h5
21/21 [==============================] - 8s 372ms/step - loss: 1.1247 - accuracy: 0.5612 - val_loss: 0.9512 - val_accuracy: 0.6473 - lr: 5.0000e-04
Epoch 19/150
21/21 [==============================] - ETA: 0s - loss: 1.0542 - accuracy: 0.5881
Epoch 19: val_loss improved from 0.95124 to 0.88986, saving model to video_classifier.h5
21/21 [==============================] - 8s 376ms/step - loss: 1.0542 - accuracy: 0.5881 - val_loss: 0.8899 - val_accuracy: 0.6384 - lr: 5.0000e-04
Epoch 20/150
21/21 [==============================] - ETA: 0s - loss: 0.9117 - accuracy: 0.6448
Epoch 20: val_loss improved from 0.88986 to 0.72669, saving model to video_classifier.h5
21/21 [==============================] - 8s 376ms/step - loss: 0.9117 - accuracy: 0.6448 - val_loss: 0.7267 - val_accuracy: 0.7411 - lr: 5.0000e-04
Epoch 21/150
21/21 [==============================] - ETA: 0s - loss: 0.8571 - accuracy: 0.6582
Epoch 21: val_loss improved from 0.72669 to 0.58782, saving model to video_classifier.h5
21/21 [==============================] - 8s 381ms/step - loss: 0.8571 - accuracy: 0.6582 - val_loss: 0.5878 - val_accuracy: 0.7946 - lr: 5.0000e-04
Epoch 22/150
21/21 [==============================] - ETA: 0s - loss: 0.6869 - accuracy: 0.7119
Epoch 22: val_loss did not improve from 0.58782
21/21 [==============================] - 8s 368ms/step - loss: 0.6869 - accuracy: 0.7119 - val_loss: 0.7148 - val_accuracy: 0.7768 - lr: 5.0000e-04
Epoch 23/150
21/21 [==============================] - ETA: 0s - loss: 0.7359 - accuracy: 0.7209
Epoch 23: val_loss improved from 0.58782 to 0.50223, saving model to video_classifier.h5
21/21 [==============================] - 8s 380ms/step - loss: 0.7359 - accuracy: 0.7209 - val_loss: 0.5022 - val_accuracy: 0.8080 - lr: 5.0000e-04
Epoch 24/150
21/21 [==============================] - ETA: 0s - loss: 0.8604 - accuracy: 0.7015
Epoch 24: val_loss did not improve from 0.50223
21/21 [==============================] - 8s 383ms/step - loss: 0.8604 - accuracy: 0.7015 - val_loss: 0.8538 - val_accuracy: 0.6652 - lr: 5.0000e-04
Epoch 25/150
21/21 [==============================] - ETA: 0s - loss: 0.7952 - accuracy: 0.7239
Epoch 25: val_loss did not improve from 0.50223
21/21 [==============================] - 8s 383ms/step - loss: 0.7952 - accuracy: 0.7239 - val_loss: 0.5878 - val_accuracy: 0.7723 - lr: 5.0000e-04
Epoch 26/150
21/21 [==============================] - ETA: 0s - loss: 0.6332 - accuracy: 0.7582
Epoch 26: val_loss did not improve from 0.50223
21/21 [==============================] - 8s 362ms/step - loss: 0.6332 - accuracy: 0.7582 - val_loss: 0.6044 - val_accuracy: 0.8080 - lr: 5.0000e-04
Epoch 27/150
21/21 [==============================] - ETA: 0s - loss: 0.5249 - accuracy: 0.8000
Epoch 27: val_loss improved from 0.50223 to 0.35592, saving model to video_classifier.h5
21/21 [==============================] - 8s 395ms/step - loss: 0.5249 - accuracy: 0.8000 - val_loss: 0.3559 - val_accuracy: 0.8571 - lr: 5.0000e-04
Epoch 28/150
21/21 [==============================] - ETA: 0s - loss: 0.4194 - accuracy: 0.8328
Epoch 28: val_loss did not improve from 0.35592
21/21 [==============================] - 8s 363ms/step - loss: 0.4194 - accuracy: 0.8328 - val_loss: 0.3905 - val_accuracy: 0.8616 - lr: 5.0000e-04
Epoch 29/150
21/21 [==============================] - ETA: 0s - loss: 0.3785 - accuracy: 0.8567
Epoch 29: val_loss did not improve from 0.35592
21/21 [==============================] - 8s 386ms/step - loss: 0.3785 - accuracy: 0.8567 - val_loss: 0.3870 - val_accuracy: 0.8839 - lr: 5.0000e-04
Epoch 30/150
21/21 [==============================] - ETA: 0s - loss: 0.3246 - accuracy: 0.8791
Epoch 30: val_loss did not improve from 0.35592
21/21 [==============================] - 8s 380ms/step - loss: 0.3246 - accuracy: 0.8791 - val_loss: 0.3947 - val_accuracy: 0.8482 - lr: 5.0000e-04
Epoch 31/150
21/21 [==============================] - ETA: 0s - loss: 0.5172 - accuracy: 0.8388
Epoch 31: val_loss did not improve from 0.35592
21/21 [==============================] - 8s 384ms/step - loss: 0.5172 - accuracy: 0.8388 - val_loss: 0.6125 - val_accuracy: 0.8304 - lr: 5.0000e-04
Epoch 32/150
21/21 [==============================] - ETA: 0s - loss: 0.4220 - accuracy: 0.8507
Epoch 32: val_loss improved from 0.35592 to 0.26445, saving model to video_classifier.h5
21/21 [==============================] - 8s 374ms/step - loss: 0.4220 - accuracy: 0.8507 - val_loss: 0.2645 - val_accuracy: 0.8929 - lr: 5.0000e-04
Epoch 33/150
21/21 [==============================] - ETA: 0s - loss: 0.3732 - accuracy: 0.8851
Epoch 33: val_loss did not improve from 0.26445
21/21 [==============================] - 7s 347ms/step - loss: 0.3732 - accuracy: 0.8851 - val_loss: 0.8684 - val_accuracy: 0.7991 - lr: 5.0000e-04
Epoch 34/150
21/21 [==============================] - ETA: 0s - loss: 0.3653 - accuracy: 0.8537
Epoch 34: val_loss did not improve from 0.26445
21/21 [==============================] - 7s 353ms/step - loss: 0.3653 - accuracy: 0.8537 - val_loss: 0.3014 - val_accuracy: 0.9286 - lr: 5.0000e-04
Epoch 35/150
21/21 [==============================] - ETA: 0s - loss: 0.5128 - accuracy: 0.8597
Epoch 35: val_loss did not improve from 0.26445
21/21 [==============================] - 8s 376ms/step - loss: 0.5128 - accuracy: 0.8597 - val_loss: 0.3647 - val_accuracy: 0.8929 - lr: 5.0000e-04
Epoch 36/150
21/21 [==============================] - ETA: 0s - loss: 0.5249 - accuracy: 0.8179
Epoch 36: val_loss did not improve from 0.26445
21/21 [==============================] - 8s 379ms/step - loss: 0.5249 - accuracy: 0.8179 - val_loss: 0.4021 - val_accuracy: 0.8705 - lr: 5.0000e-04
Epoch 37/150
21/21 [==============================] - ETA: 0s - loss: 0.2971 - accuracy: 0.8970
Epoch 37: val_loss improved from 0.26445 to 0.21931, saving model to video_classifier.h5
21/21 [==============================] - 8s 393ms/step - loss: 0.2971 - accuracy: 0.8970 - val_loss: 0.2193 - val_accuracy: 0.9375 - lr: 5.0000e-04
Epoch 38/150
21/21 [==============================] - ETA: 0s - loss: 0.2033 - accuracy: 0.9373
Epoch 38: val_loss improved from 0.21931 to 0.19596, saving model to video_classifier.h5
21/21 [==============================] - 8s 383ms/step - loss: 0.2033 - accuracy: 0.9373 - val_loss: 0.1960 - val_accuracy: 0.9420 - lr: 5.0000e-04
Epoch 39/150
21/21 [==============================] - ETA: 0s - loss: 0.1579 - accuracy: 0.9418
Epoch 39: val_loss improved from 0.19596 to 0.15910, saving model to video_classifier.h5
21/21 [==============================] - 8s 377ms/step - loss: 0.1579 - accuracy: 0.9418 - val_loss: 0.1591 - val_accuracy: 0.9643 - lr: 5.0000e-04
Epoch 40/150
21/21 [==============================] - ETA: 0s - loss: 0.1467 - accuracy: 0.9597
Epoch 40: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 373ms/step - loss: 0.1467 - accuracy: 0.9597 - val_loss: 0.2291 - val_accuracy: 0.9464 - lr: 5.0000e-04
Epoch 41/150
21/21 [==============================] - ETA: 0s - loss: 0.1791 - accuracy: 0.9478
Epoch 41: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 382ms/step - loss: 0.1791 - accuracy: 0.9478 - val_loss: 0.2765 - val_accuracy: 0.9464 - lr: 5.0000e-04
Epoch 42/150
21/21 [==============================] - ETA: 0s - loss: 0.1389 - accuracy: 0.9612
Epoch 42: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 370ms/step - loss: 0.1389 - accuracy: 0.9612 - val_loss: 0.3172 - val_accuracy: 0.9152 - lr: 5.0000e-04
Epoch 43/150
21/21 [==============================] - ETA: 0s - loss: 0.1349 - accuracy: 0.9567
Epoch 43: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 375ms/step - loss: 0.1349 - accuracy: 0.9567 - val_loss: 0.2837 - val_accuracy: 0.9598 - lr: 5.0000e-04
Epoch 44/150
21/21 [==============================] - ETA: 0s - loss: 0.0795 - accuracy: 0.9821
Epoch 44: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 369ms/step - loss: 0.0795 - accuracy: 0.9821 - val_loss: 0.2354 - val_accuracy: 0.9643 - lr: 5.0000e-04
Epoch 45/150
21/21 [==============================] - ETA: 0s - loss: 0.1032 - accuracy: 0.9761
Epoch 45: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 367ms/step - loss: 0.1032 - accuracy: 0.9761 - val_loss: 0.2441 - val_accuracy: 0.9643 - lr: 5.0000e-04
Epoch 46/150
21/21 [==============================] - ETA: 0s - loss: 0.1502 - accuracy: 0.9597
Epoch 46: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 374ms/step - loss: 0.1502 - accuracy: 0.9597 - val_loss: 0.4816 - val_accuracy: 0.9286 - lr: 5.0000e-04
Epoch 47/150
21/21 [==============================] - ETA: 0s - loss: 0.2629 - accuracy: 0.9522
Epoch 47: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 371ms/step - loss: 0.2629 - accuracy: 0.9522 - val_loss: 1.4639 - val_accuracy: 0.8170 - lr: 5.0000e-04
Epoch 48/150
21/21 [==============================] - ETA: 0s - loss: 0.5310 - accuracy: 0.8716
Epoch 48: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 382ms/step - loss: 0.5310 - accuracy: 0.8716 - val_loss: 0.2919 - val_accuracy: 0.9196 - lr: 5.0000e-04
Epoch 49/150
21/21 [==============================] - ETA: 0s - loss: 0.4053 - accuracy: 0.8910
Epoch 49: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 380ms/step - loss: 0.4053 - accuracy: 0.8910 - val_loss: 0.3633 - val_accuracy: 0.9107 - lr: 5.0000e-04
Epoch 50/150
21/21 [==============================] - ETA: 0s - loss: 0.1824 - accuracy: 0.9448
Epoch 50: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 369ms/step - loss: 0.1824 - accuracy: 0.9448 - val_loss: 0.2838 - val_accuracy: 0.9464 - lr: 2.5000e-04
Epoch 51/150
21/21 [==============================] - ETA: 0s - loss: 0.0969 - accuracy: 0.9791
Epoch 51: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 371ms/step - loss: 0.0969 - accuracy: 0.9791 - val_loss: 0.2146 - val_accuracy: 0.9554 - lr: 2.5000e-04
Epoch 52/150
21/21 [==============================] - ETA: 0s - loss: 0.0584 - accuracy: 0.9925
Epoch 52: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 362ms/step - loss: 0.0584 - accuracy: 0.9925 - val_loss: 0.2057 - val_accuracy: 0.9688 - lr: 2.5000e-04
Epoch 53/150
21/21 [==============================] - ETA: 0s - loss: 0.0499 - accuracy: 0.9896
Epoch 53: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 379ms/step - loss: 0.0499 - accuracy: 0.9896 - val_loss: 0.2031 - val_accuracy: 0.9643 - lr: 2.5000e-04
Epoch 54/150
21/21 [==============================] - ETA: 0s - loss: 0.0368 - accuracy: 0.9940
Epoch 54: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 373ms/step - loss: 0.0368 - accuracy: 0.9940 - val_loss: 0.2148 - val_accuracy: 0.9732 - lr: 2.5000e-04
Epoch 55/150
21/21 [==============================] - ETA: 0s - loss: 0.0505 - accuracy: 0.9866
Epoch 55: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 366ms/step - loss: 0.0505 - accuracy: 0.9866 - val_loss: 0.2190 - val_accuracy: 0.9688 - lr: 2.5000e-04
Epoch 56/150
21/21 [==============================] - ETA: 0s - loss: 0.0367 - accuracy: 0.9910
Epoch 56: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 373ms/step - loss: 0.0367 - accuracy: 0.9910 - val_loss: 0.2302 - val_accuracy: 0.9688 - lr: 2.5000e-04
Epoch 57/150
21/21 [==============================] - ETA: 0s - loss: 0.0402 - accuracy: 0.9881
Epoch 57: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 361ms/step - loss: 0.0402 - accuracy: 0.9881 - val_loss: 0.2482 - val_accuracy: 0.9732 - lr: 2.5000e-04
Epoch 58/150
21/21 [==============================] - ETA: 0s - loss: 0.0253 - accuracy: 0.9925
Epoch 58: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 376ms/step - loss: 0.0253 - accuracy: 0.9925 - val_loss: 0.2731 - val_accuracy: 0.9688 - lr: 2.5000e-04
Epoch 59/150
21/21 [==============================] - ETA: 0s - loss: 0.0420 - accuracy: 0.9896
Epoch 59: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 371ms/step - loss: 0.0420 - accuracy: 0.9896 - val_loss: 0.2724 - val_accuracy: 0.9688 - lr: 2.5000e-04
Epoch 60/150
21/21 [==============================] - ETA: 0s - loss: 0.0252 - accuracy: 0.9940
Epoch 60: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 364ms/step - loss: 0.0252 - accuracy: 0.9940 - val_loss: 0.2719 - val_accuracy: 0.9688 - lr: 1.2500e-04
Epoch 61/150
21/21 [==============================] - ETA: 0s - loss: 0.0322 - accuracy: 0.9925
Epoch 61: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 380ms/step - loss: 0.0322 - accuracy: 0.9925 - val_loss: 0.2693 - val_accuracy: 0.9688 - lr: 1.2500e-04
Epoch 62/150
21/21 [==============================] - ETA: 0s - loss: 0.0313 - accuracy: 0.9940
Epoch 62: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 366ms/step - loss: 0.0313 - accuracy: 0.9940 - val_loss: 0.2761 - val_accuracy: 0.9732 - lr: 1.2500e-04
Epoch 63/150
21/21 [==============================] - ETA: 0s - loss: 0.0263 - accuracy: 0.9955
Epoch 63: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 356ms/step - loss: 0.0263 - accuracy: 0.9955 - val_loss: 0.2729 - val_accuracy: 0.9732 - lr: 1.2500e-04
Epoch 64/150
21/21 [==============================] - ETA: 0s - loss: 0.0161 - accuracy: 0.9985
Epoch 64: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 379ms/step - loss: 0.0161 - accuracy: 0.9985 - val_loss: 0.2704 - val_accuracy: 0.9732 - lr: 1.2500e-04
Epoch 65/150
21/21 [==============================] - ETA: 0s - loss: 0.0257 - accuracy: 0.9955
Epoch 65: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 371ms/step - loss: 0.0257 - accuracy: 0.9955 - val_loss: 0.2780 - val_accuracy: 0.9688 - lr: 1.2500e-04
Epoch 66/150
21/21 [==============================] - ETA: 0s - loss: 0.0212 - accuracy: 0.9955
Epoch 66: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 379ms/step - loss: 0.0212 - accuracy: 0.9955 - val_loss: 0.2767 - val_accuracy: 0.9688 - lr: 1.2500e-04
Epoch 67/150
21/21 [==============================] - ETA: 0s - loss: 0.0199 - accuracy: 0.9955
Epoch 67: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 380ms/step - loss: 0.0199 - accuracy: 0.9955 - val_loss: 0.2824 - val_accuracy: 0.9688 - lr: 1.2500e-04
Epoch 68/150
21/21 [==============================] - ETA: 0s - loss: 0.0190 - accuracy: 0.9955
Epoch 68: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 377ms/step - loss: 0.0190 - accuracy: 0.9955 - val_loss: 0.2845 - val_accuracy: 0.9688 - lr: 1.2500e-04
Epoch 69/150
21/21 [==============================] - ETA: 0s - loss: 0.0230 - accuracy: 0.9940
Epoch 69: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 379ms/step - loss: 0.0230 - accuracy: 0.9940 - val_loss: 0.2723 - val_accuracy: 0.9688 - lr: 1.2500e-04
Epoch 70/150
21/21 [==============================] - ETA: 0s - loss: 0.0203 - accuracy: 0.9925
Epoch 70: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 380ms/step - loss: 0.0203 - accuracy: 0.9925 - val_loss: 0.2736 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 71/150
21/21 [==============================] - ETA: 0s - loss: 0.0199 - accuracy: 0.9925
Epoch 71: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 381ms/step - loss: 0.0199 - accuracy: 0.9925 - val_loss: 0.2859 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 72/150
21/21 [==============================] - ETA: 0s - loss: 0.0151 - accuracy: 0.9970
Epoch 72: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 367ms/step - loss: 0.0151 - accuracy: 0.9970 - val_loss: 0.2909 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 73/150
21/21 [==============================] - ETA: 0s - loss: 0.0161 - accuracy: 0.9940
Epoch 73: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 371ms/step - loss: 0.0161 - accuracy: 0.9940 - val_loss: 0.2940 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 74/150
21/21 [==============================] - ETA: 0s - loss: 0.0229 - accuracy: 0.9940
Epoch 74: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 384ms/step - loss: 0.0229 - accuracy: 0.9940 - val_loss: 0.2991 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 75/150
21/21 [==============================] - ETA: 0s - loss: 0.0110 - accuracy: 1.0000
Epoch 75: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 376ms/step - loss: 0.0110 - accuracy: 1.0000 - val_loss: 0.3037 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 76/150
21/21 [==============================] - ETA: 0s - loss: 0.0292 - accuracy: 0.9925
Epoch 76: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 383ms/step - loss: 0.0292 - accuracy: 0.9925 - val_loss: 0.3041 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 77/150
21/21 [==============================] - ETA: 0s - loss: 0.0147 - accuracy: 0.9970
Epoch 77: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 382ms/step - loss: 0.0147 - accuracy: 0.9970 - val_loss: 0.3028 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 78/150
21/21 [==============================] - ETA: 0s - loss: 0.0232 - accuracy: 0.9940
Epoch 78: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 387ms/step - loss: 0.0232 - accuracy: 0.9940 - val_loss: 0.3100 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 79/150
21/21 [==============================] - ETA: 0s - loss: 0.0153 - accuracy: 0.9970
Epoch 79: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 374ms/step - loss: 0.0153 - accuracy: 0.9970 - val_loss: 0.3160 - val_accuracy: 0.9688 - lr: 6.2500e-05
Epoch 80/150
21/21 [==============================] - ETA: 0s - loss: 0.0140 - accuracy: 0.9955
Epoch 80: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 375ms/step - loss: 0.0140 - accuracy: 0.9955 - val_loss: 0.3179 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 81/150
21/21 [==============================] - ETA: 0s - loss: 0.0186 - accuracy: 0.9940
Epoch 81: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 370ms/step - loss: 0.0186 - accuracy: 0.9940 - val_loss: 0.3173 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 82/150
21/21 [==============================] - ETA: 0s - loss: 0.0167 - accuracy: 0.9955
Epoch 82: val_loss did not improve from 0.15910
21/21 [==============================] - 9s 442ms/step - loss: 0.0167 - accuracy: 0.9955 - val_loss: 0.3147 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 83/150
21/21 [==============================] - ETA: 0s - loss: 0.0193 - accuracy: 0.9955
Epoch 83: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 358ms/step - loss: 0.0193 - accuracy: 0.9955 - val_loss: 0.3138 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 84/150
21/21 [==============================] - ETA: 0s - loss: 0.0156 - accuracy: 0.9985
Epoch 84: val_loss did not improve from 0.15910
21/21 [==============================] - 9s 411ms/step - loss: 0.0156 - accuracy: 0.9985 - val_loss: 0.3097 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 85/150
21/21 [==============================] - ETA: 0s - loss: 0.0168 - accuracy: 0.9940
Epoch 85: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 386ms/step - loss: 0.0168 - accuracy: 0.9940 - val_loss: 0.3076 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 86/150
21/21 [==============================] - ETA: 0s - loss: 0.0143 - accuracy: 0.9940
Epoch 86: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 365ms/step - loss: 0.0143 - accuracy: 0.9940 - val_loss: 0.3071 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 87/150
21/21 [==============================] - ETA: 0s - loss: 0.0158 - accuracy: 0.9970
Epoch 87: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 385ms/step - loss: 0.0158 - accuracy: 0.9970 - val_loss: 0.3020 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 88/150
21/21 [==============================] - ETA: 0s - loss: 0.0149 - accuracy: 0.9985
Epoch 88: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 376ms/step - loss: 0.0149 - accuracy: 0.9985 - val_loss: 0.3047 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 89/150
21/21 [==============================] - ETA: 0s - loss: 0.0121 - accuracy: 0.9985
Epoch 89: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 378ms/step - loss: 0.0121 - accuracy: 0.9985 - val_loss: 0.3050 - val_accuracy: 0.9688 - lr: 3.1250e-05
Epoch 90/150
21/21 [==============================] - ETA: 0s - loss: 0.0202 - accuracy: 0.9925
Epoch 90: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 375ms/step - loss: 0.0202 - accuracy: 0.9925 - val_loss: 0.3042 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 91/150
21/21 [==============================] - ETA: 0s - loss: 0.0129 - accuracy: 1.0000
Epoch 91: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 363ms/step - loss: 0.0129 - accuracy: 1.0000 - val_loss: 0.3052 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 92/150
21/21 [==============================] - ETA: 0s - loss: 0.0176 - accuracy: 0.9925
Epoch 92: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 380ms/step - loss: 0.0176 - accuracy: 0.9925 - val_loss: 0.3055 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 93/150
21/21 [==============================] - ETA: 0s - loss: 0.0242 - accuracy: 0.9925
Epoch 93: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 345ms/step - loss: 0.0242 - accuracy: 0.9925 - val_loss: 0.3049 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 94/150
21/21 [==============================] - ETA: 0s - loss: 0.0126 - accuracy: 0.9985
Epoch 94: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 369ms/step - loss: 0.0126 - accuracy: 0.9985 - val_loss: 0.3062 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 95/150
21/21 [==============================] - ETA: 0s - loss: 0.0134 - accuracy: 0.9940
Epoch 95: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 392ms/step - loss: 0.0134 - accuracy: 0.9940 - val_loss: 0.3076 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 96/150
21/21 [==============================] - ETA: 0s - loss: 0.0231 - accuracy: 0.9925
Epoch 96: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 382ms/step - loss: 0.0231 - accuracy: 0.9925 - val_loss: 0.3104 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 97/150
21/21 [==============================] - ETA: 0s - loss: 0.0157 - accuracy: 0.9955
Epoch 97: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 377ms/step - loss: 0.0157 - accuracy: 0.9955 - val_loss: 0.3106 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 98/150
21/21 [==============================] - ETA: 0s - loss: 0.0147 - accuracy: 0.9940
Epoch 98: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 365ms/step - loss: 0.0147 - accuracy: 0.9940 - val_loss: 0.3114 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 99/150
21/21 [==============================] - ETA: 0s - loss: 0.0169 - accuracy: 0.9955
Epoch 99: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 353ms/step - loss: 0.0169 - accuracy: 0.9955 - val_loss: 0.3115 - val_accuracy: 0.9688 - lr: 1.5625e-05
Epoch 100/150
21/21 [==============================] - ETA: 0s - loss: 0.0102 - accuracy: 0.9970
Epoch 100: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 388ms/step - loss: 0.0102 - accuracy: 0.9970 - val_loss: 0.3116 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 101/150
21/21 [==============================] - ETA: 0s - loss: 0.0139 - accuracy: 0.9970
Epoch 101: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 388ms/step - loss: 0.0139 - accuracy: 0.9970 - val_loss: 0.3111 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 102/150
21/21 [==============================] - ETA: 0s - loss: 0.0107 - accuracy: 0.9970
Epoch 102: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 379ms/step - loss: 0.0107 - accuracy: 0.9970 - val_loss: 0.3103 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 103/150
21/21 [==============================] - ETA: 0s - loss: 0.0107 - accuracy: 0.9970
Epoch 103: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 384ms/step - loss: 0.0107 - accuracy: 0.9970 - val_loss: 0.3105 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 104/150
21/21 [==============================] - ETA: 0s - loss: 0.0202 - accuracy: 0.9925
Epoch 104: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 369ms/step - loss: 0.0202 - accuracy: 0.9925 - val_loss: 0.3050 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 105/150
21/21 [==============================] - ETA: 0s - loss: 0.0104 - accuracy: 1.0000
Epoch 105: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 378ms/step - loss: 0.0104 - accuracy: 1.0000 - val_loss: 0.3047 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 106/150
21/21 [==============================] - ETA: 0s - loss: 0.0125 - accuracy: 0.9970
Epoch 106: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 379ms/step - loss: 0.0125 - accuracy: 0.9970 - val_loss: 0.3054 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 107/150
21/21 [==============================] - ETA: 0s - loss: 0.0198 - accuracy: 0.9955
Epoch 107: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 388ms/step - loss: 0.0198 - accuracy: 0.9955 - val_loss: 0.3081 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 108/150
21/21 [==============================] - ETA: 0s - loss: 0.0187 - accuracy: 0.9940
Epoch 108: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 377ms/step - loss: 0.0187 - accuracy: 0.9940 - val_loss: 0.3103 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 109/150
21/21 [==============================] - ETA: 0s - loss: 0.0098 - accuracy: 1.0000
Epoch 109: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 373ms/step - loss: 0.0098 - accuracy: 1.0000 - val_loss: 0.3118 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 110/150
21/21 [==============================] - ETA: 0s - loss: 0.0118 - accuracy: 0.9985
Epoch 110: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 377ms/step - loss: 0.0118 - accuracy: 0.9985 - val_loss: 0.3119 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 111/150
21/21 [==============================] - ETA: 0s - loss: 0.0137 - accuracy: 0.9955
Epoch 111: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 347ms/step - loss: 0.0137 - accuracy: 0.9955 - val_loss: 0.3112 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 112/150
21/21 [==============================] - ETA: 0s - loss: 0.0129 - accuracy: 0.9985
Epoch 112: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 353ms/step - loss: 0.0129 - accuracy: 0.9985 - val_loss: 0.3113 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 113/150
21/21 [==============================] - ETA: 0s - loss: 0.0154 - accuracy: 0.9940
Epoch 113: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 340ms/step - loss: 0.0154 - accuracy: 0.9940 - val_loss: 0.3108 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 114/150
21/21 [==============================] - ETA: 0s - loss: 0.0275 - accuracy: 0.9940
Epoch 114: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 376ms/step - loss: 0.0275 - accuracy: 0.9940 - val_loss: 0.3108 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 115/150
21/21 [==============================] - ETA: 0s - loss: 0.0155 - accuracy: 0.9970
Epoch 115: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 378ms/step - loss: 0.0155 - accuracy: 0.9970 - val_loss: 0.3098 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 116/150
21/21 [==============================] - ETA: 0s - loss: 0.0128 - accuracy: 0.9955
Epoch 116: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 375ms/step - loss: 0.0128 - accuracy: 0.9955 - val_loss: 0.3088 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 117/150
21/21 [==============================] - ETA: 0s - loss: 0.0097 - accuracy: 0.9985
Epoch 117: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 346ms/step - loss: 0.0097 - accuracy: 0.9985 - val_loss: 0.3089 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 118/150
21/21 [==============================] - ETA: 0s - loss: 0.0158 - accuracy: 0.9940
Epoch 118: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 338ms/step - loss: 0.0158 - accuracy: 0.9940 - val_loss: 0.3082 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 119/150
21/21 [==============================] - ETA: 0s - loss: 0.0244 - accuracy: 0.9925
Epoch 119: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 344ms/step - loss: 0.0244 - accuracy: 0.9925 - val_loss: 0.3101 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 120/150
21/21 [==============================] - ETA: 0s - loss: 0.0103 - accuracy: 0.9985
Epoch 120: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 370ms/step - loss: 0.0103 - accuracy: 0.9985 - val_loss: 0.3100 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 121/150
21/21 [==============================] - ETA: 0s - loss: 0.0121 - accuracy: 0.9970
Epoch 121: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 333ms/step - loss: 0.0121 - accuracy: 0.9970 - val_loss: 0.3095 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 122/150
21/21 [==============================] - ETA: 0s - loss: 0.0102 - accuracy: 1.0000
Epoch 122: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 373ms/step - loss: 0.0102 - accuracy: 1.0000 - val_loss: 0.3080 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 123/150
21/21 [==============================] - ETA: 0s - loss: 0.0139 - accuracy: 0.9955
Epoch 123: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 357ms/step - loss: 0.0139 - accuracy: 0.9955 - val_loss: 0.3072 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 124/150
21/21 [==============================] - ETA: 0s - loss: 0.0175 - accuracy: 0.9955
Epoch 124: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 355ms/step - loss: 0.0175 - accuracy: 0.9955 - val_loss: 0.3076 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 125/150
21/21 [==============================] - ETA: 0s - loss: 0.0142 - accuracy: 0.9970
Epoch 125: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 378ms/step - loss: 0.0142 - accuracy: 0.9970 - val_loss: 0.3111 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 126/150
21/21 [==============================] - ETA: 0s - loss: 0.0090 - accuracy: 0.9985
Epoch 126: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 347ms/step - loss: 0.0090 - accuracy: 0.9985 - val_loss: 0.3132 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 127/150
21/21 [==============================] - ETA: 0s - loss: 0.0101 - accuracy: 0.9985
Epoch 127: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 344ms/step - loss: 0.0101 - accuracy: 0.9985 - val_loss: 0.3139 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 128/150
21/21 [==============================] - ETA: 0s - loss: 0.0102 - accuracy: 0.9985
Epoch 128: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 374ms/step - loss: 0.0102 - accuracy: 0.9985 - val_loss: 0.3138 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 129/150
21/21 [==============================] - ETA: 0s - loss: 0.0115 - accuracy: 0.9970
Epoch 129: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 370ms/step - loss: 0.0115 - accuracy: 0.9970 - val_loss: 0.3145 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 130/150
21/21 [==============================] - ETA: 0s - loss: 0.0215 - accuracy: 0.9940
Epoch 130: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 330ms/step - loss: 0.0215 - accuracy: 0.9940 - val_loss: 0.3112 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 131/150
21/21 [==============================] - ETA: 0s - loss: 0.0091 - accuracy: 0.9970
Epoch 131: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 336ms/step - loss: 0.0091 - accuracy: 0.9970 - val_loss: 0.3120 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 132/150
21/21 [==============================] - ETA: 0s - loss: 0.0176 - accuracy: 0.9940
Epoch 132: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 350ms/step - loss: 0.0176 - accuracy: 0.9940 - val_loss: 0.3144 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 133/150
21/21 [==============================] - ETA: 0s - loss: 0.0183 - accuracy: 0.9955
Epoch 133: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 364ms/step - loss: 0.0183 - accuracy: 0.9955 - val_loss: 0.3158 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 134/150
21/21 [==============================] - ETA: 0s - loss: 0.0126 - accuracy: 0.9970
Epoch 134: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 341ms/step - loss: 0.0126 - accuracy: 0.9970 - val_loss: 0.3192 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 135/150
21/21 [==============================] - ETA: 0s - loss: 0.0106 - accuracy: 0.9970
Epoch 135: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 383ms/step - loss: 0.0106 - accuracy: 0.9970 - val_loss: 0.3192 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 136/150
21/21 [==============================] - ETA: 0s - loss: 0.0169 - accuracy: 0.9925
Epoch 136: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 345ms/step - loss: 0.0169 - accuracy: 0.9925 - val_loss: 0.3201 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 137/150
21/21 [==============================] - ETA: 0s - loss: 0.0205 - accuracy: 0.9910
Epoch 137: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 350ms/step - loss: 0.0205 - accuracy: 0.9910 - val_loss: 0.3183 - val_accuracy: 0.9688 - lr: 1.0000e-05
Epoch 138/150
21/21 [==============================] - ETA: 0s - loss: 0.0134 - accuracy: 0.9970
Epoch 138: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 347ms/step - loss: 0.0134 - accuracy: 0.9970 - val_loss: 0.3129 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 139/150
21/21 [==============================] - ETA: 0s - loss: 0.0073 - accuracy: 1.0000
Epoch 139: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 402ms/step - loss: 0.0073 - accuracy: 1.0000 - val_loss: 0.3105 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 140/150
21/21 [==============================] - ETA: 0s - loss: 0.0099 - accuracy: 0.9985
Epoch 140: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 374ms/step - loss: 0.0099 - accuracy: 0.9985 - val_loss: 0.3102 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 141/150
21/21 [==============================] - ETA: 0s - loss: 0.0106 - accuracy: 0.9970
Epoch 141: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 374ms/step - loss: 0.0106 - accuracy: 0.9970 - val_loss: 0.3105 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 142/150
21/21 [==============================] - ETA: 0s - loss: 0.0097 - accuracy: 0.9970    
Epoch 142: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 369ms/step - loss: 0.0097 - accuracy: 0.9970 - val_loss: 0.3105 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 143/150
21/21 [==============================] - ETA: 0s - loss: 0.0204 - accuracy: 0.9925
Epoch 143: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 353ms/step - loss: 0.0204 - accuracy: 0.9925 - val_loss: 0.3133 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 144/150
21/21 [==============================] - ETA: 0s - loss: 0.0169 - accuracy: 0.9925
Epoch 144: val_loss did not improve from 0.15910
21/21 [==============================] - 7s 347ms/step - loss: 0.0169 - accuracy: 0.9925 - val_loss: 0.3199 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 145/150
21/21 [==============================] - ETA: 0s - loss: 0.0110 - accuracy: 0.9985
Epoch 145: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 367ms/step - loss: 0.0110 - accuracy: 0.9985 - val_loss: 0.3209 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 146/150
21/21 [==============================] - ETA: 0s - loss: 0.0121 - accuracy: 0.9970
Epoch 146: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 371ms/step - loss: 0.0121 - accuracy: 0.9970 - val_loss: 0.3207 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 147/150
21/21 [==============================] - ETA: 0s - loss: 0.0111 - accuracy: 0.9985
Epoch 147: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 368ms/step - loss: 0.0111 - accuracy: 0.9985 - val_loss: 0.3204 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 148/150
21/21 [==============================] - ETA: 0s - loss: 0.0090 - accuracy: 0.9985
Epoch 148: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 375ms/step - loss: 0.0090 - accuracy: 0.9985 - val_loss: 0.3213 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 149/150
21/21 [==============================] - ETA: 0s - loss: 0.0124 - accuracy: 0.9955    
Epoch 149: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 368ms/step - loss: 0.0124 - accuracy: 0.9955 - val_loss: 0.3221 - val_accuracy: 0.9732 - lr: 1.0000e-05
Epoch 150/150
21/21 [==============================] - ETA: 0s - loss: 0.0058 - accuracy: 1.0000
Epoch 150: val_loss did not improve from 0.15910
21/21 [==============================] - 8s 381ms/step - loss: 0.0058 - accuracy: 1.0000 - val_loss: 0.3230 - val_accuracy: 0.9732 - lr: 1.0000e-05
Test video label: now
1/1 [==============================] - 1s 521ms/step
Model prediction probabilities:
  now: 100.00%
  like:  0.00%
  book:  0.00%
  chair:  0.00%
  walk:  0.00%
  fine:  0.00%
  before:  0.00%
  cousin:  0.00%
  drink:  0.00%
  deaf:  0.00%
Test video label: drink
1/1 [==============================] - 0s 37ms/step
Model prediction probabilities:
  drink: 100.00%
  deaf:  0.00%
  fine:  0.00%
  cousin:  0.00%
  like:  0.00%
  before:  0.00%
  chair:  0.00%
  walk:  0.00%
  book:  0.00%
  now:  0.00%
Test video label: book
1/1 [==============================] - 0s 40ms/step
Model prediction probabilities:
  book: 100.00%
  like:  0.00%
  before:  0.00%
  now:  0.00%
  chair:  0.00%
  walk:  0.00%
  fine:  0.00%
  cousin:  0.00%
  drink:  0.00%
  deaf:  0.00%
(.venv) PS D:\Documents\Python proj
```
