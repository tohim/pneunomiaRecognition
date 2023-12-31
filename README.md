# pneunomiaRecognition
Thomas Himmelstoß, José Trapero, Cornelius Breitenbach, Mohammad Zamani 

Neural & Cognitive Systems Pneunomia Recognition Project --> `ct_img_rec.ipynb`

We tried around with different approaches.
Some can be found in other branches - but only this one in the main is working - the others were more for trying out.
There is also an additional file: `helper.py` which is necessary to run the code in `ct_img_rec.ipynb`. It contains some additional functions.

Our approach: `ct_img_rec.ipynb`

Run with `helper.py` rooted to the same folder for image plotting

1. Creating a class with all basic parameters and hyperparameters for the model.
2. We use the pre-trained CNN "EfficientNet" as a backbone model.
3. Add some additional layers.
4. Do basic transformations.
5. We create a class to handle the training (PneumoniaTrainer) - initializing important variables (like criterion, optimizer, schedular, and empty loss & accuracy lists for train and validation).
6. Also in this class is the function "fit" which handles the model.train() and model.eval() and also saves the model if the avg_valid_loss is smaller or equal to the valid_min_los.
7. As a criterion we use CrossEntropyLoss() --> this is why the output of the last layer is "2" --> Multi-class Classification needs 2 or more outputs.
8. Then we fit the train to the validation data. (expected vs reality)
9. Then plotting the results.
10. At the very end we pick some images out of the test set by ourselves to see what the model predicts.
