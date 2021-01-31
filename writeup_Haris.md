# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals/steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization_1.png "Visualization"
[image2]: ./examples/visualization_2.png "Visualization"
[image3]: ./examples/visualization_3.png "Visualization"
[image4]: ./examples/rgb_to_gray.png "Grayscaling"
[image5]: ./examples/traffic_sign_1.jpg "Traffic Sign 1"
[image6]: ./examples/traffic_sign_2.jpg "Traffic Sign 2"
[image7]: ./examples/traffic_sign_3.jpg "Traffic Sign 3"
[image8]: ./examples/traffic_sign_4.jpg "Traffic Sign 4"
[image9]: ./examples/traffic_sign_5.jpg "Traffic Sign 5"

---

#### Please refer to file `Traffic_Sign_Classifier.ipynb` for the main code implementation.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32x32x3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. These are histograms for each of the data sets.

![alt text][image1] ![alt text][image2] ![alt text][image3]

### Design and Test a Model Architecture

#### 1. Preprocessing

First, I converted the RGB images to grayscale since we do not really need the color component in training our model. This is implemented in the function `rgb_to_gray()`.

```python
    def rgb_to_gray(images):
        # converting 3D image array to 2D array: for each pixel, taking average of red, green and
        # blue pixel values to get grayscale values
        avg = np.mean(images, axis=3)
        # expanding the shape of the array by adding an axis to the axis location 3
        expanded = np.expand_dims(avg, axis=3)
        return expanded
```
![alt text][image4]

Then I normalized the grayscale dataset as follows. This step allows for the data set to have zero mean and equal variance which will help in modelling.

```python
    X_train_norm = (X_train_gray - 128) / 128
    X_valid_norm = (X_valid_gray - 128) / 128
    X_test_norm = (X_test_gray - 128) / 128
```

#### 2. Model Architecture

My model `LeNet()` consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Image						| 
| Convolution 5x5     	| 1x1 stride, VALID padding, output 28x28x13 	|
| RELU					|												|
| Max Pooling	      	| 2x2 stride, output 14x14x13   				|
| Convolution 5x5	    | 1x1 stride, VALID padding, output 10x10x23 	|
| RELU          		|           									|
| Dropout				| keep rate 0.6									|
| Max Pooling			| 2x2 stride, output 5x5x23     				|
| Flatten				| output 575									|
| Dropout				| keep rate 0.6									|
| Fully Connected		| output 275									|
| RELU					|												|
| Dropout				| keep rate 0.6									|
| Fully Connected		| output 125									|
| RELU					|												|
| Fully Connected		| output 43										|
| Softmax   			|												|


#### 3. Model Training

| Parameter          | Value                                    |
|:------------------:|:----------------------------------------:|
| Epochs             | 30                                       |
| Batch Size         | 128                                      |
| Learning Rate      | 0.001                                    |
| Optimizer          | AdamOptimizer                            |
| Cross Entropy      | `tf.nn.softmax_cross_entropy_with_logits`|

```python
    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
```

#### 4. Solution Approach

My final model results were:
* Training set accuracy of 0.999
* Validation set accuracy of 0.957
* Test set accuracy of 0.955

First, I implemented the original LeNet-5 architecture on normalized RGB images with 10 Epochs to find out that my model was underfitting. Both the training and validation sets had low accuracies. Second, I converted the images to grayscale ending up with a model that was overfitting. Training accuracy improved but validation accuracy was still low.

Third, I introduced `dropout` method in some layers to remove some hidden layer nodes. It was implemented at the following locations.
* 2nd convolution layer
* flatten layer
* 1st fully connected layer

Finally, I increased the depth of convolutional outputs to 13 and 23 respectively so as to have more kernel filters. Also, the number of epochs was increased to 30. These steps resulted in high accuracies for both the training and validation sets.


### Test a Model on New Images

#### 1. Acquiring New Images

Here are five German traffic signs that I found on the web.

```python
    files = glob.glob('./web_images_jpg/*.jpg')
    images = []

    for file in files:
        img = plt.imread(file)
        img = cv2.resize(img, (32,32))
        images.append(img)
    
    new_images = np.array(images)

    fig, axs = plt.subplots(1, 5, figsize=(15,2))
    i = 0
    for img in new_images:
        axs[i].imshow(img)
        i += 1
```

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The images were resized to 32x32x3 for our model to operate on them. The images might be difficult to get classified correctly since some classes have less samples than others hence affecting the effectiveness of the model for some classes/features.

#### 2. Performance on New Images

```python
    ### Pre-processing
    X_new_gray = rgb_to_gray(new_images)
    y_new = [25, 31, 34, 38, 16]
    X_new_norm = (X_new_gray - 128) / 128
```

```python
    with tf.Session() as sess:
        saver.restore(sess, './lenet')
        new_logits = sess.run(logits, feed_dict={x: X_new_norm, keep_rate: 1.0})
        prediction = np.argmax(new_logits, axis=1)
        print('Prediction: ', prediction)
```

```python
    with tf.Session() as sess:
        saver.restore(sess, './lenet')
        new_accuracy = evaluate(X_new_norm, y_new)
        print()
        print('Accuracy = {:.3f}'.format(new_accuracy))
```

Here are the results of the prediction:

| Image			                            |     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Road work      		                    | Road work   									| 
| Wild animals crossing     			    | Wild animals crossing 						|
| Turn left ahead					        | Turn left ahead								|
| Keep right	      		                | Keep right					 				|
| Vehicles over 3.5 metric tons prohibited	| Vehicles over 3.5 metric tons prohibited      |


First, I used PNG images to evaluate the performance of my model and it failed miserable with 0% accuracy. Then I tried the same images but JPG version and it worked. The model was able to correctly classify 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the original test set of 0.955.

#### 3. Model Certainty - Softmax Probabilities

```python
    soft = tf.nn.softmax(logits)
    top5 = tf.nn.top_k(soft, k=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        top = sess.run(top5, feed_dict={x: X_new_norm, keep_rate: 1.0})
        print(top)
```

Softmax probabilities outputs are as follows.

First image...

                [0.05923032,  0.04933028,  0.04568163,  0.04445925,  0.04149671]
                indices = [37, 26,  1, 35,  6]
                
Second image...

                `[0.05258067,  0.04920242,  0.04734491,  0.03760579,  0.0361222]`
                `indices = [1, 36, 26, 37, 35]`

Third image...

                `[0.0536339 ,  0.04639268,  0.0448772 ,  0.03803122,  0.0369914]`
                `indices = [35, 26,  1, 37, 42]`

Fourth image...

                `[0.05563966,  0.04934505,  0.04367107,  0.04169506,  0.0406721]`
                `indices = [1, 26, 37, 36,  6]`

Fifth image...

                `[0.05163704,  0.04603516,  0.0444846 ,  0.04102569,  0.03644843]`
                `indices = [26,  5,  6,  1, 35]`
           