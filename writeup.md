# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* random_data.py containing the code to generate randomized data from the sample dataset.  
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data
I tried to record my own training data. I find that I am not a good driver. The same model training on my training data is terrible (like it cannot drive straight, drive in circle from the beginning etc). I find that there is a sample training dataset in the symbolic link under the CarND-Behavioral-Cloning-P3/data directory. The result is a much smoother auto driver.
I tried 3 different models on the training data set. All of them failed on the section where there is a dirt patch opening on the road at a sharp turn. I


Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one described in the course. I choose it because it is fast. I implemented the NVIDIA model in parallel and find it takes 10x time to train the model. Therefore, for faster figure out the correct parameter like the correction for left and right camera, I need to use one that is better.
First version:
6,5,5 conv2d
maxpooling
6,5,5 conv2d
maxpooling
dense 120, 84, 1
epoch 5

Second version
6,5,5 conv2d
dropout 0.5
maxpooling
24,5,5 conv2d
maxpooling
dense 120, 84, 1
epoch 3
result is immediately driving out of the road.

Using the NVIDIA version
result is immediately driving out of the road.



In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

### 3. Creation of the Training Set & Training Process

My goal of this project is to learn how different model compares with each other. Therefore, I tried to use one set of data. I found that the course provided dataset in `/carnd_p3/data`. Therefore, I want to use this set of data and experiment with different ways of processing the data to make a good training set.

#### Image size and cropping

First, let's take a look at how the sample training image looks like. Here are the views of the same moment from left, center and right.

<img src="examples/left.png" width="200" alt="children" />
<img src="examples/center.png" width="200" alt="children" />
<img src="examples/right.png" width="200" alt="children" />

I can crop the top 50 pixels and bottom 20 pixels out from the image and still get a good view of the road. This approach works for this specific case because the road is flat. But it won't work as well if the road contains a lot of ups and downs.

Then I read the blog from [Kasper Sakmann](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713). Looks like he is able to achieve the goal by using a 64x64 image.

Using small image has the following benefits:
1. smaller training data -- smaller disk space and faster transfer over the network
2. Faster training.
3. Smaller models

Therefore, I decided to use 64x64 as the image

#### Steering data
Because the nature of the driving simulation, I suspect that the steering data is not well distributed. Therefore, I use the following code to visualize the steering data from the dataset. Actual code is in `calculate_steering_keep_rate()` and `normalize_steering_lines()` in `random_data.py`
```
import matplotlib.pyplot as plt
steerings = lines[:,3].astype(np.float)
plt.hist(steerings, bins=30)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
```
<img src="examples/rawHistogram.png" width="600" alt="" />

It is obvious that there are too many samples near 0 steering. This will make the model more likely to stay straight, thus making it harder to survive a sharp turn.

```
hist, bin_edges= np.histogram(steerings,bins=30)
threshold=np.average(hist)
# get the random keep % from the histogram
keep_rate = []
for i in range(len(hist)):
    if (hist[i]>threshold):
        keep_rate.append(threshold/hist[i])
    else:
        keep_rate.append(1)
keep_rate = np.asarray(keep_rate)

# simulate the new histogram with the keep_rate
# lines = np.ndarray.sort(lines)
lines_keep=[]
for i in range(len(lines)):
    h = -1
    for j in range(len(bin_edges)):
        if (bin_edges[j]>= steerings[i]):
            h = j - 1
            break
    if (h!=-1 and np.random.randint(100000) < keep_rate[h] * 100000):
        lines_keep.append(lines[i])
lines_keep=np.asarray(lines_keep)
print(len(lines_keep))
steerings_keep = lines_keep[:,3].astype(np.float)
plt.hist(steerings_keep, bins=30)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
```
After running the above code, the histogram has a better distribution. 2446 samples out of 8036 original data points are in the output dataset.

<img src="examples/averageHistogram.png" width="600" alt="" />

**other things I have tried**

At first, I use median to calculate the threshold.
```
threshold=np.median(hist)
```
This turns out to be a bad idea because the values are too evenly distributed. Therefore, the car sometimes make too big turn and gets out of the road.
Here is how the distribution looks like, only 346 samples out of 8036 oritinal data points are in the output dataset

<img src="examples/medianHistogram.png" width="600" alt="" />

#### Using left, center and right Images

To increase the amount of data, I utilized the left, center and right images. I use the steering angle calculation from Kasper Sakmann's blog post. Here is his method:
`ignoring perspective distortions one could reason that if the side cameras are about 1.2 meters off-center and the car is supposed to get back to the middle of the road within the next 20 meters the correction to the steering should be about 1.2/20 radians (using tan(𝛼)~𝛼). `

#### Randomize dataset
To make sure the model is generalized enough, I need to randomize the input images.
I use the following methods: flipping, brightness adjustment, affineTransform and cropping.

**Affine Transform **

To simulate different angle of view, I use affine transformation with a target point randomly moved alone the x-axis. See `affineTransform()` in `random_data.py`

**Crop**

To make sure the view of the road is not fixed, I use a random crop to move the cropping region. The image is resized to 64x64. See `crop()` in `random_data.py`

**Flip**

The image is randomly flipped alone the y-axis to make a mirrored view. The steering angle is also mirrored. See `flip()` in `random_data.py`

**Brightness adjustment**
To make the model work with shadows better, the training data brightness are randomly adjusted. See `brightness()` in `random_data.py`

#### Generating training and validation dataset

I use the generator methods described in the course to generate the training and validation set. See `generator()` in `model.py`
