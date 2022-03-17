# Neural Networks for Regression

### Learning Objectives

* **Linear Regression**&#x20;
* **Cost**
* **Learning Rate**&#x20;
* **Epochs**&#x20;
* **Batch Size**

### What is linear regression? - an example

* Suppose you are thinking of selling your home. And, various houses with diﬀerent sizes (area in sq.ft) around you have sold for diﬀerent prices as listed below:

![](<../.gitbook/assets/8 (3)>)

* And considering, **your home is 3000 square feet**. How much should you sell it for?
* When you **look at the existing price patterns (data) and predict a price** **for your home** that is an example of **linear regression.**
* Here's an easy way to do it. Plotting the 3 data points (information about previous homes) we have so far:

![](https://lh4.googleusercontent.com/qvOA4hbtKF3pbLMqEG9R\_COJpMcPI\_6Zb7H7afps0EzUppbDgpZxmx3NwujKV8i85D01Wo7teF3kT4Tp5k\_fXqs45pbsbFmuRkPTqL68xF\_9mdogWLGlNRlJgCTBtNGeeJYIqh2p-rc)

* Now you can eyeball it and roughly draw a line that gets pretty close to all of these points. Then look at the price shown by the line, where the square footage is 3000:

![](../.gitbook/assets/12)

* Boom! Your home should sell for $260,000.
* That's all! (Well kinda) You plot your data, make a rough line, and use the line to make predictions. You just need to make sure **your line fits** **the data well** (but not too well :)

![](<../.gitbook/assets/15 (2)>)

**But of course we don't want to roughly make a line, we want to compute the exact line that best "fits" our data. That’s where**

### What is linear regression? - to summarize

* Linear regression is a linear model i.e. a model that assumes a **linear** **relationship** (straight-line relationship) between the input variables (x) **** and the single output variable (y).
* When there is a single input variable (x), the method is referred to as **simple linear regression** or just linear regression. For eg. The area of houses in this case.
* When there are multiple input variables, it is often referred to as **multiple linear regression**. For eg. If area of house, neighbourhood, rooms, etc were given.

### Simple Linear Regression

![](<../.gitbook/assets/20 (4)>)

* In case of linear regression, the relationship between the dependant and the independent variable is a simple equation of a straight line (**y = mx + c**)

### Which line is good?

* **Consider this line and the predictions that it makes**

![](https://lh4.googleusercontent.com/uAt5GtGh4QUYu1srMGpETz1XrLOwdqGHBe1qVUdrmCfm3vPMMDzMxAKQegD6UTHJp69wIOxgo2K--wztRd0vHF3itSSP0aK\_t-gshsWCP8r1Nb\_IAYmugsvmElV5qe1IlViJ7jgyvEM)

* **This above drawn line is way oﬀ. For example, according to the line, a 1000 sq foot house should sell for $310,000, whereas we know it actually sold for $200,000.**
* **Here's a better line:**

![](https://lh3.googleusercontent.com/DRtyqH57Lga\_CN7YEhuF2cGJ4zmGizAhWaOQFrAhPY7rOGXwrLF8DsUvtRNSTCcH3JSuOUw0CVM65VqJbWjsOBVDKlI0qrpoNlnFgTH1Om5T9nDrJaFv7ldMkaws4oQ19Dpg5ugSXUQ)

* The **residual** (absolute diﬀerence between the actual and the predicted value) in this case are: $5000, $15000 and $5000.
* This line is an average of **$8,333** dollars oﬀ **** (adding all the distances and dividing by 3).
* This **$8,333** is called the **cost** of using this line.

### Cost

* The **cost** is how far oﬀ the **learned** line is from the **real data**. The best line is the one that is the least oﬀ from the real data.
* To find out what line is the best line (to find the values of coeﬃcients), we need to define and use a cost function.
* In ML, cost functions are used to estimate how badly models are performing.
* Put simply, a cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between X and y.

![](https://lh6.googleusercontent.com/scVcazKuVjxvMbIzy4HklgqANkZTQCgSUq0gcT4uft2A8MuS5vt\_pLQ\_xFZioR\_9R76kyo9jz1JGNjAceUrcgXBhAGMjtC9IkJjNFjw2vtSahYDNLFddR5FhhmR03nAzQhckAu-omO0)

### Cost Function

* Now let’s take a step back and reflect on why we are performing Linear Regression. It is being performed so that when we are finally using our model for prediction, it will predict the value of target variable (or dependent variable) y for the input value (s) of x (even on a new unseen data).
* By achieving the best-fit regression line, the model aims to predict y value such that the **error diﬀerence between predicted value and true value is** **minimum.**
* So, it is very important to update the _m_ and _b_ values, to **reach the best value** **that minimize the error between predicted value (pred)** and **true value (y)** per sample _i_.
* ![](https://lh3.googleusercontent.com/XeXWy2tBwn4UC5vJ2EsZk-AvvvwyHt2LbTsnabiCCNsCwj9WQRRzPoamOgIf1UByzSky-GrPeJqI-9PnVYWpA19A6qckui8dnhLKrLSbZ3bp7EaFnMc5X6oCDeF0Ta5qBzBT6yQ4N58)
* Here, _n_ is the total no. of observations/data points.
* Now let’s have a look at how exactly we got the above formula

![](https://lh4.googleusercontent.com/X00qGx3Cwci9S-bbgiPfJ0kkT4ganf6tIDnbWRV2BZjkcNhRX2bmBWeJ6tO5\_thJ25KtOAcxjaE0SM6vwiojcUu8NoxFPHsr8TMS\_gmNctDcJsKd3zmtLedWKDtzJIp6e3aOv2-Gqjw)

* Here, **pred = mx + c.** Pred can also be termed as the hypothesis function
* The diﬀerence between the model’s **predicted value** and the **true value** is called **RESIDUAL or COST or LOSS**, which needs to be minimized.

### Cost Function - Types

*   There are three primary metrics used to evaluate linear models (to find how well a model is performing):

    * Mean Squared Error
    * Root Mean Squared Error
    * Mean Absolute Error



### Mean Squared Error (MSE)

* MSE is simply the average of the squared diﬀerence between the true target value and the value predicted by the regression model.
* As it squares the diﬀerences, it **penalizes** (gives some penalty or weight for deviating from the objective) **even a small error** which leads to **over-estimation of how bad the model is.**

![](https://lh4.googleusercontent.com/qqKoTV9COESj6pgUEQe0Ujgw-50nCRrGFIPcRTs4lguhTtl-MBWziXJ-tPocibZkV-zOQUfIUOrt2wNhYQnG5yHnM-9fpwujqGZO6WQvfU5l5DdH7QT\_vJ\_Jb26sS74Da6YgKrZVXNg)

### Root Mean Squared Error (RMSE)

* It is just the square root of the mean square error.
* It is preferred more in some cases because the errors are first squared before averaging which poses a high penalty on large errors. This implies that **RMSE is useful when large errors are** **undesired.**

![](https://lh3.googleusercontent.com/VFFbyCztvjy9fAQAzyVvPYEJyr\_11ytxfk31qTPXaqFYrcs2wjBAfmxn8XK4NBhOYbxRyzmqDHXc2FiDMU07m0SG5WafpbwYkE0-OlR6GmkyUd9-gd-kS6fLGldXifPoax5n6pWNLrY)

### Mean Absolute Error(MAE)

* MAE is the absolute diﬀerence between the target value and the value predicted by the model.
* MAE **does not penalize the errors as eﬀectively as mse** making it not suitable for use-cases where you want to pay more attention to the outliers.

![](https://lh5.googleusercontent.com/QvzWm-T0izFsOiww6RoQWMAcHfZEEzCxD\_Z-cMrH6TJrnUjfn90pbJ-h7BUmqomKgCg24Q\_jGCA7-1H9jtyrm4D1ab0n8UNJLgPmG\_h-Bpf36Yr2A\_Fpy30VT2p7GvNWES3kqMnV2L8)

### Objective: Find Optimal Values that provide eﬃcient/effective predictions

![](https://lh4.googleusercontent.com/q9Mhfa\_jV9LcBwKy3zrkDcff8O8mr1XhM8Vr9KY67Qf8fWknBLVqcn75uKspq2pEvTeZ6AZQ1DNVeZHVdj\_EjR2E-XKnAWKKA-bKnU6-oYksiFabi0APXsYKl4oDsBV3pKjBMNwhsH0)

* The values of _m (coeﬃcient of x)_ and _c_ (y intercept) are updated repeatedly until the line fits the data well enough to get the optimal solution
* **y = 1.47 \* x + 0.0296**

**How do we find the optimal values, you ask? Gradient Descent!**

* Gradient Descent (GD) is an optimization technique in the machine learning process which helps us **minimize the** **cost function** by learning the weights of the model such **** that it fits the training data well.

### Gradient Descent - let’s break it down

* **Gradient** = rate of inclination or declination of a slope (**how steep** a **** slope is and in **which direction** it is going) of a function at some point.
* **Descent** = an act of moving downwards.

![](https://lh4.googleusercontent.com/XDiLk5E5p17mMpPUys5oTeq1X8RcGVtzuGx0QXFrCQmGhqR6MMDtTrj3qpcexAWh559ukeDOpzS3kWxjFDJ9-EMKijZuuaNweH1sA9sXF4Ou2iW72Iz0YC3hjuH\_-vzQYaGaeRjiobg)

Simple way to memorize:

* line going up as we move right → positive slope, gradient
* line going down as we move right → negative slope, gradient

### Gradient Descent- A technique to minimize cost

* At the start, the parameter values of the hypothesis are randomly initialized (can be 0)
* Then, Gradient Descent **iteratively** (one step at a time) adjusts the parameters and gradually finds the best values for them by minimizing the cost
* In this case we’ll use it to find the parameters **m** and **c** of the linear regression model
  * **m** = **slope** of a line that we just learned about ****&#x20;
  * **c = bias** : where the line crosses the Y axis.

### Gradient Descent - An Example

* Imagine a person at the top of a mountain who wants to get to the bottom of the valley below the mountain through the fog (he cannot see clearly ahead and only can look around the point he’s standing at)
* He goes down the slope and takes **large steps when the slope is** **steep** and **small steps when the slope is less steep**.
* He decides his next position based on his current position and **stops** **when he gets to the bottom of the valley which was his goal.**

![](https://lh4.googleusercontent.com/5ib\_3wZKHKN2xLte5dYKcyYy0ayYNm5Pj\_KV-VISVEkxmIGAxJNYUxYo0x0XtJePB1Aij\_ZY8i6V\_1VvBb0EQx\_EDgxn9tqmkQQX0Yv57S0aF91boC4W6jR9W1enqy-07lwokO9hshQ)

### Gradient Descent in action

![](https://lh3.googleusercontent.com/J58Zo-DLcenpdXI7jg9CsFhO7h\_F2bpepuKNCjXzvGAI5P\_NEtc3s0YLyL3hx8jfKSEKuT4lZDm26XwL70mE3eZeAnY90Az9OxWja\_2UYMTPHFLExMa5YkHtxm\_kgWsD7iA-n5o8RJw)

![](https://lh4.googleusercontent.com/qqKoTV9COESj6pgUEQe0Ujgw-50nCRrGFIPcRTs4lguhTtl-MBWziXJ-tPocibZkV-zOQUfIUOrt2wNhYQnG5yHnM-9fpwujqGZO6WQvfU5l5DdH7QT\_vJ\_Jb26sS74Da6YgKrZVXNg)

* Now, think of the valley as a cost function. The objective of GD is to minimise the cost function (by updating it’s parameter values). In this case the minimum value would be 0.

### What does a cost function look like?

* **J** is the cost function, plotted along it’s parameter values. Notice the **** crests and troughs.

![](https://lh4.googleusercontent.com/LuRnHH19j7YOW2c80AS3JzOwEjICbL7Gngfjuu3d1XJPXVtAXNY\_4Rols3N6a6Coeh6IGRMPT770J9em1KMCFME34RBRWYbFhRXDULa1\_4Jv1AWY0o-RZiPvVAOAg4V5Vb1AKZGT\_Fo)

![](https://lh4.googleusercontent.com/Znp9SjEL2XZrZw-D3lqlwuG3pto6UPl5aA5bVL8zEiBnCZihKdEnTtRE3r4elkilSsYgNDfoEI7rfUpJ5RieXZjQl-I0AMGbqTYtpDsb9zvJsIBHHFtIzNuQnXMGzXqGkvrhk69Jw90)





### How we navigate the cost function via GD

![](https://lh3.googleusercontent.com/ex-DM68usgGN7qVb10DuWvzeat1hO5V4cv2gV38P\_k\_InSmpkHFar5gTQOHvVkdiIlFUE8K6iZFeGqj8Aytk8nhCmep2ON6Zq10pOKMGwB3yNlw\_ohov4TnA9949KDZYTyztcTLjd5k)

### Hyperparameters

* **Hyperparameters** are parameters which determine the **** model’s performance (or how well we are able to train it)
* These are **parameters outside of the model** which are used to **tune** the model so that it **fits** the training data for the given hypothesis
* We can change the values of these hyperparameters during successive runs of training a model.
* Here you will learn three important hyper parameters:
  * **Learning Rate**
  * **Epoch**
  * **Batch Size**

### Learning Rate

* How does learning rate play a role in getting the optimal weights?
* Learning rate is the rate at which we update (increase or decrease) the values of the parameters (m) during gradient descent

![](https://lh3.googleusercontent.com/J58Zo-DLcenpdXI7jg9CsFhO7h\_F2bpepuKNCjXzvGAI5P\_NEtc3s0YLyL3hx8jfKSEKuT4lZDm26XwL70mE3eZeAnY90Az9OxWja\_2UYMTPHFLExMa5YkHtxm\_kgWsD7iA-n5o8RJw)

* As you can see in the image, **initially the steps are bigger (huge jumps)**
* As the point goes down, the **steps become** **shorter(smaller jumps)**
* When updating weights during GD, if the learning rate is high, the weights will be changed aggressively and when it is low, they’ll be updated slowly

![](https://lh3.googleusercontent.com/L7ccgHg-GJbFbOL9G7xkTAl2X2A3a2MUh4afEBcT1OvlRVU9bG7eW9nppIqRwMP-2MpiKA4zPF0vt\_Emp6YM\_EkaIBg6y5iLqyL1fN\_xgA8zrg48OsOiHqQFC6MYH2QmhoIJ8CBtEQQ)

### Epochs

* **One epoch** is when an **ENTIRE dataset is passed forward and backward through the neural network only ONCE.**
* Usually, training a neural network takes more than a few epochs. It is because just passing the dataset once through the Neural Network is not enough (depends on the problem too).
* We need to pass the full dataset multiple times to the same neural network so that the model is able to **learn** the patterns present in the data
* **Note:** There is no guarantee a network will **converge** or **get better** by letting it learn the data for multiple epochs. It is an art in machine learning to decide the number of epochs suﬃcient for a network.

![](https://lh3.googleusercontent.com/mNmwASuWwKUcGSkLqi9Nkt51TOystHYWNKbeXnSzkUziDs4yHu7Lt\_hRxMhpgSVIaWBSbnnoIUUGUAJAZdtBKDMS-x9JGdY3a0oMIJCaQqaRBkwuDUp6jzW5qQbxePC7cBqzLixem54)

### Batch

* When the data is too big, we can’t pass all the data to the computer at once (think on the order of millions records of data).
* To overcome this problem we need to **divide the data into** **smaller sizes** and **give it to our computer one by one** and **** update the weights of the neural networks at the end of every step.
* **Batch Size** = **Total number of training examples present in a single batch.**
* But What is a **Batch**? As we said, you can’t pass the entire dataset into the neural net at once. So, you divide dataset into **Number of Batches** or sets or parts.

### Batch Size

* Let’s say we have **2000 training examples** that we are going to use .
* We can divide the dataset of 2000 samples into **batches of 500**.
* What is the **Batch Size** here? → **500**
* What will be the **number of batches**? → Simple, 2000/500 = **4**
* Thus, when all the 4 batches (the whole data) pass through the model back and forth once, it’ll complete **1 epoch**.

### The 5 Step Model Life-Cycle

* A model has a life-cycle, and this very simple knowledge provides the backbone for both modeling a dataset and understanding the tf.keras API.
* The five steps in the life-cycle are as follows:
  * Define the model.
  * Compile the model.
  * Fit the model.
  * Make predictions on the test data.
  * Evaluate the model.
* The above 5 steps are explained in the notebook with practical implementation.

Notebook for practice

* [Beginners’ Notebook](https://github.com/dphi-official/Deep\_Learning\_Bootcamp/blob/master/Linear\_Regression/Linear\_Regression\_with\_tf\_keras\_Beginners.ipynb)
* [Intermediate Learners Notebook](https://github.com/dphi-official/Deep\_Learning\_Bootcamp/blob/master/Linear\_Regression/Linear\_Regression\_with\_tf\_keras\_Intermediates.ipynb)

### Overfit - Optimal - Underfit Example

![](https://lh5.googleusercontent.com/IX9C551ATys6xRRsKkUXcEM7Pmt85tXPMgDcdk\_lUeq1Y-FNscRa-JAyn0f\_Tu6LDcGkyi0lGSIzEv6LoakHJwccNOOJoh-zggQFC\_hQ-T9TJepNweNyXPD7Tynulj-3xX0g-AvUiUY)

References

* [https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9) [dfb9c7ce9c9](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

