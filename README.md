# The Overfitting "Myth"

It is general wisdom that training for too many iterations on your data will lead to a model
*overfitting* to you training data. This then causes worse performance on your evaluation data. This
is a misconception. If your training data and testing data are drawn from the same distribution then
there cannot be overfitting. This obviously holds true if we have an unlimited amount of training
data, but it may seem that a limited amount of training data may have a few outliers that degrade
the quality of a model by causing strange parameter optimizations to accommodate those point.
However, if those outliers represent the true distribution of the data then there is nothing wrong
with your model adapting to put them on the correct side of some decision boundary.

This is actually especially true for models with an excessive amount of free parameters. They are
free to make tiny "bubbles" within their parameter space to shepherd those lone outlier points into
the correct output without accidentally including nearby points from the incorrect distribution.
This stand in contrast to some other approaches, such as SVMs, which are more sensitive to outliers
and where the decision boundaries cannot easily isolate single outlier points 

# Experimental Setup

The *basic_detector.py* will let us quickly explore some ideas about network fitting. For example,
if we wanted to see how well a neural network would learn a situation with one signal that was
perfectly correlated with a desired about and another signal that was present 50% of the time when
the signal was present and 50% of the time when the signal was not present we would run:

> python3 fitter.py --correlations 1.0 0.5 --anticorrelations 0.0 0.5

This would create a network with one linear layer that directly maps two inputs to a single output.
After 5000 iterations of batch size 32 we get:

> Batch 5000 loss is 0.00034125553793273866
> At batch 5000 layer 0 has weights [0.9997571110725403, -0.0004018025647383183] and bias -0.00018702048691920936
(Keep in mind that we are not doing anything to achieve determinism so you may get different results.)

Loss is the L1 loss, the absolute difference between the target value, 0 or 1, and the output of the
network. The weights show that the network has learned to directly map the first input to the output
(the 0.9997.... weight) and ignore the second input. This is correct.

If we add more layers:

> python3 fitter.py --correlations 1.0 0.5 --anticorrelations 0.0 0.5 --layers 3

The *--layers* option allows the user to specify an arbitrary number of linear layers. All but
the last linear layer will be followed with a *ReLU* layer.

Examining the weights gives a less clear picture of results when there are multiple layers, so
instead we probe the network by setting one input to `1` the other inputs to `0` and checking the
output. We observe these results:

> Batch 5000 loss is 0.00021663540974259377
> At batch 5000 input 0 has correlation 1.0004584789276123
> At batch 5000 input 1 has correlation -0.00017554312944412231

Again, this seems correct.

The program has the following additional options:

*--dropout*: Enable dropout after the first ReLU.
*--expand*: Double layer size with each linear layer.

# Spurious correlations

The concerns about overfitting often have to do with a network learning a spurious signal, one that
only occurs in the training data but not in the validation data or real world. We could represent
that with this trial:

> python3 fitter.py --correlations 1.0 0.8 --anticorrelations 0.0 0.0

Here a spurious signal shows up 80% percent of the time the network is expected to have a detection.
To an intelligent observer we can see that the first input is both necessary and sufficient to
predict the output and the second parameter should be ignored.

Running the above command three times yields these loss curves:
![Loss with spurious input](spurious_1layer_loss.png)

The first trial looks like it gets stuck at some point, but if we look at the actual weights for the
first and second input we can see that they are moving in the correct direction and would eventually
get to their correct values (0 and 1 respectively).

![Weight 1 with spurious input](spurious_1layer_input_1_weights.png)
![Weight 2 with spurious input](spurious_1layer_input_2_weights.png)

It is important to notice that the weights do move in the wrong direction in the beginning. The
weights for input 2 start off by moving towards 1 before reversing and moving towards 0. This has to
do with how long it takes for the weights from the first input be correlate it with the output.
Imagine that these two inputs are actually outputs from several convolution layers serving as a
feature extractor. If the spurious signal is for some reason easier for the convolutions to detect
than the desired signal then the neural network will look like it is learning the wrong thing
quickly in early training epochs. With enough time however, the desired signal is better correlated
with the output and the network will converge to the desired solution. A risk of early stopping
during training based upon a fear of overfitting rather than an observed plateau in performance is that the network has not yet converged and incorrect correlations still exist.

A three layer network shows similar results in loss and input correlation, but with an important
difference:

![3-layer loss with spurious input](spurious_3layer_loss.png)
![3-layer correlation 1 with spurious input](spurious_3layer_input_1_correlation.png)
![3-layer correlation 2 with spurious input](spurious_3layer_input_2_correlation.png)

Here note that the correlation of the second variable is somewhat arbitrary. That is because with a
more complicated network structure there is no reason to force the correlation to 0. One obvious
approach here is to add dropout to break the correlation. Doing that we get this disaster:

![3-layer loss with spurious input and dropout](spurious_3layer_dropout_loss.png)
![3-layer correlation 1 with spurious input and dropout](spurious_3layer_dropout_input_1_correlation.png)
![3-layer correlation 2 with spurious input and dropout](spurious_3layer_dropout_input_2_correlation.png)

Is this evidence of the overfitting problem of deep neural networks? No! This is evidence that we
are making some big mistakes. The two inputs are always correlated in the training data, and with
dropout the network must lean on the second input to deduce the state of the first.

So what is the correct approach to take if all of our training data has some bad correlation that we
want to break? Let's try fixing the data just a little bit but adding a few negative examples for
input 2:

> python3 fitter.py --correlations 1.0 0.8 --anticorrelations 0.0 0.001 --layers 3

![3-layer loss with spurious input and negative examples](spurious_3layer_negative_loss.png)
![3-layer correlation 1 with spurious input and negative examples](spurious_3layer_negative_input_1_correlation.png)
![3-layer correlation 2 with spurious input and negative examples](spurious_3layer_negative_input_2_correlation.png)

This fixes the correlation of the first input at least, although we don't always get the correlation
of the second input to go to 0.

# Incorrect local minima due to unnecessary correlations

Is the general wisdom of overfitting correct in the real world where our training data is limited?
For example, what if you are training a vehicle to drive through a course and there is some "tell" always
associated with a particular left curve in your training data but then in the test data that "tell"
is gone. Shouldn't the model be able to "figure out" that the "tell" isn't present during all left
curves so it shouldn't have been learned? And is this a problem of overfitting? In such cases we
presume that early termination of training will stop the parameters from fully fitting to the
"tell".

What happens if the "tell" always occurs with the signal?

> python3 fitter.py --correlations 1.0 1.0 --anticorrelations 0.0 0.0 --layers 1

The training loss curve looks good:
![Loss with tell](two_variable_overfit_1.png)

However the model is combining the two inputs in an arbitrary way:

![Two variable input 1 parameter convergence](two_variable_overfit_input_1_weights.png)
![Two variable input 2 parameter convergence](two_variable_overfit_input_1_weights.png)

This is not surprising as the inputs are exactly the same and this model has only 1 layer.


# More complications

If the "tell" only occurs in 90% of left curves then the parameters for the "tell" should be learned
more slowly than the parameters for the road itself so early stopping may look like the correct
thing to do. Try running this command and see how the parameters are tuned:

> python3 fitter.py --correlations 1.0 0.9 --anticorrelations 0.0 0.0 --layers 1

In this case the correlation of `1.0` is the correlation of the road to the output and the
correlation of `0.9` is the correlation of the "tell". An example loss curve is:

![Loss with tell](example_overfit_1.png)

If training is perfect then the weight for the first parameter should converge to 1 and the weight
of the second should converge to 0:

![Tell first parameter convergence](example_overfit_1_input_1_weights.png)
![Tell second parameter convergence](example_overfit_1_input_2_weights.png)

In some of the trials it is taking a while to converge to the right place, but things do seem to go
in the correct direction. What if the "tell" has a different correlation to our desired output
though? For example, what if the "tell" is always present when turning left, but also shows up 20%
of the time when not turning left.

> python3 fitter.py --correlations 1.0 1.0 --anticorrelations 0.0 0.2 --layers 1

This seems to be much harder, with the network take a long time to converge:

![Loss with tell](example_overfit_2.png)

Weights do still converge to the correct values though:

![Tell first parameter convergence](example_overfit_2_input_1_weights.png)
![Tell second parameter convergence](example_overfit_2_input_2_weights.png)

Now what happens if our label is noisy and, for some reason, the road input is not always present?

> python3 fitter.py --correlations 0.99 1.0 --anticorrelations 0.0 0.2 --layers 1

This feels like it should be harder, but we actually see the network converging faster:

![Loss with tell](example_overfit_3.png)

The trend holds in the weights as well:

![Tell first parameter convergence](example_overfit_3_input_1_weights.png)
![Tell second parameter convergence](example_overfit_3_input_2_weights.png)
