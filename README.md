The dataset consists of points labelled in blue and red(graph has been plotted using Matplotlib). 
I have implemented a single hidden layer neural network

The mathematical model, that we need to implement in the program ,  is as follows :
z[1](i)=W[1]x(i)+b[1]
a[1](i)=tanh(z[1](i))
z[2](i)=W[2]a[1](i)+b[2]
y^(i)=a[2](i)=σ(z[2](i))
y(i)prediction={10if a[2](i)>0.5
                0 ,otherwise }
cost J : 
J=−1m∑i=0m(y(i)log(a[2](i))+(1−y(i))log(1−a[2](i)))

a[1] is the input given to the second(hidden layer), whose output, after applying activation produces the output layer 


Steps followed : 
a) in "layer_sizes", we decide :  n_x: the size of the input layer - n_h: the size of the hidden layer (set this to 4) - n_y: the size of the output layer

b) in "initialize_parameters" we randomly set the initial values of weights and biases for our model

c) Next , we implement the loop - forward propagation, calculating loss, backward propagation to get gradients, update parameters(gradient descent) 

1) forward_propagation ( the arguments and what it returns has been commented in the codebox) "cache" is the dictionary which will be required in back propagation.Sigmoid and tanh are the activation functions used

2)compute_cost : using the numpy functions to calculate cross entropy loss, as mentioned above

3)backward_propagation : here we calculate the gradients that are required for the gradient descent (θ=θ−α*∂J/∂θ, where alpha is the learning rate) here we only calculate the gradients

4) update_parameters : use the gradients from previous step to update the parameters


d) We incorporate the above steps into "nn_model" , and return the dictionary containing learnt parameters from c)4). These would be used for predictions.
In here, we run the loop for "num_iterations" time.(Reminds of the 'epochs' :))

Next, we make predictions using the learnt parameters. For accuracy, we have defined the function ourselves.

At last, we try different values of number of units in the hidden layer (fine tuning, hyperparameter tuning)

