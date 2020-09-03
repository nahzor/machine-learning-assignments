function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];
z2 = X * Theta1';
h2 = sigmoid(z2);

h2 = [ones(size(h2, 1), 1) h2];
z3 = h2 * Theta2';
h3 = sigmoid(z3);

A3 = h3;
[~,p] = max(A3,[],2);

yv = repmat(1:num_labels, size(y,1) , 1) == repmat(y, 1, num_labels);
Y=yv;
Uno=ones(m,num_labels);
Cost=-1/m*(Y.*log(A3)+(Uno-Y).*log(Uno-A3));

J=J+sum(sum(Cost));
%J = sum(ftemp) / m;

regVal = 0;

for i = 1:size(Theta1,1)
  for j = 2:size(Theta1,2)
    regVal = regVal + (Theta1(i,j)*(Theta1(i,j)));
endfor
endfor

for i = 1:size(Theta2,1)
  for j = 2:size(Theta2,2)
    regVal = regVal + (Theta2(i,j)*(Theta2(i,j)));
endfor
endfor

regVal = regVal*lambda/(2*m);

J=J+regVal;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%Step 1
a1 = X;

z2 = a1 * Theta1';
h2 = sigmoid(z2);

h2 = [ones(m, 1) h2];
z3 = h2 * Theta2';
h3 = sigmoid(z3);

delta3 = zeros(size(h3));
%Step2
for i = 1:size(h3, 2)
ytemp = zeros(size(y));

  for j = 1:size(ytemp)
    if (y(j,1) == i)
      ytemp(j,1) = 1;
    endif
  endfor

hi = h3(:,i);

delta3(:,i) = hi.-ytemp;

endfor

%Step3

%Theta2(:,1) = [];
temp = (delta3*Theta2);
z2temp = [ones(size(z2,1), 1) z2];
delta2 = temp.*sigmoidGradient(z2temp);

delta2(:,1) = [];

Theta2_grad = delta3'*h2;
Theta1_grad = delta2'*a1;

regVal1 = Theta1*lambda/m;
regVal2 = Theta2*lambda/m;

regVal1(:,1) = zeros(size(regVal1,1),1);
regVal2(:,1) = zeros(size(regVal2,1),1);


Theta2_grad = Theta2_grad/m + regVal2;
Theta1_grad = Theta1_grad/m + regVal1;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
