function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);

l_temp = -(log(h) .* y);
r_temp = -(log(ones(size(h))-h) .* (ones(size(y))-y));
temp = l_temp + r_temp;

J = sum(temp) / m;

grad = h .-y;
grad = grad .*X;
grad = sum(grad) /m;

grad_temp = theta * (lambda/m);

grad_temp = grad_temp';

grad_temp(1) = 0;
grad = grad + grad_temp;

theta = theta .*theta;

J_temp = lambda / (2 * m);
J_temp = J_temp * sum(theta(2:size(theta)));
J = J + J_temp;

% =============================================================

end
