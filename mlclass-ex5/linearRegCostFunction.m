function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;

temp = h .-y;
temp = temp.^2;


J1 = sum(temp) / (2 * m);

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
J = J1 + J_temp;

% =========================================================================

grad = grad(:);

end
