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

hypotesys_func = X * theta;
sigmoid_func = sigmoid(hypotesys_func);

unreg_J = (1/m) * (((-y') * log(sigmoid_func)) - ((1 - y)' * (log(1 - sigmoid_func))));

theta(1) = 0;


sum_theta = theta' * theta;
reg_term_cost = (lambda / (2 * m)) * sum_theta;

J = unreg_J + reg_term_cost;


unreg_grad = (1/m) * X' * (sigmoid_func - y);
reg_term_grad = theta * (lambda / m);

grad = unreg_grad + reg_term_grad;



% =============================================================

end
