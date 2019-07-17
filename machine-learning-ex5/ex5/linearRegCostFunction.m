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
yy=y;
h=(theta')*X';
h=h';
diff=h-y;
diff=diff.^2;
J=sum(diff);
J=J/(2*m);
theta_excluding_zero=theta(2:size(theta));
theta_excluding_zero.^=2;
sm=lambda*sum(theta_excluding_zero)/(2*m);
J=J+sm;

hh=h;
diff=hh-yy;
pr=(diff'*X)';
pr=pr/m;
grad=pr;
grad_excluding_zero=grad(2:size(grad));
smm=theta(2:size(theta))*lambda/m;
grad_excluding_zero+=smm;
grad=[grad(1:1);grad_excluding_zero];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
