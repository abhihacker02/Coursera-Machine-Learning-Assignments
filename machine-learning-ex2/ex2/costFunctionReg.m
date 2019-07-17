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

yy=y;
y=y';
h=(theta'*X')';
h=sigmoid(h);
hh=h;
one_minus_h=log(1-h);
one_minus_y=1-y;
h=log(h);
J=-y*h-one_minus_y*one_minus_h;
theta_excluding_zero=theta(2:size(theta));
theta_excluding_zero.^=2;
sm=lambda*sum(theta_excluding_zero)/(2*m);
J=J/m+sm;

diff=hh-yy;
pr=(diff'*X)';
pr=pr/m;
grad=pr;
grad_excluding_zero=grad(2:size(grad));
smm=theta(2:size(theta))*lambda/m;
grad_excluding_zero+=smm;
grad=[grad(1:1);grad_excluding_zero];


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
