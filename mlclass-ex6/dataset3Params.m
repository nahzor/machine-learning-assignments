function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

clist = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmalist = [0.01 0.03 0.1 0.3 1 3 10 30];
ycalc = 2000;
c_index = 0;
sigma_index = 0;
count = 0;
for i = 1:length(clist)
	for j = 1:length(sigmalist)
		count = count+1;
		fprintf('\nCombination  #%d\n',count);
		cval = clist(i);
		sigmaval = sigmalist(i);

		model1= svmTrain(X, y, clist(i), @(x1,x2) gaussianKernel(x1, x2,sigmalist(j)));
		
		predictions = svmPredict(model1, Xval);
		temp = mean(double(predictions ~= yval));

		if temp < ycalc
			ycalc = temp;
			c_index = i;
			sigma_index = j;
		end

	end
end

C = clist(c_index);
sigma = sigmalist(sigma_index);

% =========================================================================

end
