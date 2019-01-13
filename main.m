% Minover algorithm

% Machine learning assignment 2
% Raphaël Scherrer

clear, clc, close all

% Parameters
ndims = 10;
datastd = 5; % standard deviation for generating points
nreplicates = 10;
tmax = 1000; % max training time
alphavalues = 0.25:0.25:3;
[~, nalphas] = size(alphavalues);

% Initialize vector of generalization errors
generrors = zeros(1, nalphas);

% Loop through values of alpha
for j = 1:nalphas
    
    % Current alpha value
    alpha = alphavalues(j);
    
    % Alpha determines the size of the dataset
    npoints = round(alpha * ndims);
    
    % Max training time
    %tmax = tmax * npoints; 
    
    % Initialize the error for the current set of replicates
    generror = 0;
    
    % Loop through replicates
    for i = 1:nreplicates

        msg = "alpha = " + alpha + ", replicate = " + i;
        disp(msg)
        
        % Generate a dataset from a multivariate normal distribution
        dataset = normrnd(0, datastd, npoints, ndims);

        % Generate a teacher perceptron (normalized such that norm = sqrt of ndims)
        teacher = normrnd(0, 1, ndims, 1);
        teacher = teacher * sqrt(ndims) / norm(teacher);

        % Generate labels of the data according to the teacher perceptron
        labels = sign(dataset * teacher);

        % Train the perceptron with the minover algorithm and record the
        % generalization error
        generror = generror + train_minover(dataset, labels, teacher, tmax);

    end

    % Compute the average error across replicates
    generrors(j) = generror / nreplicates;
    
end

% Plot
plot(alphavalues, generrors)
title('Minover performance')
xlabel('\alpha (= P/N)')
ylabel('Generalization error')