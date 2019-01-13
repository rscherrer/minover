% Function to train a perceptron with the minover algorithm
function generror = train_minover(dataset, labels, teacher, tmax)

[~, ndims] = size(dataset);

% Initialize the student
student = zeros(ndims, 1);

% Minover algorithm until convergence or max time
for t = 1:tmax
    
    % Calculate stabilities of all points
    stabilities = dataset * student .* labels;

    % What point is the least stable?
    [~, weakpoint] = min(stabilities);

    % Update the student perceptron
    student = student + dataset(weakpoint,:)' .* labels(weakpoint) ./ ndims;
    
end

% Calculate generalization error
generror = acos(dot(student, teacher) / norm(student) / norm(teacher)) / pi;

end