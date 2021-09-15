a = [1 2 3 4]; % Semicolon supresses output
a = [1 2 3; 4 5 6; 7 8 9]; % Row by row
a(1,2) % Get single index
a(1, :) % Get entire row
a(6) % Can access matrix by a flattened index

average( [1 2 1] ) % Weird, function definitions MUST be at the end of the file

% ave is the return value
% average is the function name
% x is the input name
function ave = average(x)
    number_of_elements = numel(x); %numel is built in to get num elements
    ave = sum(x(:)) / number_of_elements;
end

