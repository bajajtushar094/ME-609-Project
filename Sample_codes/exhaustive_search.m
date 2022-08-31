a = input('Enter a = lower limit of x = ');
b = input('Enter a = upper limit of x = ');
Exhaustive_Search(a, b);

function fun_val = Objective_Fun(x)
    fun_val = x*x + (54/x);
end

function Exhaustive_Search(a, b)
    fprintf('Exhaustive Search Method');
    n = input('\nEnter n = number of Steps = ');
    % Step 1
    delta = (b - a) / n; % Set data value
    x1 = a; x2 = x1 + delta; x3 = x2 + delta; % New points
    feval = 0; % Number of function evaluations
    f1 = Objective_Fun(x1); % Calculate objective function value
    f2 = Objective_Fun(x2);
    f3 = Objective_Fun(x3);
    i = 1; % Number of iterations
    feval = feval + 3;
    out = fopen('Exhaustive_search_iterations.out', 'w'); % Output file
    fprintf(out, '#It\tx1\tx2\tx3\tf(x1)\tf(x2)\tf(x3)\n');
    condition = true;
    while condition
        fprintf(out, '%d\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\n',i,x1,x2,x3,f1,f2,f3);
        % Step 2: Termination condition
        if (f1 >= f2)
            if (f2 <= f3)
                break;
            end
        end
        % If not terminated, update values
        x1 = x2; x2 = x3; x3 = x2 + delta;
        f1 = f2;
        f2 = f3;
        f3 = Objective_Fun(x3);
        feval = feval + 1;
        i = i + 1;
        if (x3 > b) % Step 3
            condition = false;
        end
    end
    fprintf('\n*************************\n');
    fprintf('The minimum point lies between (%8.3f, %8.3f)', x1, x3);
    fprintf('\nTotal number of function evaluations: %d\n', feval);
    % Store in the file
    fprintf(out, '\nThe minimum point lies between (%8.3f, %8.3f)', x1, x3);
    fprintf(out, '\nTotal number of function evaluations: %d', feval);
end