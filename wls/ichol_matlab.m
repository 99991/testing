% 0.192634 seconds to build preconditioner, 12.9 nonzero elements per row.
% 0.20792167099716607 in python

f = fopen('size.bin', 'r');
s = fread(f, 'uint32');
fclose(f);

h = s(1);
w = s(2);

f = fopen('zi.bin', 'r');
i = fread(f, 'uint32') + 1;
fclose(f);

f = fopen('j.bin', 'r');
j = fread(f, 'uint32') + 1;
fclose(f);

f = fopen('v.bin', 'r');
v = fread(f, 'double');
fclose(f);

f = fopen('b.bin', 'r');
b = fread(f, 'double');
fclose(f);

A = sparse(i, j, v);

opt.type = 'ict';
opt.droptol = 1e-4;

tic;

M = ichol(A, opt);

toc;
'to compute preconditioner'

tic;

[x, flag, relres, iter] = pcg(A, b, 1e-7, 100, M, M');

toc;
'to solve'

iter

result = reshape(x, [w, h])';

[i, j, v] = find(M);

sprintf('%f nnz per row', size(v) / (w * h))

a = min(result(:));
b = max(result(:));

result = (result - a) ./ (b - a);

imshow(result)
%pause
