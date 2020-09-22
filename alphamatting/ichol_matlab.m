f = fopen('shape.bin', 'r');
shape = fread(f, 'uint32');
fclose(f);

h = shape(1);
w = shape(2);
n = w * h;
droptol = 1e-4;

f = fopen('A_i.bin', 'r');
i = fread(f, 'uint32') + 1;
fclose(f);

f = fopen('A_j.bin', 'r');
j = fread(f, 'uint32') + 1;
fclose(f);

f = fopen('A_v.bin', 'r');
v = fread(f, 'double');
fclose(f);

A = sparse(i, j, v, n, n);

opt.type = 'ict';
opt.droptol = droptol;

for i = 1:5
    sprintf('compute ichol MATLAB')
    tic;

    M = ichol(A, opt);

    toc;
end

[i, j, v] = find(M);

f = fopen('L_i.bin', 'w');
fwrite(f, uint32(i - 1), 'uint32');
fclose(f);

f = fopen('L_j.bin', 'w');
fwrite(f, uint32(j - 1), 'uint32');
fclose(f);

f = fopen('L_v.bin', 'w');
fwrite(f, v, 'double');
fclose(f);
