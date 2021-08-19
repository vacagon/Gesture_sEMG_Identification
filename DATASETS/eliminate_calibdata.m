function x = eliminate_calibdata(data)
C = mat2cell(data, [20000, length(data)-20000]);
x = cell2mat(C(2));
end