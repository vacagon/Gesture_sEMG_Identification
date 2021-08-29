function ZC = ZC(segment)
ZC = 0;
L = length(segment);
for i = 1:(L-1)
    if (segment(i+1)>0) && (segment(i) < 0)
       ZC = ZC + 1; 
    end
    if (segment(i+1)<0) && (segment(i) > 0)
       ZC = ZC + 1; 
    end
end
end