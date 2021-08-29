function SSC = SSC(segment)
SSC = 0;
L = length(segment);
for i = 2:(L-1)
    if (segment(i) > segment(i-1)) && (segment(i) > segment(i+1))
        SSC = SSC + 1;
    end
    if (segment(i) < segment(i-1)) && (segment(i) < segment(i+1))
        SSC = SSC + 1;
    end
end
end