function WL = WL(segment)
    WL = 0;
    L = length(segment);
    for i = 1:(L-1)
        WL = WL + (segment(i+1)-segment(i));
    end
end