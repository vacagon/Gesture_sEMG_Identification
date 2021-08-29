function MAV = MAV(segment)
    L = length(segment);
    S = sum(abs(segment));
    MAV = S/L;
end