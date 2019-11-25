function Nij = gen_N(i,j,t,T)



if j == 0
    if t>=T(i) && t<T(i+1)
        Nij = 1;
    else
        Nij = 0;
    end
else
    j = j-1;
    Nij = (t-T(i))/(T(i+j+1)-T(i))*gen_N(i,j,t,T)+...
        (T(i+j+2)-t)/(T(i+j+2)-T(i+1))*gen_N(i+1,j,t,T);
end


end