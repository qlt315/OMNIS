function rate_str = get_rate_str(rate)
    if rate == 1/2
        rate_str = '1_2';
    elseif rate == 2/3
        rate_str = '2_3';
    elseif rate == 3/4
        rate_str = '3_4';
    elseif rate == 5/6
        rate_str = '5_6';
    else
        error('Invalid rate. Supported rates are 1/2, 2/3, 3/4, 5/6.');
    end
end

