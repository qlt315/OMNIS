function est_err_str = get_est_err_str(est_err_para)
    est_err_str = strrep(num2str(est_err_para), '.', '_'); % Replace '.' with '_'
end
