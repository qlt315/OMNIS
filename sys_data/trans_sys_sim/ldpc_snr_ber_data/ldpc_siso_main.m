clear all;
% close all;

% Simulation parameters
max_runs = 2000; % Maximum number of simulation runs
max_decode_iterations = 20; % Maximum decoding iterations
ldpc_code = ldpc_code(0, 0); % Initialize LDPC code object
min_sum = 1; % Use min-sum algorithm for decoding
n_0 = 1/2; % Noise power spectral density

% LDPC parameters
block_length = 648; % Codeword length, options: 648, 1296, 1944

% Modulation parameters
constellation_name = 'bpsk'; % Modulation scheme, options: 'bpsk', 'ask4', 'ask8' (equivalently QPSK, 16-QAM, 64-QAM)
modulation = ldpc_constellation(constellation_name); % Initialize modulation object

% SNR range
ebno_db_vec = 0:2:10; % Eb/N0 range in dB

% Code rates and channel estimation error parameters
rates = [1/2, 2/3, 3/4, 5/6]; % Code rates
est_err_para_vec = [0, 0.3, 0.5]; % Channel estimation error parameters

% Initialize storage for all results
bler_all_results = struct();
ber_all_results = struct();
% Start timer
tic;

% Outer loop: Code rates
for rate = rates
    % Inner loop: Channel estimation error
    for est_err_para = est_err_para_vec
        % Initialize error statistics
        num_block_err = zeros(length(ebno_db_vec), 1);
        ber = zeros(length(ebno_db_vec), 1);

        % Load LDPC code
        ldpc_code.load_wifi_ldpc(block_length, rate);
        info_length = ldpc_code.K; % Information bit length
        disp(['Running LDPC with N = ', num2str(block_length), ', rate = ', num2str(rate), ', constellation = ', constellation_name, ', est_err_para = ', num2str(est_err_para)]);

        % Convert Eb/N0 to SNR
        snr_db_vec = ebno_db_vec + 10*log10(info_length/block_length) + 10*log10(modulation.n_bits);

        % Main simulation loop
        for i_run = 1:max_runs
            % Display progress
            if mod(i_run, max_runs/10) == 1
                disp(['Current run = ', num2str(i_run), ' percentage complete = ', num2str((i_run-1)/max_runs * 100), '%, time elapsed = ', num2str(toc), ' seconds']);
            end

            % Generate noise
            noise = sqrt(n_0) * randn(block_length/modulation.n_bits, 1);

            % Generate random information bits
            info_bits = rand(info_length, 1) < 0.5;

            % LDPC encoding
            coded_bits = ldpc_code.encode_bits(info_bits);

            % Scrambling
            scrambling_bits = (rand(block_length, 1) < 0.5);
            scrambled_bits = mod(coded_bits + scrambling_bits, 2);

            % Modulation
            x = modulation.modulate(scrambled_bits);

            % Loop over different SNR values
            for i_snr = 1:length(snr_db_vec)
                snr_db = snr_db_vec(i_snr);
                snr = 10^(snr_db/10); % Convert SNR from dB to linear scale

                % Simulate channel (Rayleigh fading)
                h = randn(1, 1) + 1i * randn(1, 1) / sqrt(2);

                % Add noise to the signal
                y = h .* x + noise / sqrt(snr);

                % Channel equalization
                est_err = est_err_para * abs(h); % Channel estimation error
                h_est = h + est_err + est_err * 1i; % Estimated channel
                y = y ./ h_est; % Equalization

                % Compute LLR (Log-Likelihood Ratio)
                [llr, ~] = modulation.compute_llr(y, n_0/snr);
                llr = llr .* (1 - 2 * scrambling_bits); % Apply scrambling

                % LDPC decoding
                [decoded_codeword, ~] = ldpc_code.decode_llr(llr, max_decode_iterations, min_sum);
                

                % Count bit error
                num_bit_errors = sum(decoded_codeword ~= coded_bits);
                ber(i_snr) = ber(i_snr) + num_bit_errors;
                

                % Count block errors
                if any(decoded_codeword ~= coded_bits)
                    num_block_err(i_snr) = num_block_err(i_snr) + 1;
                else
                    break; % If decoding succeeds at current SNR, assume it will succeed at higher SNRs
                end
            end
        end
                
               


        % Calculate BLER (Block Error Rate)
        bler = num_block_err / max_runs;

        % Calculate BER (Bit Error Rate)
        ber = ber / (block_length * max_runs);

        % Store BLER results
        bler_result_key = sprintf('rate_%s_esterr_%s', get_rate_str(rate), get_est_err_str(est_err_para));
        bler_all_results.(bler_result_key) = bler;
        

        % Store BER results
        ber_result_key = sprintf('rate_%s_esterr_%s', get_rate_str(rate), get_est_err_str(est_err_para));
        ber_all_results.(ber_result_key) = ber;

        % Save results to file
        filename = sprintf('ldpc_siso_data/snr_0_0.5_10_%s_esterr_%.1f_rate_%s.mat', ...
            constellation_name, est_err_para, get_rate_str(rate));
        save(filename, 'bler', 'ber', 'ebno_db_vec');
    end
end




% Define line styles for different estimation errors
line_styles = {'-', '--', ':', '-.'};

% Get unique rate values for consistent color assignment
unique_rates = unique(rates);
num_rates = length(unique_rates);
colors = lines(num_rates); % Generate distinct colors for each rate

% Plot all results in one figure
figure(1);
hold on; grid on;

for rate_idx = 1:num_rates
    rate = unique_rates(rate_idx);
    color = colors(rate_idx, :); % Assign color based on rate
    
    for est_idx = 1:length(est_err_para_vec)
        est_err_para = est_err_para_vec(est_idx);
        bler_result_key = sprintf('rate_%s_esterr_%s', get_rate_str(rate), get_est_err_str(est_err_para));
        bler = bler_all_results.(bler_result_key);
        
        semilogy(ebno_db_vec, bler, 'Color', color, 'LineStyle', line_styles{mod(est_idx - 1, length(line_styles)) + 1}, ...
            'LineWidth', 2, 'Marker', 'o', 'Markersize', 7, ...
            'DisplayName', sprintf('Rate = %s, est\\_err = %.1f', get_rate_str(rate), est_err_para));
    end
end
set(gca, 'YScale', 'log')
xlabel('SNR (dB)');
ylabel('BLER');
title('LDPC BLER Performance');
legend show;
% saveas(gcf, 'ldpc_ber.png'); % Save plot as PNG



figure(2);
hold on; grid on;
for rate_idx = 1:num_rates
    rate = unique_rates(rate_idx);
    color = colors(rate_idx, :); % Assign color based on rate
    
    for est_idx = 1:length(est_err_para_vec)
        est_err_para = est_err_para_vec(est_idx);
        ber_result_key = sprintf('rate_%s_esterr_%s', get_rate_str(rate), get_est_err_str(est_err_para));
        ber = ber_all_results.(ber_result_key);
        
        semilogy(ebno_db_vec, ber, 'Color', color, 'LineStyle', line_styles{mod(est_idx - 1, length(line_styles)) + 1}, ...
            'LineWidth', 2, 'Marker', 'o', 'Markersize', 7, ...
            'DisplayName', sprintf('Rate = %s, est\\_err = %.1f', get_rate_str(rate), est_err_para));
    end
end
set(gca, 'YScale', 'log')
xlabel('SNR (dB)');
ylabel('BER');
title('LDPC BER Performance');