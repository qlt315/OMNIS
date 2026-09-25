function colors = algo_family_colors(algo_keys)
%ALGO_FAMILY_COLORS Same-family schemes share a hue; differ by shade only.
%
%   Families:
%     OMNIS+ — causal / cto        (blue, dark → light)
%     OMNIS  — ucb                 (cyan/teal, separate)
%     GDO    — gdo                 (green)
%     DQN    — dqn                 (purple)
%     PPO    — ppo / mappo         (orange-red, dark → light)
%
%   colors is [numel(algo_keys) x 3] RGB in [0,1].

n = numel(algo_keys);
colors = zeros(n, 3);

% Base hues (saturated, paper-friendly)
blue   = [0.10, 0.32, 0.72];
teal   = [0.05, 0.58, 0.62];
green  = [0.12, 0.52, 0.28];
purple = [0.48, 0.20, 0.62];
orange = [0.82, 0.38, 0.12];

% Within-family shade: 0 = darkest (base), 1 = lightest (toward white)
omnis_plus_shade = containers.Map( ...
    {'causal', 'cto'}, {0.00, 0.35});
ppo_shade = containers.Map( ...
    {'ppo', 'mappo'}, {0.00, 0.40});

for i = 1:n
    key = algo_keys{i};
    if omnis_plus_shade.isKey(key)
        colors(i, :) = local_shade(blue, omnis_plus_shade(key));
    elseif strcmp(key, 'ucb')
        colors(i, :) = teal;
    elseif strcmp(key, 'gdo')
        colors(i, :) = green;
    elseif strcmp(key, 'dqn')
        colors(i, :) = purple;
    elseif ppo_shade.isKey(key)
        colors(i, :) = local_shade(orange, ppo_shade(key));
    else
        colors(i, :) = [0.45, 0.45, 0.45];
        warning('algo_family_colors: unknown key "%s"', key);
    end
end
end

function c = local_shade(base, t)
% Mix base toward white; t in [0,1].
c = base + (1 - base) * t;
c = min(max(c, 0), 1);
end
