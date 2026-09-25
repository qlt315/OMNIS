function tighten_lr(fig)
%TIGHTEN_LR Remove leftover left/right figure margins.
if nargin < 1 || isempty(fig)
    fig = gcf;
end
drawnow;

tls = findall(fig, 'Type', 'tiledlayout');
if ~isempty(tls)
    set(tls, 'Padding', 'none');
    return
end

axs = findall(fig, 'Type', 'axes');
if isempty(axs)
    return
end
tags = get(axs, 'Tag');
if ~iscell(tags)
    tags = {tags};
end
axs = axs(~strcmpi(tags, 'legend'));
if isempty(axs)
    return
end

set(axs, 'Units', 'normalized');
n = numel(axs);
pos = zeros(n, 4);
outerL = inf;
outerR = -inf;
for i = 1:n
    pos(i, :) = axs(i).Position;
    ti = axs(i).TightInset;
    outerL = min(outerL, pos(i, 1) - ti(1));
    outerR = max(outerR, pos(i, 1) + pos(i, 3) + ti(3));
end

margin = 0.005;
span = outerR - outerL;
if ~(span > 0)
    return
end
scale = (1 - 2 * margin) / span;
shift = margin - outerL * scale;
for i = 1:n
    axs(i).Position = [pos(i, 1) * scale + shift, pos(i, 2), pos(i, 3) * scale, pos(i, 4)];
end
end
