function tighten_all(fig, margin)
%TIGHTEN_ALL Crop leftover figure margins on all four sides.
% Uses each axes TightInset so xlabel/ylabel sit flush to the figure edge.
if nargin < 1 || isempty(fig)
    fig = gcf;
end
if nargin < 2 || isempty(margin)
    margin = 0.008;
end
drawnow;

tls = findall(fig, 'Type', 'tiledlayout');
if ~isempty(tls)
    set(tls, 'Padding', 'tight');
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
for i = 1:numel(axs)
    ax = axs(i);
    if isprop(ax, 'PositionConstraint')
        ax.PositionConstraint = 'outerposition';
    end
    outer = ax.OuterPosition;
    ti = ax.TightInset;
    left = outer(1) + ti(1) + margin;
    bottom = outer(2) + ti(2) + margin;
    width = outer(3) - ti(1) - ti(3) - 2 * margin;
    height = outer(4) - ti(2) - ti(4) - 2 * margin;
    if width > 0.05 && height > 0.05
        ax.Position = [left, bottom, width, height];
    end
end
drawnow;
end
