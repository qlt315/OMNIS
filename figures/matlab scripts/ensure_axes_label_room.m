function ensure_axes_label_room(fig)
%ENSURE_AXES_LABEL_ROOM Keep axis labels inside the figure after resize.
% Call after plotting (and after tighten_lr). Re-run on SizeChangedFcn when
% the figure is vertically squashed so xlabel is not clipped.
if nargin < 1 || isempty(fig)
    fig = gcf;
end
drawnow;
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

margin = 0.01;
min_w = 0.25;
min_h = 0.20;
for i = 1:numel(axs)
    ax = axs(i);
    ax.Units = 'normalized';
    ti = ax.TightInset;
    pos = ax.Position;
    left = max(pos(1), ti(1) + margin);
    bottom = max(pos(2), ti(2) + margin);
    right = min(pos(1) + pos(3), 1 - (ti(3) + margin));
    top = min(pos(2) + pos(4), 1 - (ti(4) + margin));
    w = max(right - left, min_w);
    h = max(top - bottom, min_h);
    if left + w > 1 - margin
        left = max(margin, 1 - margin - w);
    end
    if bottom + h > 1 - margin
        bottom = max(margin, 1 - margin - h);
    end
    ax.Position = [left, bottom, w, h];
end
end
