function make_rr(listfile)
% MAKE_RR  Build Wald reduced-resolution inputs (MTF-matched MS and PAN) for each chip.
%   Writes <chip>_RRin.mat with MS_LR, PAN_LR (degraded by ratio 4) and GT (= MS).
setup_toolbox();
files = strtrim(splitlines(fileread(listfile)));
files = files(~cellfun(@isempty, files));
for k = 1:numel(files)
    out = strrep(files{k}, '.mat', '_RRin.mat');
    if exist(out, 'file'), continue; end
    d = load(files{k}, 'MS', 'PAN', 'SENSOR', 'EVAL');
    if isfield(d, 'EVAL') && d.EVAL ~= 1, continue; end
    GT = double(d.MS);
    [MS_LR, PAN_LR] = resize_images(GT, double(d.PAN), 4, d.SENSOR);
    MS_LR = single(MS_LR); PAN_LR = single(PAN_LR); GT = single(GT);
    save(out, 'MS_LR', 'PAN_LR', 'GT');
end
end
