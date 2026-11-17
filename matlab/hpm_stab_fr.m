function hpm_stab_fr(listfile, outdir)
% HPM_STAB_FR  Full-resolution MTF-GLP-HPM and its numerically stabilised
% variant (HPM*) for a list of chips, with the same inputs, toolbox call and
% parameters as fuse_and_eval.m (Vivone et al., 2021 toolbox, ratio 4, chip
% SENSOR). HPM* floors the interpolated MS band, the equalised PAN and its
% low-pass version at one stored count (1e-4 reflectance) before the
% modulation; otherwise it is identical to the toolbox routine (see
% hpm_stab in quality_ext.m).
% Output per chip: <name>_hpmstab.mat with
%   L      [4h 4w 2] single, luminance (mean of R,G,B = bands 5,3,2) of the
%          recomputed MTF-GLP-HPM and of HPM*, order in NAMES
%   HPMS   [4h 4w 8] single, HPM* image (all bands)
%   FLOOR  [3 8] fraction of PAN-grid pixels where the floor is active,
%          rows: interpolated MS, equalised PAN, low-pass PAN
%   FLOOR_ANY [1 8] fraction where at least one of the three floors is active
setup_toolbox();
ratio = 4; RGB = [5 3 2];
NAMES = {'MTF_GLP_HPM', 'MTF_GLP_HPM_stab'};
files = strtrim(splitlines(fileread(listfile)));
files = files(~cellfun(@isempty, files));
if ~exist(outdir, 'dir'), mkdir(outdir); end
t0 = tic;
for k = 1:numel(files)
    [~, name] = fileparts(files{k});
    out = fullfile(outdir, [name '_hpmstab.mat']);
    if exist(out, 'file'), continue; end
    d = load(files{k}, 'MS', 'PAN', 'SENSOR');
    sensor = d.SENSOR;
    MS = double(d.MS); PAN = double(d.PAN);
    EXP = interp23tap(MS, ratio);
    H = MTF_GLP_HPM(EXP, PAN, sensor, ratio);
    [S, FLOOR, FLOOR_ANY] = hpm_stab(EXP, PAN, sensor, ratio);
    L = zeros(size(PAN, 1), size(PAN, 2), 2, 'single');
    L(:, :, 1) = single(mean(H(:, :, RGB), 3));
    L(:, :, 2) = single(mean(S(:, :, RGB), 3));
    HPMS = single(S);
    save(out, 'L', 'HPMS', 'FLOOR', 'FLOOR_ANY', 'NAMES', '-v7');
    fprintf('%d/%d %s (%.0f s)\n', k, numel(files), name, toc(t0));
end
end

function [I, FL, FA] = hpm_stab(EXP, PAN, sensor, ratio)
% Identical to hpm_stab in quality_ext.m, with floor-activity counts.
h = genMTF(ratio, sensor, size(EXP, 3));
nb = size(EXP, 3);
I = zeros(size(EXP)); FL = zeros(3, nb); FA = zeros(1, nb);
E = max(EXP, 1);
for b = 1:nb
    Pk = (PAN - mean2(PAN)) .* (std2(E(:, :, b)) ./ std2(LPfilterGauss(PAN, ratio))) + mean2(E(:, :, b));
    PL = imfilter(Pk, real(h(:, :, b)), 'replicate');
    PL = interp23tap(imresize(PL, 1/ratio, 'nearest'), ratio);
    I(:, :, b) = E(:, :, b) .* (max(Pk, 1) ./ max(PL, 1));
    a = EXP(:, :, b) < 1; p = Pk < 1; l = PL < 1;
    FL(:, b) = [mean(a(:)); mean(p(:)); mean(l(:))];
    FA(b) = mean(a(:) | p(:) | l(:));
end
end
