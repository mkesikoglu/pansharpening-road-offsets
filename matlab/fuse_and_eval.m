function fuse_and_eval(listfile, outdir)
% FUSE_AND_EVAL  Pansharpening with the toolbox of Vivone et al. (2021):
% EXP, GSA, BDSD-PC, MTF-GLP-HPM, AWLP; full-resolution HQNR for these and for
% Z-PNN; Wald-protocol indices when a reduced-resolution input exists.
% Inputs per chip (.mat): MS [h w 8], PAN [4h 4w], ROADMASK [h w] (MS grid),
%   SENSOR, EVAL, optional ZPNN [4h 4w 8]; optional <chip>_RRin.mat with MS_LR,
%   PAN_LR, GT and ZPNN_RR [h w 8].
% Output per chip: <name>_res.mat with luminance images L (4h 4w 7, order in
%   NAMES), RGB of each product (evaluation chips only) and index tables.
setup_toolbox();
ratio = 4; blk = 32; dim_cut = 21;
RGB = [5 3 2];
NAMES = {'PAN', 'EXP', 'GSA', 'BDSD_PC', 'MTF_GLP_HPM', 'AWLP', 'ZPNN'};
files = strtrim(splitlines(fileread(listfile)));
files = files(~cellfun(@isempty, files));
if ~exist(outdir, 'dir'), mkdir(outdir); end
for k = 1:numel(files)
    [~, name] = fileparts(files{k});
    out = fullfile(outdir, [name '_res.mat']);
    if exist(out, 'file'), continue; end
    d = load(files{k});
    sensor = d.SENSOR;
    MS = double(d.MS); PAN = double(d.PAN);
    [F, T] = fuse_all(MS, PAN, ratio, sensor);
    zf = strrep(files{k}, '.mat', '_zpnn.mat'); z = struct();
    if exist(zf, 'file'), z = load(zf); end
    if isfield(z, 'ZPNN'), F.ZPNN = double(z.ZPNN); end
    meth = NAMES(2:end);
    doeval = ~isfield(d, 'EVAL') || d.EVAL == 1;
    % ---- full-resolution indices (HQNR)
    FR = nan(numel(meth), 3);
    for m = 1:numel(meth)
        if ~doeval || ~isfield(F, meth{m}), continue; end
        I = max(F.(meth{m}), 0);
        [q, dl, ds] = HQNR(I, MS, F.EXP, PAN, blk, sensor, ratio);
        FR(m, :) = [dl, ds, q];
    end
    % ---- luminance and RGB products
    L = nan(size(PAN, 1), size(PAN, 2), numel(NAMES), 'single');
    L(:, :, 1) = single(PAN);
    RGBs = struct();
    for m = 1:numel(meth)
        if ~isfield(F, meth{m}), continue; end
        I = F.(meth{m});
        L(:, :, m + 1) = single(mean(I(:, :, RGB), 3));
        if doeval, RGBs.(meth{m}) = single(I(:, :, RGB)); end
    end
    % ---- reduced resolution (Wald protocol)
    RR = nan(numel(meth), 8); rrin = strrep(files{k}, '.mat', '_RRin.mat');
    if doeval && exist(rrin, 'file') && isfield(d, 'ROADMASK')
        r = load(rrin);
        if isfield(z, 'ZPNN_RR')
            [G, ~] = fuse_all(double(r.MS_LR), double(r.PAN_LR), ratio, sensor);
            G.ZPNN = double(z.ZPNN_RR);
            c = dim_cut:size(r.GT, 1) - dim_cut;
            GT = double(r.GT(c, c, :)); mask = logical(d.ROADMASK(c, c));
            for m = 1:numel(meth)
                I = max(G.(meth{m})(c, c, :), 0);
                q2 = q2n(GT, I, blk, blk);
                sam = SAM(GT, I);
                erg = ERGAS(GT, I, ratio);
                scc = SCC(I, GT);
                [samr, ergr] = masked_sam_ergas(GT, I, mask, ratio);
                [samb, ergb] = masked_sam_ergas(GT, I, ~mask, ratio);
                RR(m, :) = [q2, sam, erg, scc, samr, ergr, samb, ergb];
            end
        end
    end
    save(out, 'L', 'RGBs', 'FR', 'RR', 'T', 'NAMES', '-v7');
    fprintf('%d/%d %s\n', k, numel(files), name);
end
end

function [F, T] = fuse_all(MS, PAN, ratio, sensor)
EXP = interp23tap(MS, ratio);
F.EXP = EXP;
t = tic; F.GSA = GSA(EXP, PAN, MS, ratio); T.GSA = toc(t);
t = tic; F.BDSD_PC = BDSD_PC(EXP, PAN, ratio, sensor); T.BDSD_PC = toc(t);
t = tic; F.MTF_GLP_HPM = MTF_GLP_HPM(EXP, PAN, sensor, ratio); T.MTF_GLP_HPM = toc(t);
t = tic; F.AWLP = AWLP(EXP, PAN, ratio); T.AWLP = toc(t);
end

function [sam, erg] = masked_sam_ergas(GT, F, mask, ratio)
% SAM (deg) and ERGAS restricted to the pixels of a mask
nb = size(GT, 3);
X = reshape(GT, [], nb); Y = reshape(F, [], nb); X = X(mask(:), :); Y = Y(mask(:), :);
num = sum(X .* Y, 2); den = sqrt(sum(X .^ 2, 2) .* sum(Y .^ 2, 2));
ok = den > 0;
sam = mean(real(acosd(min(num(ok) ./ den(ok), 1))));
rmse = sqrt(mean((X - Y) .^ 2, 1)); mu = mean(X, 1);
erg = 100 / ratio * sqrt(mean((rmse ./ mu) .^ 2));
end
