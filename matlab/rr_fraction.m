function rr_fraction(listfile, outdir)
% RR_FRACTION  Reduced-scale fractional-phase and misregistration experiment.
% Each line of listfile: <chip.mat>;<m>  where m is the misregistration of the
% latent MS image in latent MS pixels (columns, eastward). The latent images are
% degraded by the Wald protocol (MTF filter and decimation by 4 for MS, bicubic
% reduction for PAN), so a dislocation of k latent MS pixels becomes k PAN pixels
% and k/4 MS pixels on the reduced grids. Toolbox products are fused at the
% reduced scale; the latent MS (truth image) is kept as reference.
setup_toolbox();
ratio = 4; RGB = [5 3 2];
lines = strtrim(splitlines(fileread(listfile)));
lines = lines(~cellfun(@isempty, lines));
if ~exist(outdir, 'dir'), mkdir(outdir); end
for k = 1:numel(lines)
    parts = strsplit(lines{k}, ';');
    f = parts{1}; m = str2double(parts{2});
    [~, name] = fileparts(f);
    out = fullfile(outdir, sprintf('%s_m%d.mat', name, m));
    if exist(out, 'file'), continue; end
    d = load(f, 'MS', 'PAN', 'SENSOR', 'EVAL');
    MS = double(d.MS); PAN = double(d.PAN); sensor = d.SENSOR;
    GT = MS;                                   % latent reference (truth geometry)
    if m > 0                                   % shift the latent MS eastward by m pixels
        MSm = MS; MSm(:, m+1:end, :) = MS(:, 1:end-m, :);
        MSm(:, 1:m, :) = repmat(MS(:, 1, :), [1 m 1]);
    else
        MSm = MS;
    end
    [MS_LR, PAN_LR] = resize_images(MSm, PAN, ratio, sensor);
    EXP = interp23tap(MS_LR, ratio);
    F.EXP = EXP;
    F.GSA = GSA(EXP, PAN_LR, MS_LR, ratio);
    F.BDSD_PC = BDSD_PC(EXP, PAN_LR, ratio, sensor);
    F.MTF_GLP_HPM = MTF_GLP_HPM(EXP, PAN_LR, sensor, ratio);
    F.AWLP = AWLP(EXP, PAN_LR, ratio);
    meth = fieldnames(F);
    L = zeros(size(PAN_LR, 1), size(PAN_LR, 2), numel(meth) + 2, 'single');
    L(:, :, 1) = single(PAN_LR);
    L(:, :, 2) = single(mean(GT(:, :, RGB), 3));
    for q = 1:numel(meth)
        L(:, :, q + 2) = single(mean(F.(meth{q})(:, :, RGB), 3));
    end
    NAMES = [{'PAN', 'REF'}, meth'];
    MS_LR = single(MS_LR); PAN_LR = single(PAN_LR);
    if m == 0 && isfield(d, 'EVAL') && d.EVAL == 1
        FUS = struct();
        for q = 1:numel(meth), FUS.(meth{q}) = single(F.(meth{q})); end
        GT = single(GT);
        save(out, 'L', 'NAMES', 'MS_LR', 'PAN_LR', 'FUS', 'GT', 'sensor', '-v7');
    else
        save(out, 'L', 'NAMES', 'MS_LR', 'PAN_LR', 'sensor', '-v7');
    end
    if mod(k, 50) == 0, fprintf('%d/%d\n', k, numel(lines)); end
end
end
