function rr_side(listfile, outdir)
% RR_SIDE  Reduced-scale experiment with side-dependent PAN-MS misregistration.
% Each line of listfile: <chip.mat>;<orientation>. On the displaced side of the
% synthetic fault (MS columns >= 129 for 'NS', rows >= 129 for 'EW'), the latent
% MS image is shifted by one latent pixel parallel to the fault in the direction
% of the left-lateral dislocation, before Wald degradation. PAN is not shifted, so
% the MS bands carry an extra offset of one latent MS pixel that PAN does not.
setup_toolbox();
ratio = 4; RGB = [5 3 2];
lines = strtrim(splitlines(fileread(listfile)));
lines = lines(~cellfun(@isempty, lines));
if ~exist(outdir, 'dir'), mkdir(outdir); end
for k = 1:numel(lines)
    parts = strsplit(lines{k}, ';');
    f = parts{1}; orient = parts{2};
    [~, name] = fileparts(f);
    out = fullfile(outdir, sprintf('%s_s1.mat', name));
    if exist(out, 'file'), continue; end
    d = load(f, 'MS', 'PAN', 'SENSOR');
    MS = double(d.MS); PAN = double(d.PAN); sensor = d.SENSOR;
    GT = MS; MSm = MS; h = size(MS, 1); c0 = h / 2 + 1;
    if strcmp(orient, 'NS')
        MSm(1:end-1, c0:end, :) = MS(2:end, c0:end, :);
    else
        MSm(c0:end, 2:end, :) = MS(c0:end, 1:end-1, :);
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
    save(out, 'L', 'NAMES', 'MS_LR', 'PAN_LR', 'sensor', '-v7');
    if mod(k, 50) == 0, fprintf('%d/%d\n', k, numel(lines)); end
end
end
