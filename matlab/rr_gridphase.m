function rr_gridphase(listfile, outdir)
% RR_GRIDPHASE  Reduced-scale experiment with the MS grid phase varied at a fixed
% synthetic dislocation (no misregistration).
% Each line of listfile: <chip.mat>;<orientation>;<s>;<dir>
%   s   offset of the degradation window in original (latent) MS pixels, 0..3
%   dir 'a' = offset along strike (rows for 'NS', columns for 'EW'),
%       'c' = offset across strike (columns for 'NS', rows for 'EW').
% The degradation window is WIN x WIN latent MS pixels (4*WIN PAN pixels) and
% starts at latent MS pixel BASE (+ s on the shifted axis). BASE is a multiple of
% the ratio, so at s = 0 the fault lies on a reduced-MS pixel boundary, as in
% rr_fraction.m; an offset s moves the reduced MS grid by s/4 reduced MS pixel
% relative to the fault and the road, while the imposed dislocation k is unchanged.
% The window always lies inside the stored 256 x 256 chip (BASE + 3 + WIN <= 256),
% which itself was cut from a read with a 4-pixel margin, so no pixel outside the
% read is used. Degradation and fusion are identical to rr_fraction.m.
setup_toolbox();
ratio = 4; RGB = [5 3 2]; BASE = 4; WIN = 248;
lines = strtrim(splitlines(fileread(listfile)));
lines = lines(~cellfun(@isempty, lines));
if ~exist(outdir, 'dir'), mkdir(outdir); end
for k = 1:numel(lines)
    parts = strsplit(lines{k}, ';');
    f = parts{1}; orient = parts{2}; s = str2double(parts{3}); dr = parts{4};
    [~, name] = fileparts(f);
    out = fullfile(outdir, sprintf('%s_%s%d.mat', name, dr, s));
    if exist(out, 'file'), continue; end
    d = load(f, 'MS', 'PAN', 'SENSOR');
    MS = double(d.MS); PAN = double(d.PAN); sensor = d.SENSOR;
    alongRows = strcmp(orient, 'NS');          % along-strike axis is rows for NS
    if strcmp(dr, 'c'), alongRows = ~alongRows; end
    r0 = BASE; c0 = BASE;
    if alongRows, r0 = r0 + s; else, c0 = c0 + s; end
    MSw = MS(r0 + (1:WIN), c0 + (1:WIN), :);
    PANw = PAN(ratio * r0 + (1:ratio * WIN), ratio * c0 + (1:ratio * WIN));
    GT = MSw;
    [MS_LR, PAN_LR] = resize_images(MSw, PANw, ratio, sensor);
    EXP = interp23tap(MS_LR, ratio);
    F = struct();
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
    R0 = r0; C0 = c0;                          % window origin in latent MS pixels
    save(out, 'L', 'NAMES', 'MS_LR', 'PAN_LR', 'sensor', 'R0', 'C0', '-v7');
    if mod(k, 50) == 0, fprintf('%d/%d\n', k, numel(lines)); end
end
end
