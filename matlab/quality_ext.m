function quality_ext(listfile, outfile)
% QUALITY_EXT  Reduced-resolution indices on three band sets (all eight bands,
% seven bands without Coastal, and Red-Green-Blue) for all products, plus a
% numerically stabilised MTF-GLP-HPM in which the interpolated MS image and the
% low-pass PAN are floored at one reflectance unit (1e-4) before the modulation.
% Also records diagnostics of the Coastal band (negative interpolated values,
% minimum denominator).
setup_toolbox();
ratio = 4; blk = 32; dim_cut = 21;
SETS = {1:8, 2:8, [5 3 2]}; SETN = {'B8', 'B7', 'RGB'};
files = strtrim(splitlines(fileread(listfile)));
files = files(~cellfun(@isempty, files));
fid = fopen(outfile, 'w');
fprintf(fid, 'job,product,bandset,Q2n,SAM,ERGAS,SCC\n');
fd = fopen(strrep(outfile, '.csv', '_hpmdiag.csv'), 'w');
fprintf(fd, 'job,coastal_mean,coastal_neg_frac,min_den_b1,max_hpm_b1,max_hpm_b5\n');
for k = 1:numel(files)
    [~, name] = fileparts(files{k});
    rrin = strrep(files{k}, '.mat', '_RRin.mat'); zf = strrep(files{k}, '.mat', '_zpnn.mat');
    if ~exist(rrin, 'file') || ~exist(zf, 'file'), continue; end
    d = load(files{k}, 'SENSOR'); sensor = d.SENSOR;
    r = load(rrin); z = load(zf, 'ZPNN_RR');
    MS = double(r.MS_LR); PAN = double(r.PAN_LR); GT = double(r.GT);
    EXP = interp23tap(MS, ratio);
    F.EXP = EXP; F.GSA = GSA(EXP, PAN, MS, ratio); F.BDSD_PC = BDSD_PC(EXP, PAN, ratio, sensor);
    F.MTF_GLP_HPM = MTF_GLP_HPM(EXP, PAN, sensor, ratio);
    F.MTF_GLP_HPM_stab = hpm_stab(EXP, PAN, sensor, ratio);
    F.AWLP = AWLP(EXP, PAN, ratio); F.ZPNN = double(z.ZPNN_RR);
    c = dim_cut:size(GT, 1) - dim_cut;
    meth = fieldnames(F);
    for q = 1:numel(meth)
        I = max(F.(meth{q})(c, c, :), 0);
        for s = 1:numel(SETS)
            b = SETS{s}; G = GT(c, c, b); J = I(:, :, b);
            fprintf(fid, '%s,%s,%s,%.5f,%.5f,%.5f,%.5f\n', name, meth{q}, SETN{s}, q2n(G, J, blk, blk), ...
                SAM(G, J), ERGAS(G, J, ratio), SCC(J, G));
        end
    end
    % Coastal-band diagnostics of the plain HPM
    h = genMTF(ratio, sensor, size(EXP, 3));
    Pk = (PAN - mean2(PAN)) .* (std2(EXP(:, :, 1)) ./ std2(LPfilterGauss(PAN, ratio))) + mean2(EXP(:, :, 1));
    PL = imfilter(Pk, real(h(:, :, 1)), 'replicate'); PL = interp23tap(imresize(PL, 1/ratio, 'nearest'), ratio);
    fprintf(fd, '%s,%.2f,%.4f,%.3f,%.1f,%.1f\n', name, mean2(GT(:, :, 1)), mean(EXP(:, :, 1) < 0, 'all'), ...
        min(abs(PL(:))), max(F.MTF_GLP_HPM(:, :, 1), [], 'all'), max(F.MTF_GLP_HPM(:, :, 5), [], 'all'));
    if mod(k, 10) == 0, fprintf('%d/%d\n', k, numel(files)); end
end
fclose(fid); fclose(fd);
end

function I = hpm_stab(EXP, PAN, sensor, ratio)
% Toolbox MTF_GLP_HPM (2021) with the interpolated MS, the matched PAN and the
% low-pass PAN floored at 1 reflectance unit; otherwise identical.
h = genMTF(ratio, sensor, size(EXP, 3));
I = zeros(size(EXP));
E = max(EXP, 1);
for b = 1:size(EXP, 3)
    Pk = (PAN - mean2(PAN)) .* (std2(E(:, :, b)) ./ std2(LPfilterGauss(PAN, ratio))) + mean2(E(:, :, b));
    PL = imfilter(Pk, real(h(:, :, b)), 'replicate');
    PL = interp23tap(imresize(PL, 1/ratio, 'nearest'), ratio);
    I(:, :, b) = E(:, :, b) .* (max(Pk, 1) ./ max(PL, 1));
end
end
