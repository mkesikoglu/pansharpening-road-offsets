function run_fusion(listfile, outdir, mode)
% RUN_FUSION  Apply toolbox pansharpening methods to a list of chips.
%   listfile : text file, one input .mat per line (variables MS [h w 8], PAN [4h 4w])
%   outdir   : output folder
%   mode     : 'FR' (full resolution) or 'RR' (Wald reduced-resolution protocol)
% Methods follow Vivone et al. (2015) toolbox v1.3; sensor 'WV2', ratio 4.

here = fileparts(mfilename('fullpath'));
tb = fullfile(here, '..', 'external', 'vivone_toolbox', 'pansharpening_tool_v1.3');
addpath(genpath(tb));
ratio = 4; sensor = 'WV2'; S_bdsd = 128; L = 16;
files = strtrim(splitlines(fileread(listfile)));
files = files(~cellfun(@isempty, files));
if ~exist(outdir, 'dir'), mkdir(outdir); end
for k = 1:numel(files)
    [~, name] = fileparts(files{k});
    out = fullfile(outdir, [name '_' mode '.mat']);
    if exist(out, 'file'), continue; end
    d = load(files{k});
    MS = double(d.MS); PAN = double(d.PAN);
    if strcmp(mode, 'RR')
        GT = MS;
        [MS, PAN] = resize_images(MS, PAN, ratio, sensor);
    end
    EXP = interp23tap(MS, ratio);
    F = struct();
    F.EXP = EXP;
    t = tic; F.GSA = GSA(EXP, PAN, MS, ratio); T.GSA = toc(t);
    t = tic; F.BDSD = BDSD(EXP, PAN, ratio, S_bdsd, sensor); T.BDSD = toc(t);
    t = tic; F.MTF_GLP_HPM = MTF_GLP_HPM(PAN, EXP, sensor, sensor, ratio); T.MTF_GLP_HPM = toc(t);
    t = tic; F.AWLP = AWLP(EXP, PAN, ratio); T.AWLP = toc(t);
    meth = fieldnames(F);
    for m = 1:numel(meth)
        F.(meth{m}) = single(F.(meth{m}));
    end
    if strcmp(mode, 'RR')
        save(out, '-struct', 'F', '-v7.3');
        save(out, 'GT', 'MS', 'PAN', 'T', '-append');
    else
        save(out, '-struct', 'F', '-v7.3');
        save(out, 'T', '-append');
    end
    fprintf('%d/%d %s\n', k, numel(files), name);
end
end
