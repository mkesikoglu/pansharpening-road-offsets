function fuse_full(listfile, outdir)
% FUSE_FULL  Toolbox products with all eight bands kept, for the intensity-
% definition sensitivity subset. Output <name>_full.mat with F.<method> (single).
setup_toolbox();
ratio = 4;
files = strtrim(splitlines(fileread(listfile)));
files = files(~cellfun(@isempty, files));
if ~exist(outdir, 'dir'), mkdir(outdir); end
for k = 1:numel(files)
    [~, name] = fileparts(files{k});
    out = fullfile(outdir, [name '_full.mat']);
    if exist(out, 'file'), continue; end
    d = load(files{k}, 'MS', 'PAN', 'SENSOR');
    MS = double(d.MS); PAN = double(d.PAN); sensor = d.SENSOR;
    EXP = interp23tap(MS, ratio);
    F.EXP = single(EXP);
    F.GSA = single(GSA(EXP, PAN, MS, ratio));
    F.BDSD_PC = single(BDSD_PC(EXP, PAN, ratio, sensor));
    F.MTF_GLP_HPM = single(MTF_GLP_HPM(EXP, PAN, sensor, ratio));
    F.AWLP = single(AWLP(EXP, PAN, ratio));
    save(out, 'F', '-v7');
    fprintf('%d/%d %s\n', k, numel(files), name);
end
end
