function setup_toolbox()
% Add the folders of the pansharpening toolbox of Vivone et al. (2021) that are used here.
here = fileparts(mfilename('fullpath'));
tb = fullfile(here, '..', 'external', 'vivone_toolbox_2021', 'Pansharpening Toolbox for Distribution');
for f = {'Tools', 'Quality_Indices', 'GS', 'BDSD', 'GLP', 'AWLP'}
    addpath(fullfile(tb, f{1}));
end
end
