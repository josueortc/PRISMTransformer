function A = atlas(name, nChannels)
%ATLAS A named atlas from the optional atlas add-on ([] when there is none, or it does not fit).
%
%   A = prismt.internal.atlas(name)               % by name, or by a file's parcellation name
%   A = prismt.internal.atlas(name, nChannels)    % only if it has this many channels
%
%   PRISMT draws channel maps from channel positions (ChannelX, ChannelY) or from a label
%   image stored with the dataset (AtlasImage). Named atlases are an add-on: a package
%   +prismt/+atlas on the path with resolve(name, nChannels) and names(). A has Name,
%   NChannels, X, Y and, when it is a label image, Labels (pixel value k = channel k).
A = [];
if nargin < 2, nChannels = []; end
name = string(name);
if ~isscalar(name) || strlength(name) == 0 || isempty(which('prismt.atlas.resolve')), return; end
A = prismt.atlas.resolve(name, nChannels);
end
