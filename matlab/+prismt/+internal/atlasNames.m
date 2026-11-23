function names = atlasNames()
%ATLASNAMES Names of the atlases the optional atlas add-on provides (empty without it).
names = strings(1, 0);
if ~isempty(which('prismt.atlas.names')), names = reshape(string(prismt.atlas.names()), 1, []); end
end
