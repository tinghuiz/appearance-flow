Usage:
To render Shapenet models, please refer to the example config file "config.txt"

Parameters:
folder_path = test/  -- root folder of all obj models to render (Remember to put a '/' at the end)
envmap_path = envmaps/envmap.hdr  -- path of the environment map file.
theta_inc = 30  -- theta increment of the views.
phi_inc = 10 -- phi increment of the views.
phi_max = 30 -- maximum phi angle of the views.
output_coord = 0 -- output vertex coordinate images or not.
render_size = 512 -- size for rendering
output_size = 256 -- size for output images (Actual rendering gets downsampled to antialias)
reverse_normals = 1 -- default for Shapenet
brightness = 2 -- overall brightness of output images

Images will be generated in a folder called '[x]_views' under the same directory where the obj model file is located. Where '[x]' is the name of the obj model file excluding the extension.
