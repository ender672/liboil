require 'mkmf'

# OSX by default keeps libpng in the X11 dirs
if RUBY_PLATFORM =~ /darwin/
  png_idefault = '/usr/X11/include'
  png_ldefault = '/usr/X11/lib'
else
  png_idefault = nil
  png_ldefault = nil
end

dir_config('jpeg')
dir_config('png', png_idefault, png_ldefault)

unless have_header('jpeglib.h')
  abort "libjpeg headers were not found."
end

unless have_library('jpeg')
  abort "libjpeg was not found."
end

unless have_header('png.h')
  abort "libpng headers were not found."
end

unless have_library('png')
  abort "libpng was not found."
end

create_makefile('oil')
