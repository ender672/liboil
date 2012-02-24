require 'mkmf'

dir_config('jpeg')

unless have_header('jpeglib.h')
  abort "libjpeg headers were not found."
end

unless have_library('jpeg')
  abort "libjpeg was not found."
end

create_makefile('oil')
