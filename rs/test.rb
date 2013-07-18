require 'oil'
#include Oil

def rs(path_in, path_out, analysis, width, height)
  f = File.open(path_in, 'r')
  pl = Oil.new(f, width, height, interpolation: :cubic, preserve_aspect_ratio: false)
  File.open(path_out, 'w'){ |f| pl.each{ |a| f << a } }
  `~/bin/rscope #{path_out} #{analysis}`
end

rs('pl.png', 'pl_out.png', 'pl_analysis.png', 555, 15)
rs('pd.png', 'pd_out.png', 'pd_analysis.png', 555, 275)
rs('plr.png', 'plr_out.png', 'plr_analysis.png -r', 15, 555)
rs('pdr.png', 'pdr_out.png', 'pdr_analysis.png -r', 275, 555)
