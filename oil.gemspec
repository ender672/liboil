Gem::Specification.new do |s|
  s.version = '0.0.3'
  s.test_files = %w{test/helper.rb test/test_jpeg.rb test/test_png.rb}
  s.files = %w{Rakefile README.rdoc ext/oil.c ext/extconf.rb ext/oil.jar} + s.test_files
  s.name = 'oil'
  s.platform = Gem::Platform::RUBY
  s.require_path = 'ext'
  s.summary = 'Oil resizes JPEG and PNG images.'
  s.description = "#{s.summary} It aims for fast performance and low memory use."
  s.authors = ['Timothy Elliott']
  s.extensions << 'ext/extconf.rb'
  s.email = 'tle@holymonkey.com'
  s.homepage = 'http://github.com/ender672/oil'
  s.has_rdoc = true
  s.extra_rdoc_files = ['README.rdoc']
end
