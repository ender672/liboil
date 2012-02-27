Gem::Specification.new do |s|
  s.version = '0.0.1'
  s.files = %w{Rakefile README.rdoc ext/oil.c ext/extconf.rb test/test_jpeg.rb}
  s.name = 'oil'
  s.platform = Gem::Platform::RUBY
  s.require_path = 'ext'
  s.summary = 'Resize JPEG images.'
  s.authors = ['Timothy Elliott']
  s.extensions << 'ext/extconf.rb'
  s.email = 'tle@holymonkey.com'
  s.homepage = 'http://github.com/ender672/oil'
  s.has_rdoc = true
  s.extra_rdoc_files = ['README.rdoc']
  s.test_files = ['test/test_jpeg.rb']
end
